import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
from botocore.exceptions import ClientError

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

os.system("pip install xgboost==1.1.0")
import pickle
import xgboost as xgb
import numpy as np
import io



def write_config_file(fil_model_dir):
    USE_GPU = False


    # Maximum size in bytes for input and output arrays. If you are
    # using Triton 21.11 or higher, all memory allocations will make
    # use of Triton's memory pool, which has a default size of
    # 67_108_864 bytes
    MAX_MEMORY_BYTES = 60_000_000
    NUM_FEATURES = 6
    NUM_CLASSES = 2
    bytes_per_sample = (NUM_FEATURES + NUM_CLASSES) * 4
    max_batch_size = MAX_MEMORY_BYTES // bytes_per_sample

    IS_CLASSIFIER = False
    model_format = "xgboost_json"

    # Select deployment hardware (GPU or CPU)
    if USE_GPU:
        instance_kind = "KIND_GPU"
    else:
        instance_kind = "KIND_CPU"

    # whether the model is doing classification or regression
    if IS_CLASSIFIER:
        classifier_string = "true"
    else:
        classifier_string = "false"

    # whether to predict probabilites or not
    predict_proba = False

    if predict_proba:
        predict_proba_string = "true"
    else:
        predict_proba_string = "false"

    config_text = f"""backend: "fil"
    max_batch_size: {max_batch_size}
    input [                                 
     {{  
        name: "input__0"
        data_type: TYPE_FP32
        dims: [ {NUM_FEATURES} ]                    
      }} 
    ]
    output [
     {{
        name: "output__0"
        data_type: TYPE_FP32
        dims: [ 1 ]
      }}
    ]
    instance_group [{{ kind: {instance_kind} }}]
    parameters [
      {{
        key: "model_type"
        value: {{ string_value: "{model_format}" }}
      }},
      {{
        key: "predict_proba"
        value: {{ string_value: "{predict_proba_string}" }}
      }},
      {{
        key: "output_class"
        value: {{ string_value: "{classifier_string}" }}
      }},
      {{
        key: "threshold"
        value: {{ string_value: "0.5" }}
      }},
      {{
        key: "storage_type"
        value: {{ string_value: "AUTO" }}
      }},
      {{
        key: "use_experimental_optimizations"
        value: {{ string_value: "true" }}
      }}
    ]

    dynamic_batching {{}}"""
    
    os.makedirs(os.path.dirname(fil_model_dir), exist_ok=True)
    config_path = os.path.join(fil_model_dir, "config.pbtxt")
    with open(config_path, "w") as file_:
        file_.write(config_text)


def covert_xgboost_model(model_dir):
    # Load the saved XGBoost model
    with open(f'{model_dir}/xgboost-model', 'rb') as f:
        model = pickle.load(f)
        
    print(model.__dict__)
    print("Saving model in JSON format required by FIL on Triton")
    model.save_model(f'{model_dir}/xgboost.json')
    os.system(f'rm {model_dir}/xgboost-model')
    print("Final location:")
    print(os.system(f'ls {model_dir}/'))


if __name__ == "__main__":
    logger.info("Starting postprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", type=str, required=True)
    parser.add_argument("--bucket-name", type=str, required=True)
    parser.add_argument("--tenant-tier", type=str, required=True)
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--app-id", type=str, required=True)

    args = parser.parse_args()
    local_file = "/opt/ml/processing/model/model.tar.gz"


    repackaged_location = "/opt/ml/processing/model/repackaged"
    os.system(f'mkdir -p {repackaged_location}/fil/1/')
    os.system(f'tar -xzf {local_file} -C {repackaged_location}/fil/1/')

    write_config_file(f'{repackaged_location}/fil/')
    covert_xgboost_model(f'{repackaged_location}/fil/1')
    
    
    os.system(f'tar -czvf {repackaged_location}/model.tar.gz -C {repackaged_location} fil')
    
    s3 = boto3.resource("s3")
    file_name = f"{args.tenant_id}-{args.app_id}.model.{args.model_version}.tar.gz"

    if args.tenant_tier == "Bronze":
        logger.info("Starting copying model artifacts from the local file to the shared s3 prefix (model_artifacts_mme).")
        logger.info("SageMaker Multi-model endpoint will use this bucket to load and unload its models")
        object_key = f"model_artifacts_mme/{file_name}"
    elif args.tenant_tier == "Gold":
        logger.info("Starting copying model artifacts from the local file to a dedicated s3 prefix")
        object_key = f"{args.tenant_id}/model_artifacts/{file_name}"
    else:
        raise ValueError("Invalid tenant tier. The tenant tier must be either 'Bronze' or 'Gold'.")

    s3.Bucket(args.bucket_name).upload_file(f'{repackaged_location}/model.tar.gz', object_key)
    logger.info("Copying model artifacts ended")
              
