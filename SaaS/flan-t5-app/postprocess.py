import argparse
import logging
import os

import boto3

import jinja2
from pathlib import Path

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def write_serving_file(flan_model_dir, s3_url):
    file_content = "engine=Python\noption.tensor_parallel_degree=2\noption.s3url={{ s3url }}"
    
    with open(f'{flan_model_dir}/serving.properties', 'w') as file:
        file.write(file_content)

    # we plug in the appropriate model location into our `serving.properties` file based on the region in which this notebook is running
    jinja_env = jinja2.Environment()
    template = jinja_env.from_string(Path(f"{flan_model_dir}/serving.properties").open().read())
    Path(f"{flan_model_dir}/serving.properties").open("w").write(template.render(s3url=s3_url))
    
def write_model_file(flan_model_dir):
    file_content = """
from djl_python import Input, Output
import deepspeed
import torch
import logging
import math
import os
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

def load_model(properties):
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    logging.info(f"Loading model in {model_location}")
    logging.info(os.system(f"ls {model_location}"))
    model_name = "google/flan-t5-xxl"

    config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_location, load_in_8bit=True)

    state_dict = torch.load(model_location + "/flan_t5_weights.pt")

    model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path = model_name,
            config=config,
            state_dict=state_dict,
            device_map="balanced_low_0", 
            load_in_8bit=True,
            cache_dir="/tmp"
        )
    model.requires_grad_(False)
    model.eval()

    return model, tokenizer

model = None
tokenizer = None
generator = None

def run_inference(model, tokenizer, data, params):
    generate_kwargs = params
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer.batch_encode_plus(data,
                                               return_tensors="pt",
                                               padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
    outputs = model.generate(**input_tokens, **generate_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def handle(inputs: Input):
    global model, tokenizer
    if not model:
        model, tokenizer = load_model(inputs.get_properties())

    if inputs.is_empty():
        return None
    data = inputs.get_as_json()

    input_sentences = data["inputs"]
    params = data["parameters"]

    outputs = run_inference(model, tokenizer, input_sentences, params)
    result = {"outputs": outputs}
    return Output().add_as_json(result)
    """

    with open(f'{flan_model_dir}/model.py', 'w') as file:
        file.write(file_content)

def write_req_file(flan_model_dir):
    with open(f"{flan_model_dir}/requirements.txt", "w") as file:
        file.write("bitsandbytes==0.38.1\n")
        file.write("accelerate==0.18.0\n")
        file.write("transformers==4.28.1\n")
    
    
if __name__ == "__main__":
    logger.info("Starting postprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", type=str, required=True)
    parser.add_argument("--bucket-name", type=str, required=True)
    parser.add_argument("--tenant-tier", type=str, required=True)
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--app-id", type=str, required=True)
    parser.add_argument("--checkpoint-s3-path", type=str, required=True)

    args = parser.parse_args()

    s3_url = args.checkpoint_s3_path

    repackaged_location = "/opt/ml/processing/model/repackaged"
    
    os.system(f'mkdir -p {repackaged_location}/code_flant5_accelerate/')
    
    write_serving_file(f'{repackaged_location}/code_flant5_accelerate',s3_url )
    write_model_file(f'{repackaged_location}/code_flant5_accelerate')
    write_req_file(f'{repackaged_location}/code_flant5_accelerate')
    
    os.system(f'tar -czvf {repackaged_location}/model.tar.gz -C {repackaged_location} code_flant5_accelerate')
    
    s3 = boto3.resource("s3")
    file_name = f"{args.tenant_id}-{args.app_id}.model.{args.model_version}.tar.gz"

    if args.tenant_tier == "Bronze":
        logger.info("Starting copying model artifacts from the local file to a dedicated s3 prefix")
        object_key = f"{args.tenant_id}/model_artifacts/{file_name}"
    elif args.tenant_tier == "Gold":
        logger.info("Starting copying model artifacts from the local file to a dedicated s3 prefix")
        object_key = f"{args.tenant_id}/model_artifacts/{file_name}"
    else:
        raise ValueError("Invalid tenant tier. The tenant tier must be either 'Bronze' or 'Gold'.")

    s3.Bucket(args.bucket_name).upload_file(f'{repackaged_location}/model.tar.gz', object_key)
    print(f's3://{args.bucket_name}/{object_key}')
    logger.info("Copying model artifacts ended ")
              
