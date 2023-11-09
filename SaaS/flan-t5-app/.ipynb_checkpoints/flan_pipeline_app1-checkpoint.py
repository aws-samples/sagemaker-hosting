##################
# Pipeline DAG for App1
##################
import os

import boto3
import logging
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import (
    ConditionLessThanOrEqualTo,
    ConditionEquals,
)
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

from botocore.exceptions import ClientError

from sagemaker.workflow.step_collections import RegisterModel

import sagemaker
from sagemaker.pytorch import PyTorch

from sagemaker.workflow.steps import CacheConfig

from sagemaker.workflow.functions import Join


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def resolve_ecr_uri_from_image_versions(sagemaker_session, image_versions, image_name):
    """ Gets ECR URI from image versions
    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_versions: list of the image versions
        image_name: Name of the image

    Returns:
        ECR URI of the image version
    """

    #Fetch image details to get the Base Image URI
    for image_version in image_versions:
        if image_version['ImageVersionStatus'] == 'CREATED':
            image_arn = image_version["ImageVersionArn"]
            version = image_version["Version"]
            logger.info(f"Identified the latest image version: {image_arn}")
            response = sagemaker_session.sagemaker_client.describe_image_version(
                ImageName=image_name,
                Version=version
            )
            return response['ContainerImage']
    return None

def resolve_ecr_uri(sagemaker_session, image_arn):
    """Gets the ECR URI from the image name

    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_name: name of the image

    Returns:
        ECR URI of the latest image version
    """

    # Fetching image name from image_arn (^arn:aws(-[\w]+)*:sagemaker:.+:[0-9]{12}:image/[a-z0-9]([-.]?[a-z0-9])*$)
    image_name = image_arn.partition("image/")[2]
    try:
        # Fetch the image versions
        next_token=''
        while True:
            response = sagemaker_session.sagemaker_client.list_image_versions(
                ImageName=image_name,
                MaxResults=100,
                SortBy='VERSION',
                SortOrder='DESCENDING',
                NextToken=next_token
            )
            ecr_uri = resolve_ecr_uri_from_image_versions(sagemaker_session, response['ImageVersions'], image_name)
            if "NextToken" in response:
                next_token = response["NextToken"]

            if ecr_uri is not None:
                return ecr_uri

        # Return error if no versions of the image found
        error_message = (
            f"No image version found for image name: {image_name}"
            )
        logger.error(error_message)
        raise Exception(error_message)

    except (ClientError, sagemaker_session.sagemaker_client.exceptions.ResourceNotFound) as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

def get_pipeline(
    tenant_id=None,
    region=None,
    role=None,
    default_bucket=None,
    sagemaker_project_arn=None,
    pipeline_name=None, # Add tenant_id?
    base_job_prefix=None, # Add tenant_id?
    project_id="SageMakerProjectId",
    processing_instance_type="ml.m5.xlarge",
    processing_instance_count=1,
    training_instance_type=None,  
    training_instance_count=1,
    checkpoint_s3_path=None
):
    # pipeline_name="pipeline-{}-app1".format(tenant_id) # Add tenant_id
    # base_job_prefix="pipeline-{}-app1".format(tenant_id) # Add tenant_id
    
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=None
    )
    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount", default_value=None
    )
    train_data_path = ParameterString(
        name="TrainDataPath", default_value=None
    )
    val_data_path = ParameterString(
        name="ValidationDataPath", default_value=None
    )
    tenant_id = ParameterString(
        name="TenantID", default_value=None,
    )
    app_id = ParameterString(
        name="AppID", default_value=None,
    )    
    tenant_tier = ParameterString(
        name="TenantTier", default_value=None,
    )
    model_version = ParameterString(
        name="ModelVersion", default_value=None,
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.p4d.24xlarge"
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value=None
    )
    checkpoint_s3_path = ParameterString(
        name="CheckpointS3Path", default_value=None
    )
    
    checkpoint_dir = "/opt/ml/checkpoints"

    smp_options = {
        "enabled":True,
        "parameters": {                        # Required
            "pipeline_parallel_degree": 1,     # Required
            "ddp": True,
            # parameters for sharded data parallelism
            "sharded_data_parallel_degree": 16,              # Add this to activate sharded data parallelism
            "partitions":1,
            "bf16":True,
            "skip_tracing": True
        }
    }

    mpi_options = {
        "enabled" : True,                      # Required
        "processes_per_host" : 8               # Required
    }
    # launch with smp
    
    hyperparameters = {}

    hyperparameters["model_name_or_path"] = "google/flan-t5-xxl"
    hyperparameters["train_file"] = "/opt/ml/input/data/train/train.csv"
    hyperparameters["validation_file"] = "/opt/ml/input/data/validation/val.csv"
    hyperparameters["per_device_train_batch_size"] = 8
    hyperparameters["per_device_eval_batch_size"] = 8
    hyperparameters["block_size"] = 2048
    hyperparameters["checkpoint_dir"] = "/opt/ml/checkpoints"
    hyperparameters["num_train_epochs"] = 1
    hyperparameters["max_train_steps"] = 250

    estimator = PyTorch(
        base_job_name=base_job_prefix,
        source_dir="./training_scripts",
        entry_point="train.py",
        role=role,
        framework_version="1.13.1",
        py_version="py39", 
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        hyperparameters=hyperparameters,
        checkpoint_local_path=checkpoint_dir,   
        checkpoint_s3_uri=checkpoint_s3_path,
        disable_profiler=True,
        keep_alive_period_in_seconds=600,
        debugger_hook_config=False,
        distribution={
            "smdistributed": {"modelparallel": smp_options},
            "mpi": mpi_options
        }
    )
    
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    step_train = TrainingStep(
        name=f"{base_job_prefix}-FLAN-T5",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data = Join(on = '/', values = [f"s3://{default_bucket}/sample-data", tenant_id,  app_id,  train_data_path]),

                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data = Join(on = '/', values = [f"s3://{default_bucket}/sample-data", tenant_id,  app_id,  val_data_path]),
                content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )
    
    
     # processing step for Bronze Tier Only. Copy the model artifacts from the S3 bucket to a central bucket accessible by SageMaker Multi-Model-Endpoint (MME)
    sklearn_processor = SKLearnProcessor(
        framework_version="0.20.0", 
        role=role, 
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=pipeline_session,
        command=["python3"],
        base_job_name=f"{base_job_prefix}/sklearn-preprocess",
    )
        
    step_args = sklearn_processor.run(
        code=os.path.join(BASE_DIR, "postprocess.py"),
        arguments=["--bucket-name",default_bucket, "--tenant-id",tenant_id, "--tenant-tier", tenant_tier, "--region",region,"--model-version",model_version, "--app-id", app_id, "--checkpoint-s3-path", checkpoint_s3_path ],
    )
    
    step_post_process = ProcessingStep(
        name=f"{base_job_prefix}-Create_LMI_Model_Artifacts_To_S3_Folder",
        step_args=step_args,
        depends_on=[step_train]
    )
    
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_count,
            training_instance_type,
            train_data_path,
            val_data_path,
            tenant_id,
            tenant_tier,
            model_version,
            app_id,
            checkpoint_s3_path
        ],
        steps=[step_train, step_post_process],
        sagemaker_session=pipeline_session,
    )
    return pipeline