import time
import boto3
from botocore.config import Config
import json
from locust import task, events
from locust.contrib.fasthttp import FastHttpUser
import os
import random

region = os.environ["REGION"]
content_type = os.environ["CONTENT_TYPE"]

class BotoClient:
    def __init__(self, host):
        
        #Consider removing retry logic to get accurate picture of failure in locust
        config = Config(
                        retries = {
                        'max_attempts': 100,
                        'mode': 'standard'
                        }
                    )

        self.sagemaker_client = boto3.client('sagemaker-runtime',config=config)
        self.endpoint_name = host.split('/')[-1]
        self.region = "us-east-1"
        self.content_type = content_type
        self.payload = '[[0,0,0,0,0,0]]'

    def sendPopular(self):

        request_meta = {
            "request_type": "InvokeEndpoint",
            "name": "SageMaker",
            "start_time": time.time(),
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        start_perf_counter = time.perf_counter()
        try:
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=self.payload,
                ContentType=self.content_type,
                TargetModel = f'sklearn-{random.randint(0,10)}.tar.gz'
            )
            response_body = response["Body"].read()
        except Exception as e:
            request_meta['exception'] = e

        request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000

        events.request.fire(**request_meta)
    
    
    def sendRest(self):

        request_meta = {
            "request_type": "InvokeEndpoint",
            "name": "SageMaker",
            "start_time": time.time(),
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        start_perf_counter = time.perf_counter()
   
        try:
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=self.payload,
                ContentType=self.content_type,
                TargetModel = f'sklearn-{random.randint(11,989)}.tar.gz'
            )
            response_body = response["Body"].read()
        except Exception as e:
            request_meta['exception'] = e

        request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000

        events.request.fire(**request_meta)

class BotoUser(FastHttpUser):
    abstract = True
    def __init__(self, env):
        super().__init__(env)
        self.client = BotoClient(self.host)

class MyUser(BotoUser):

    # 80% of traffic to top 10 models
    @task(8)
    def send_request(self):
        self.client.sendPopular()

    @task
    def send_request_major(self):
        self.client.sendRest()