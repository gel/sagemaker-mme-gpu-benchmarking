import boto3
from botocore.config import Config
import os
import sys
import json
from locust import User, task, between, events, LoadTestShape

import numpy as np
from PIL import Image
from pathlib import Path
import random
import time

# How to use
# 1. install locust & boto3
#   pip install locust boto3
# 2. run benchmark via cli
# with UI
# Since we are using a custom client for the request we need to define the "Host" as -.
#   ENDPOINT_NAME="distilbert-base-uncased-distilled-squad-6493832c-767d-4cdb-a9a" locust -f locust_benchmark_sm.py
#
# headless
# --users  Number of concurrent Locust users
# --spawn-rate  The rate per second in which users are spawned until num users
# --run-time duration of test
#   ENDPOINT_NAME="distilbert-base-uncased-distilled-squad-6493832c-767d-4cdb-a9a" locust -f locust_benchmark_sm.py \
#       --users 60 \
#       --spawn-rate 1 \
#       --run-time 360s \
#       --headless


# locust -f locust_benchmark_sm.py \
#       --users 60 \
#       --spawn-rate 1 \
#       --run-time 360s \
#       --headless

content_type = "application/octet-stream"


class SageMakerClient:

    def __init__(self):
        super().__init__()
        
        self.session = boto3.session.Session()
        self.payload = globals()["payload"]

        
        self.client=self.session.client("sagemaker-runtime")
        # self.cv_payload = json.dumps(cv_payload)
        # self.nlp_payload = json.dumps(cv_payload)
        self.content_type = content_type

    def send(self, endpoint_name, use_case, model_name, model_count):
        
        
        request_meta = {
            "request_type": "InvokeEndpoint",
            "name": model_name,
            "start_time": time.time(),
            "num_models": model_count,
            "response_length": 0,
            "response": None,
            "context": {},
            "exception": None,
        }
        
        start_perf_counter = time.perf_counter()
        
            
        try:
            response = self.client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=self.payload,
                ContentType=self.content_type,
                TargetModel="{0}-v{1}.tar.gz".format(model_name, random.randint(0, model_count-1)),
            )
            # print(response)
        except Exception as e:
            request_meta['exception'] = e
        
        request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000

        events.request.fire(**request_meta)


class SageMakerUser(User):
    abstract = True
    
    @events.init_command_line_parser.add_listener
    def _(parser):
        parser.add_argument("--endpoint-name", type=str, default="mme-cv-benchmark-pt", help="sagemaker endpoint you want to invoke")
        parser.add_argument("--model-name", type=str, default="vgg16", help="name of your model")
        parser.add_argument("--use-case", choices=["nlp", "cv"], help="the model's use case", required=True)
        parser.add_argument("--payload", type=str, help="json file with sample payload for model benchmarking", required=True)
        parser.add_argument("--model-count", type=int, default=5, help="how many models you want to invoke")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        payload_path = Path(self.environment.parsed_options.payload)
        globals()["payload"] = payload_path.open("r").read()
        self.client = SageMakerClient()
        
class SimpleSendRequest(SageMakerUser):
    wait_time = between(0.05, 0.5)

    @task
    def send_request(self):
        endpoint_name = self.environment.parsed_options.endpoint_name
        use_case = self.environment.parsed_options.use_case
        model_name = self.environment.parsed_options.model_name
        model_count = self.environment.parsed_options.model_count        
        
        self.client.send(endpoint_name, use_case, model_name, model_count)

class StagesShape(LoadTestShape):

    stages = [
        {"duration": 120, "users": 2, "spawn_rate": 2},
        {"duration": 240, "users": 4, "spawn_rate": 2},
        {"duration": 360, "users": 6, "spawn_rate": 2},
        {"duration": 480, "users": 8, "spawn_rate": 2},
        {"duration": 600, "users": 10, "spawn_rate": 2},
        {"duration": 720, "users": 12, "spawn_rate": 2},
        {"duration": 840, "users": 14, "spawn_rate": 2},
        {"duration": 960, "users": 16, "spawn_rate": 2},
        {"duration": 1080, "users": 18, "spawn_rate": 2},
        {"duration": 1200, "users": 20, "spawn_rate": 2},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data

        return None