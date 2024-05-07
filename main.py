import os 
import json 
import sys
import boto3
print("import success")


prompt = """
Hey this has been a great expereience.Can you share some information about linear algebra?
"""

bedrock_service = boto3.client(service_name = "bedrock-runtime")

payload = {
    "prompt":"[INST]"+prompt+"[/INST]",
    "max_gen_len":512,
    "temperature":0.3,
    "top_p":0.9
}



body = json.dumps(payload)

model_id = "meta.llama3-70b-instruct-v1:0"

response = bedrock_service.invoke_model(
    body = body,
    modelId = model_id,
    accept="application/json",
    contentType = "application/json"
)

response_body = json.loads(response.get("body").read())
response_text = response_body["generation"]
print(response_text)



