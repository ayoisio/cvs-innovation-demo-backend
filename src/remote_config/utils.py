import time
import json
import os
from google.cloud import storage
from functools import wraps
from flask import request, Response
from typing import Optional

import google.auth
import google.auth.transport.requests
import requests

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
BASE_URL = 'https://firebaseremoteconfig.googleapis.com'
REMOTE_CONFIG_ENDPOINT = f'v1/projects/{PROJECT_ID}/remoteConfig'
REMOTE_CONFIG_URL = f'{BASE_URL}/{REMOTE_CONFIG_ENDPOINT}'

# Cache for Remote Config values
remote_config_cache = {}
remote_config_last_fetch = 0
REMOTE_CONFIG_CACHE_DURATION = 3600  # 1 hour

# Cache for GCS prompts
gcs_prompt_cache = {}


def get_access_token():
    credentials, project_id = google.auth.default(
        scopes=['https://www.googleapis.com/auth/firebase.remoteconfig']
    )
    # Set the quota project
    credentials = credentials.with_quota_project(project_id)
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token


def fetch_remote_config():
    headers = {
        'Authorization': f'Bearer {get_access_token()}',
        'Accept-Encoding': 'gzip',
        'X-goog-user-project': PROJECT_ID
    }
    resp = requests.get(REMOTE_CONFIG_URL, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        print('Unable to get template')
        print(resp.text)
        return None


def get_remote_config_value(parameter_group, key):
    global remote_config_last_fetch
    current_time = time.time()

    if current_time - remote_config_last_fetch > REMOTE_CONFIG_CACHE_DURATION:
        config = fetch_remote_config()
        if config and 'parameterGroups' in config:
            for group_name, group_data in config['parameterGroups'].items():
                if 'parameters' in group_data:
                    for param_key, param_value in group_data['parameters'].items():
                        if param_value['valueType'] == 'JSON':
                            remote_config_cache[f"{group_name}:{param_key}"] = json.loads(param_value['defaultValue']['value'])
                        else:
                            remote_config_cache[f"{group_name}:{param_key}"] = param_value['defaultValue']['value']
        remote_config_last_fetch = current_time

    cache_key = f"{parameter_group}:{key}"
    return remote_config_cache.get(cache_key)


def get_gcs_prompt(file_name, bucket_name: Optional[str] = None):
    if not bucket_name:
        bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET")

    if file_name not in gcs_prompt_cache:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"shared/prompts/{file_name}")
        gcs_prompt_cache[file_name] = blob.download_as_text()
    return gcs_prompt_cache[file_name]


def generate_decorator(parameter_group, endpoint_key):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.json or {}

            config = get_remote_config_value(parameter_group, endpoint_key)
            if not config:
                return Response("Configuration not found", status=404)

            file_name = config['fileName']
            model = config['externalModel']
            url = config['url']

            bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET")
            prompt = get_gcs_prompt(file_name, bucket_name=bucket_name)

            # Add any additional processing of the prompt here if needed

            return f(prompt=prompt, model=model, url=url, data=data, *args, **kwargs)
        return decorated_function
    return decorator
