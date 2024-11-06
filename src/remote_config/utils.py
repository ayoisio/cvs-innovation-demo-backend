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

# Configuration constants for Firebase Remote Config
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
BASE_URL = 'https://firebaseremoteconfig.googleapis.com'
REMOTE_CONFIG_ENDPOINT = f'v1/projects/{PROJECT_ID}/remoteConfig'
REMOTE_CONFIG_URL = f'{BASE_URL}/{REMOTE_CONFIG_ENDPOINT}'

# Cache configuration
remote_config_cache = {}  # Stores Remote Config parameter values
remote_config_last_fetch = 0  # Timestamp of last Remote Config fetch
REMOTE_CONFIG_CACHE_DURATION = 3600  # Cache duration in seconds (1 hour)

# Cache for GCS prompts to avoid repeated storage access
gcs_prompt_cache = {}


def get_access_token():
    """
    Obtain a valid access token for Firebase Remote Config API.

    Uses Google default credentials to get an access token with appropriate scopes
    for accessing Firebase Remote Config.

    Returns:
        str: The access token for authentication

    Note:
        Requires the GOOGLE_CLOUD_PROJECT environment variable to be set
    """
    credentials, project_id = google.auth.default(
        scopes=['https://www.googleapis.com/auth/firebase.remoteconfig']
    )
    # Set the quota project to ensure proper billing
    credentials = credentials.with_quota_project(project_id)
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token


def fetch_remote_config():
    """
    Fetch the current Remote Config template from Firebase.

    Makes an authenticated request to the Firebase Remote Config API to retrieve
    the current configuration template.

    Returns:
        dict: The Remote Config template if successful, None otherwise

    Note:
        Prints error messages to console if the request fails
    """
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
    """
    Retrieve a value from Remote Config with caching.

    Fetches and caches Remote Config values, refreshing the cache when expired.
    Supports both JSON and string values from the Remote Config template.

    Args:
        parameter_group (str): The parameter group name in Remote Config
        key (str): The key within the parameter group

    Returns:
        Any: The configuration value for the specified key, or None if not found

    Note:
        - Uses a global cache that expires after REMOTE_CONFIG_CACHE_DURATION seconds
        - JSON values are automatically parsed into Python objects
    """
    global remote_config_last_fetch
    current_time = time.time()

    # Check if cache needs refresh
    if current_time - remote_config_last_fetch > REMOTE_CONFIG_CACHE_DURATION:
        config = fetch_remote_config()
        if config and 'parameterGroups' in config:
            for group_name, group_data in config['parameterGroups'].items():
                if 'parameters' in group_data:
                    for param_key, param_value in group_data['parameters'].items():
                        cache_key = f"{group_name}:{param_key}"
                        if param_value['valueType'] == 'JSON':
                            remote_config_cache[cache_key] = json.loads(param_value['defaultValue']['value'])
                        else:
                            remote_config_cache[cache_key] = param_value['defaultValue']['value']
        remote_config_last_fetch = current_time

    cache_key = f"{parameter_group}:{key}"
    return remote_config_cache.get(cache_key)


def get_gcs_prompt(file_name, bucket_name: Optional[str] = None):
    """
    Retrieve a prompt file from Google Cloud Storage with caching.

    Args:
        file_name (str): Name of the prompt file to retrieve
        bucket_name (Optional[str]): Custom bucket name, defaults to GOOGLE_CLOUD_BUCKET env var

    Returns:
        str: Content of the prompt file

    Note:
        - Caches prompt content in memory to avoid repeated storage access
        - Prompt files should be stored in 'shared/prompts/' directory in the bucket
    """
    if not bucket_name:
        bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET")

    if file_name not in gcs_prompt_cache:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"shared/prompts/{file_name}")
        gcs_prompt_cache[file_name] = blob.download_as_text()
    return gcs_prompt_cache[file_name]


def generate_decorator(parameter_group, endpoint_key):
    """
    Generate a decorator for endpoint configuration and prompt handling.

    Creates a decorator that automatically fetches configuration and prompts
    for endpoint handlers based on Remote Config parameters.

    Args:
        parameter_group (str): Parameter group name in Remote Config
        endpoint_key (str): Configuration key for the endpoint

    Returns:
        callable: A decorator function that wraps the endpoint handler

    Note:
        The decorated function must accept the following parameters:
        - prompt: The prompt text from GCS
        - model: The model configuration from Remote Config
        - url: The endpoint URL from Remote Config
        - data: The request JSON data
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get request data with fallback to empty dict
            data = request.json or {}

            # Fetch endpoint configuration from Remote Config
            config = get_remote_config_value(parameter_group, endpoint_key)
            if not config:
                return Response("Configuration not found", status=404)

            # Extract configuration values
            file_name = config['fileName']
            model = config['externalModel']
            url = config['url']

            # Get prompt from GCS using environment bucket
            bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET")
            prompt = get_gcs_prompt(file_name, bucket_name=bucket_name)

            # Call the wrapped function with extracted configuration
            return f(prompt=prompt, model=model, url=url, data=data, *args, **kwargs)

        return decorated_function

    return decorator
