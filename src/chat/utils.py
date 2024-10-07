import json
import jsonpickle
import mimetypes
import os
import re
from google.cloud import storage
from typing import Any, Optional
from uuid import uuid4
from vertexai.generative_models import FunctionDeclaration

import src.remote_config.utils as remote_config_utils


def clean_text(text: str):
    """Clean text."""
    return re.sub(r'\s+', ' ', text)


def get_config_and_prompt(config_key: str):
    """Helper function to get configuration and prompt."""
    config = remote_config_utils.get_remote_config_value("Prompts", config_key)
    if not config:
        raise ValueError(f"Configuration not found for {config_key}")

    file_name = config['fileName']
    prompt = remote_config_utils.get_gcs_prompt(file_name)
    return prompt


def create_function_declaration(name: str, description_key: str, parameters_key: str):
    """Helper function to create a FunctionDeclaration."""
    description = get_config_and_prompt(description_key)
    parameters = json.loads(get_config_and_prompt(parameters_key))
    return FunctionDeclaration(
        name=name,
        description=description,
        parameters=parameters,
    )


def get_chat_history(user_id: str, chat_history_id: Optional[str] = None):
    """Fetch chat history from Google Cloud Storage."""
    if not chat_history_id:
        return []

    client = storage.Client()
    bucket = client.bucket(os.getenv("GOOGLE_CLOUD_BUCKET"))
    blob = bucket.blob(f"users/{user_id}/chats/{chat_history_id}.txt")

    if blob.exists():
        return jsonpickle.decode(blob.download_as_text())
    else:
        return []


def save_chat_history(user_id: str, chat_history_id: str, chat_history: Any):
    """Save updated chat history back to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(os.getenv("GOOGLE_CLOUD_BUCKET"))
    blob = bucket.blob(f"users/{user_id}/chats/{chat_history_id}.txt")

    if isinstance(chat_history, list):
        encoded_chat_history = jsonpickle.encode(chat_history, True)
    else:
        encoded_chat_history = chat_history

    # Convert the chat history to JSON and upload
    blob.upload_from_string(encoded_chat_history, content_type='text/plain')


def upload_image_to_gcs(user_id: str, chat_history_id: str, image_bytes: bytes, image_mime_type: str):
    """Uploads image bytes to GCS bucket with a random UUID as the file name."""
    storage_client = storage.Client()
    bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET")
    bucket = storage_client.bucket(bucket_name)

    # Get the file extension from the MIME type
    extension = mimetypes.guess_extension(image_mime_type)
    if not extension:
        raise ValueError(f"Unsupported MIME type: {image_mime_type}")

    # Generate a unique name with the correct extension
    image_name = f"users/{user_id}/chats/{chat_history_id}/{uuid4()}{extension}"
    blob = bucket.blob(image_name)
    blob.upload_from_string(image_bytes, content_type=image_mime_type)
    return f"gs://{bucket_name}/{image_name}"
