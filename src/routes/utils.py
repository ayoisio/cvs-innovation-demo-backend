import json
import os
from google.cloud import firestore, storage
from google.cloud import tasks_v2
from flask import jsonify
from firebase_admin import auth
from typing import Any, Dict, List, Optional
from uuid import uuid4


def verify_auth_token(request):
    """
    Verify the authentication token from the request headers.

    Args:
        request: The incoming HTTP request object containing the authorization header

    Returns:
        tuple: A tuple containing either:
            - The decoded token information if verification is successful
            - A JSON response with error message and appropriate HTTP status code (401) if verification fails

    Raises:
        IndexError: When the authorization header format is invalid
        InvalidIdTokenError: When the provided token is invalid
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'No authorization token provided'}), 401

    try:
        # Extract the token from 'Bearer <token>' format
        auth_token = auth_header.split(' ')[1]
        return auth.verify_id_token(auth_token)
    except IndexError:
        return jsonify({'error': 'Invalid authorization header format'}), 401
    except auth.InvalidIdTokenError:
        return jsonify({'error': 'Invalid authorization token'}), 401


def parse_json_data(request):
    """
    Parse JSON data from different possible locations in the request.

    Checks for JSON data in the following order:
    1. In request.files as a file upload
    2. In request.form as a form field
    3. Directly in request.json

    Args:
        request: The incoming HTTP request object

    Returns:
        dict: The parsed JSON data or an empty dictionary if no JSON data is found
    """
    if 'json' in request.files:
        json_file = request.files['json']
        json_data = json_file.read().decode('utf-8')
        return json.loads(json_data) if json_data else {}
    elif 'json' in request.form:
        json_data = request.form.get('json')
        return json.loads(json_data) if json_data else {}
    else:
        return request.json or {}


def check_uploaded_media(user_id, chat_id, message_id):
    """
    Check for uploaded media files in Google Cloud Storage for a specific message.

    Args:
        user_id (str): The ID of the user
        chat_id (str): The ID of the chat
        message_id (str): The ID of the message

    Returns:
        list: List of dictionaries containing metadata for each uploaded file:
            - fileName: Name of the uploaded file
            - fileMimeType: MIME type of the file
            - fileSize: Size of the file in bytes
            - gcsPath: Complete Google Cloud Storage path
    """
    bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Construct the prefix for the user's media files
    prefix = f"users/{user_id}/chats/{chat_id}/uploadedMedia/{message_id}/"
    blobs = bucket.list_blobs(prefix=prefix)

    uploaded_files = []
    for blob in blobs:
        uploaded_files.append({
            "fileName": blob.name.split('/')[-1],
            "fileMimeType": blob.content_type,
            "fileSize": blob.size,
            "gcsPath": f"gs://{bucket_name}/{blob.name}"
        })

    return uploaded_files


def create_cloud_task(url, payload, **kwargs):
    """
    Create a Cloud Tasks task to handle asynchronous processing.

    Args:
        url (str): The endpoint URL where the task should be sent
        payload (dict): The primary payload data for the task
        **kwargs: Additional keyword arguments to be merged with the payload

    Returns:
        str: The name/ID of the created task

    Note:
        Requires the following environment variables:
        - GOOGLE_CLOUD_PROJECT: Project ID
        - GOOGLE_CLOUD_REGION: Project region
        - GOOGLE_CLOUD_PROJECT_NUMBER: Project number
        - CLOUD_TASKS_QUEUE: Queue name
        - CLOUD_TASKS_QUEUE_REGION: Queue region
        - K_SERVICE: Cloud Run instance name
    """
    client = tasks_v2.CloudTasksClient()

    # Get configuration from environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    project_region = os.getenv("GOOGLE_CLOUD_REGION")
    project_number = os.getenv("GOOGLE_CLOUD_PROJECT_NUMBER")
    queue = os.getenv("CLOUD_TASKS_QUEUE")
    queue_region = os.getenv("CLOUD_TASKS_QUEUE_REGION")
    instance_name = os.getenv("K_SERVICE")

    parent = client.queue_path(project_id, queue_region, queue)

    # Merge additional kwargs with the main payload
    task_payload = {**payload, **kwargs}

    # Construct the full Cloud Run instance URL
    instance_url = f'https://{instance_name}-{project_number}.{project_region}.run.app{url}'

    task = {
        'http_request': {
            'http_method': tasks_v2.HttpMethod.POST,
            'url': instance_url,
            'body': json.dumps(task_payload).encode(),
            'headers': {'Content-Type': 'application/json'}
        }
    }

    response = client.create_task(request={'parent': parent, 'task': task})
    return response.name


def update_firestore(
        user_id: str,
        chat_history_id: str,
        style_mode: str,
        output_text: Optional[str] = None,
        processed_claims: Optional[List[Dict[str, Any]]] = None,
        processed_imprecise_language_instances: Optional[List[Dict[str, Any]]] = None,
        is_final_update: bool = False
):
    """
    Update Firestore with chat processing results and metadata.

    This function handles updating multiple collections and documents in Firestore,
    including chat messages, processed claims, and imprecise language instances.
    It supports both partial and final updates to track processing status.

    Args:
        user_id (str): The ID of the user
        chat_history_id (str): The ID of the chat history document
        style_mode (str): The processing style/mode being used
        output_text (Optional[str]): The generated answer text
        processed_claims (Optional[List[Dict[str, Any]]]): List of processed claims data
        processed_imprecise_language_instances (Optional[List[Dict[str, Any]]]):
            List of identified imprecise language instances
        is_final_update (bool): Whether this is the final update for the processing

    Returns:
        DocumentReference: Reference to the updated chat document

    Note:
        - Uses batch operations for efficient updates of claims and language instances
        - Automatically generates UUIDs for new messages and instances
        - Updates timestamps using Firestore server timestamp
    """
    db = firestore.Client()
    chat_ref = db.collection('users').document(user_id).collection('chats').document(chat_history_id)

    # Prepare the base update data
    update_data = {
        'updatedAt': firestore.SERVER_TIMESTAMP,
        'mode': style_mode
    }

    # Handle message creation if output text is provided
    if output_text:
        messages_ref = chat_ref.collection('messages')
        answer_id = str(uuid4())
        messages_ref.document(answer_id).set({
            'id': answer_id,
            'content': output_text,
            'type': 'answer',
            'status': 'completed' if is_final_update else 'processing',
            'timestamp': firestore.SERVER_TIMESTAMP,
        })
        update_data['lastMessage'] = output_text

    # Update processing status
    update_data['status'] = 'completed' if is_final_update else 'processing'
    chat_ref.update(update_data)

    # Batch update processed claims if provided
    if processed_claims:
        claims_ref = chat_ref.collection('processed_claims')
        batch = db.batch()
        for claim in processed_claims:
            claim_id = claim.get('id') or str(uuid4())
            claim['id'] = claim_id
            doc_ref = claims_ref.document(claim_id)
            batch.set(doc_ref, {
                'id': claim_id,
                'claim_data': claim,
                'style_mode': style_mode,
                'timestamp': firestore.SERVER_TIMESTAMP
            }, merge=True)
        batch.commit()

    # Batch update imprecise language instances if provided
    if processed_imprecise_language_instances:
        imprecise_lang_ref = chat_ref.collection('imprecise_language_instances')
        batch = db.batch()
        for instance in processed_imprecise_language_instances:
            instance_id = instance.get('id') or str(uuid4())
            instance['id'] = instance_id
            doc_ref = imprecise_lang_ref.document(instance_id)
            batch.set(doc_ref, {
                'id': instance_id,
                'instance_data': instance,
                'style_mode': style_mode,
                'timestamp': firestore.SERVER_TIMESTAMP
            }, merge=True)
        batch.commit()

    return chat_ref
