import json
import os
from google.cloud import firestore, storage
from google.cloud import tasks_v2
from flask import jsonify
from firebase_admin import auth
from typing import Any, Dict, List, Optional
from uuid import uuid4


def verify_auth_token(request):
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'No authorization token provided'}), 401

    try:
        auth_token = auth_header.split(' ')[1]
        return auth.verify_id_token(auth_token)
    except IndexError:
        return jsonify({'error': 'Invalid authorization header format'}), 401
    except auth.InvalidIdTokenError:
        return jsonify({'error': 'Invalid authorization token'}), 401


def parse_json_data(request):
    """Parse JSON data from the request."""
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
    """Check for uploaded media in GCS and return metadata."""
    bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

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
    client = tasks_v2.CloudTasksClient()

    # Determine project ID
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    # Determine project region
    project_region = os.getenv("GOOGLE_CLOUD_REGION")

    # Determine project number
    project_number = os.getenv("GOOGLE_CLOUD_PROJECT_NUMBER")

    # Determine queue
    queue = os.getenv("CLOUD_TASKS_QUEUE")

    # Determine queue region
    queue_region = os.getenv("CLOUD_TASKS_QUEUE_REGION")

    # Determine Cloud Run instance name
    instance_name = os.getenv("K_SERVICE")

    parent = client.queue_path(project_id, queue_region, queue)

    # Add any additional kwargs to the task payload
    task_payload = {**payload, **kwargs}

    # Determine instance url
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
    Update Firestore with the generated answer, processed claims, and imprecise language instances.
    This function can be called multiple times for partial updates.
    """
    db = firestore.Client()
    chat_ref = db.collection('users').document(user_id).collection('chats').document(chat_history_id)

    # Update the chat document with the current processing status
    update_data = {
        'updatedAt': firestore.SERVER_TIMESTAMP,
        'mode': style_mode
    }

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

    if is_final_update:
        update_data['status'] = 'completed'
    else:
        update_data['status'] = 'processing'

    chat_ref.update(update_data)

    # Save processed claims
    if processed_claims:
        claims_ref = chat_ref.collection('processed_claims')
        batch = db.batch()
        for claim in processed_claims:
            claim_id = claim.get('id')
            if not claim_id:
                claim_id = str(uuid4())
                claim['id'] = claim_id
            doc_ref = claims_ref.document(claim_id)
            batch.set(doc_ref, {
                'id': claim_id,
                'claim_data': claim,
                'style_mode': style_mode,
                'timestamp': firestore.SERVER_TIMESTAMP
            }, merge=True)
        batch.commit()

    # Save processed imprecise language instances
    if processed_imprecise_language_instances:
        imprecise_lang_ref = chat_ref.collection('imprecise_language_instances')
        batch = db.batch()
        for instance in processed_imprecise_language_instances:
            instance_id = instance.get('id')
            if not instance_id:
                instance_id = str(uuid4())
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

