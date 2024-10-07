import base64
import io
import magic
import traceback

from flask import Blueprint, request, jsonify, Response
from pydub import AudioSegment
from vertexai.generative_models import Part, Tool

import src.anthropic.generate as anthropic_generate
import src.chat.basic_chat as perform_basic_chat
import src.chat.agent_chat as perform_agent_chat
import src.routes.utils as endpoint_utils
import src.remote_config.utils as remote_config_utils
from src.chat.utils import (
    clean_text,
    upload_image_to_gcs,
    get_config_and_prompt,
    create_function_declaration
)


chat_bp = Blueprint('chat', __name__, url_prefix='/chat')


@chat_bp.route("", methods=["POST"])
def chat():
    """Handle chat requests."""
    # Verify the authentication token
    auth_result = endpoint_utils.verify_auth_token(request)
    if isinstance(auth_result, tuple):
        return auth_result

    decoded_token = auth_result
    user_id = decoded_token['uid']

    # Parse JSON data from request
    data = endpoint_utils.parse_json_data(request)

    # Extract and clean text
    text = data.get("text")
    if text:
        text = clean_text(text)

    # Extract other request data
    chat_history_id = data.get("chat_id", data.get("chatId"))
    message_id = data.get("message_id", data.get("messageId"))
    system_instruction = data.get("system_instruction", data.get("systemInstruction"))
    style_mode = data.get("style_mode", data.get("styleMode"))

    # Check for uploaded media
    uploaded_files = endpoint_utils.check_uploaded_media(user_id, chat_history_id, message_id)

    # Create a task for background processing
    payload = {
        'text': text,
        'user_id': user_id,
        'chat_history_id': chat_history_id,
        'message_id': message_id,
        'system_instruction': system_instruction,
        'style_mode': style_mode,
        'uploaded_files': uploaded_files
    }

    endpoint_utils.create_cloud_task('/chat/task', payload)

    return jsonify({'status': 'processing'}), 202


@chat_bp.route("/task", methods=["POST"])
def task_chat():
    """Task: Process chat in the background."""
    data = endpoint_utils.parse_json_data(request)

    # Extract data from the request
    text = data['text']
    user_id = data['user_id']
    chat_history_id = data['chat_history_id']
    style_mode = data.get('style_mode', 'descriptive')
    uploaded_files = data['uploaded_files']

    # Prepare content for chat generation
    contents = []

    # Include uploaded files
    for uploaded_file in uploaded_files:
        contents.append(
            Part.from_uri(
                uri=uploaded_file["gcsPath"],
                mime_type=uploaded_file["fileMimeType"])
        )

    engage_workflow = False
    if "Please find all medical claims and instances of imprecise language. Be thorough and complete." in text:
        engage_workflow = True
    contents.append(Part.from_text(text))

    try:
        # Fetch role prompt
        role_prompt = get_config_and_prompt("role_prompt")

        # Fetch verification prompt
        verification_prompt = get_config_and_prompt("verification_prompt")

        # Define functions
        # Identify medical claims
        identify_medical_claims_function = create_function_declaration(
            "medical_claims_identification",
            "identify_medical_claims_multi_function_description",
            "identify_medical_claims_multi_function_parameters"
        )

        # Identify imprecise language
        identify_imprecise_language_function = create_function_declaration(
            "imprecise_language_identification",
            "identify_imprecise_language_multi_function_description",
            "identify_imprecise_language_multi_function_parameters"
        )

        citations_tool = Tool(
            function_declarations=[
                identify_imprecise_language_function,
                identify_medical_claims_function,
            ],
        )

        # Generate chat response
        output_text, _, processed_claims, processed_imprecise_language_instances = perform_agent_chat.generate_text(
            prompt=contents,
            system_instruction=role_prompt,
            verification_prompt=verification_prompt,
            style_mode=style_mode,
            engage_workflow=engage_workflow,
            user_id=user_id,
            chat_history_id=chat_history_id,
            tools=[citations_tool],
            allowed_function_names=["medical_claims_identification", "imprecise_language_identification"]
        )

        # Update Firestore with the generated answer
        endpoint_utils.update_firestore(
            user_id,
            chat_history_id,
            style_mode,
            output_text,
            processed_claims,
            processed_imprecise_language_instances,
            is_final_update=True
        )

        return jsonify({
            "output_text": output_text,
            "chat_history_id": chat_history_id
        }), 200

    except ValueError as e:
        return Response(str(e), status=404)


@chat_bp.route("/title", methods=["POST"])
def get_chat_title():
    """Generate a title for the chat."""
    # Verify the authentication token
    try:
        auth_result = endpoint_utils.verify_auth_token(request)
        if isinstance(auth_result, tuple):
            return auth_result

        decoded_token = auth_result
        user_id = decoded_token['uid']

        # Parse JSON data from request
        data = endpoint_utils.parse_json_data(request)

        # Extract and clean text
        text = data.get('text')
        if text:
            text = clean_text(text)

        # Prompt
        # Fetch configuration from Remote Config
        config = remote_config_utils.get_remote_config_value("Prompts", "generate_chat_title")
        if not config:
            return Response("Configuration not found", status=404)

        file_name = config['fileName']

        # Fetch prompt from GCS
        prompt = remote_config_utils.get_gcs_prompt(file_name)

        # Generate chat title
        prompt = prompt.format(input_text=text)
        generated_title = anthropic_generate.generate(prompt=prompt)
        return jsonify({'title': generated_title})
    except Exception as e:
        return jsonify({'title': 'Error occurred', 'error': str(e) + f"\n{traceback.format_exc()}"}), 500
