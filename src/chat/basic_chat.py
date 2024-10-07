import os
import traceback
import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
)
from typing import Optional
from uuid import uuid4

from src.chat.utils import get_chat_history, save_chat_history


def generate_text(
    prompt,
    system_instruction: Optional[str] = None,
    user_id: Optional[str] = None,
    chat_history_id: Optional[str] = None,
    save_session_history: bool = True,
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT"),
    location: str = "us-central1",
    model_name: str = "gemini-1.5-pro-002",
    max_output_tokens: int = 8192,
    response_mime_type: Optional[str] = None
):
    """Generate text."""
    vertexai.init(project=project_id, location=location)

    # Initialize Gemini model
    model = GenerativeModel(
        model_name,
        system_instruction=None if not system_instruction else [system_instruction],
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type
        ),
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        },
    )

    with_safety_settings_model = GenerativeModel(
        model_name,
        system_instruction=None if not system_instruction else [system_instruction],
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type
        )
    )

    if chat_history_id:
        chat_history = get_chat_history(user_id, chat_history_id)
        chat = model.start_chat(history=chat_history)
    else:
        chat_history_id, chat_history = str(uuid4()), []
        chat = model.start_chat()

    try:
        response = chat.send_message(prompt)

        output_text = response.text
    except Exception as e:
        try:
            with_ss_chat = with_safety_settings_model.start_chat(history=chat_history)
            response = with_ss_chat.send_message(prompt)
            output_text = response.text
        except:
            print(f"Error occurred: {e}\n{traceback.format_exc()}")
            output_text = f"Please try again. An unexpected error occurred: {e}\n{traceback.format_exc()}"

    # Save session history
    if save_session_history:
        save_chat_history(user_id, chat_history_id, chat.history)
    else:
        chat_history_id = None

    return output_text, chat_history_id
