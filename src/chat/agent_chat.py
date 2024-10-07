import os
import requests
import traceback
import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
    ToolConfig
)
from typing import Any, Dict, List, Optional
from uuid import uuid4
from vertexai.preview.generative_models import grounding

import src.routes.utils as endpoint_utils
from src.chat.multithreaded import TextGenerator
from src.chat.utils import get_chat_history, save_chat_history
from src.claims_analysis.processing import structure_claims_analysis


def generate_text(
    prompt,
    system_instruction: Optional[str] = None,
    verification_prompt: str = "",
    style_mode: str = "descriptive",
    engage_workflow: bool = False,
    user_id: Optional[str] = None,
    chat_history_id: Optional[str] = None,
    save_session_history: bool = True,
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT"),
    tools: List[Any] = None,
    allowed_function_names: List[str] = None,
    safety_settings: Optional[Dict[str, Any]] = None,
    location: str = "us-central1",
    model_name: str = "gemini-1.5-pro-002",
    temperature: int = 0,
    max_output_tokens: int = 8192
):
    """Generate text."""
    vertexai.init(project=project_id, location=location)

    if not tools:
        tools = []

    if allowed_function_names:
        allowed_function_names = []

    if not safety_settings:
        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }

    # Determine chat history
    if chat_history_id:
        chat_history = get_chat_history(user_id, chat_history_id)
    else:
        chat_history_id, chat_history = str(uuid4()), []

    first_run_through = True if not chat_history else False

    # Initialize medical claims model
    if engage_workflow:
        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                allowed_function_names=["medical_claims_identification"],
            ))
    else:
        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.AUTO,
            ))
    medical_claims_model_instance = GenerativeModel(
        model_name,
        system_instruction=None if not system_instruction else [system_instruction],
        generation_config=GenerationConfig(
            temperature=0.2,
        ),
        safety_settings=safety_settings,
        tools=tools,
        tool_config=tool_config
    )

    # Initialize imprecise language model
    if engage_workflow:
        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                allowed_function_names=["imprecise_language_identification"],
            ))
    else:
        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.AUTO,
            ))
    imprecise_language_model_instance = GenerativeModel(
        model_name,
        system_instruction=None if not system_instruction else [system_instruction],
        generation_config=GenerationConfig(
            temperature=0.2,
        ),
        safety_settings=safety_settings,
        tools=tools,
        tool_config=tool_config
    )

    # Initialize grounding model
    grounding_model_instance = GenerativeModel(
        model_name,
        system_instruction=None if not system_instruction else [system_instruction],
        generation_config=GenerationConfig(
            temperature=0.2,
        ),
        safety_settings=safety_settings,
        tools=[
            Tool.from_google_search_retrieval(
                google_search_retrieval=grounding.GoogleSearchRetrieval()
            )
        ]
    )

    # Initialize output response model
    output_response_model_instance = GenerativeModel(
        model_name,
        system_instruction=None if not system_instruction else [system_instruction],
        generation_config=GenerationConfig(
            temperature=0.2,
        ),
        safety_settings=safety_settings
    )

    # Initialize multithreading text generator
    text_generator = TextGenerator(
        project_id=project_id,
        location=location,
        model_instance=grounding_model_instance,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )

    # Initialize tracking variables
    loop_count = 0
    output_text, error_message = "", ""
    processed_imprecise_language_instances = None
    processed_claims = None
    try:
        # Send initial message
        chat = medical_claims_model_instance.start_chat(history=chat_history, response_validation=False)
        response = chat.send_message(prompt)

        break_loop = False
        while True:
            response_parts = []
            processing_parts = response.candidates[0].content.parts
            for part in processing_parts:
                function_call_name = part.function_call.name
                if not function_call_name:
                    # Direct text response
                    output_text = part.text
                    break_loop = True
                    break
                elif function_call_name == 'imprecise_language_identification':
                    try:
                        print(f"Calling function {function_call_name}...")
                        # Handle imprecise language identification request
                        args = dict(part.function_call.args)
                        rc_to_json_array = lambda repeated_composite: [dict(item) for item in repeated_composite]
                        identified_instances_data = rc_to_json_array(args.get('identified_instances', []))
                        processed_imprecise_language_instances = identified_instances_data

                        if processed_imprecise_language_instances:
                            for processed_instance in processed_imprecise_language_instances:
                                processed_instance["id"] = str(uuid4())

                        # Partial update with imprecise language instances
                        endpoint_utils.update_firestore(
                            user_id,
                            chat_history_id,
                            style_mode,
                            processed_imprecise_language_instances=processed_imprecise_language_instances
                        )

                        api_response = "Processed all imprecise language identified."
                        if engage_workflow:
                            next_steps_instruction = (
                                "Respond that the input text has been processed and summarize the findings. "
                                " Ask if the user needs help understanding the findings or would like to know more "
                                " about any finding in particular."
                            )
                        else:
                            next_steps_instruction = "Proceed"
                        response_part = Part.from_function_response(
                            name=function_call_name,
                            response={
                                "content": api_response,
                                "processed_imprecise_language_instances": processed_imprecise_language_instances,
                                "next_steps_instruction": next_steps_instruction
                            },
                        )
                        response_parts.append(response_part)

                        if len(response_parts) < len(processing_parts):
                            continue
                    except Exception as e:
                        # Handle any errors that occurred during the request
                        local_error_message = f"Error when Identifying Imprecise Language: {str(e)}"
                        error_message += f"\n{local_error_message}"
                        print(f"error_message: {local_error_message}\n{traceback.format_exc()}")
                        response_part = Part.from_function_response(
                            name=function_call_name,
                            response={
                                "error": local_error_message,
                                "instructions": "Error occurred while identifying imprecise language. Please try again."},
                        )
                        response_parts.append(response_part)
                elif 'medical_claims_identification':
                    try:
                        print(f"Calling function {function_call_name}...")
                        # Handle medical claims identification request
                        args = dict(part.function_call.args)
                        rc_to_json_array = lambda repeated_composite: [dict(item) for item in repeated_composite]
                        identified_claims_data = rc_to_json_array(args.get('identified_claims', []))

                        claims = [identified_claim["claim"] for identified_claim in identified_claims_data]
                        prompts = [verification_prompt.format(input_claim=claim) for claim in claims]
                        results = text_generator.generate_texts(prompts, num_threads=8)
                        structured_claims_results = [structure_claims_analysis(result.to_dict()) for result in results]
                        claims_data = [{**claim_data, **structured_claims_result} for claim_data, structured_claims_result
                                       in zip(identified_claims_data, structured_claims_results)]
                        processed_claims = claims_data

                        if processed_claims:
                            for processed_claim in processed_claims:
                                processed_claim["id"] = str(uuid4())

                        # Partial update with processed claims
                        endpoint_utils.update_firestore(
                            user_id,
                            chat_history_id,
                            style_mode,
                            processed_claims=processed_claims
                        )

                        api_response = "Processed all claims and generated analysis."
                        if engage_workflow:
                            next_steps_instruction = "Now perform imprecise language identification analysis."
                        else:
                            next_steps_instruction = "Proceed"
                        response_part = Part.from_function_response(
                            name=function_call_name,
                            response={
                                "content": api_response,
                                "processed_claims": processed_claims,
                                "next_steps_instruction": next_steps_instruction
                            },
                        )
                        response_parts.append(response_part)
                    except Exception as e:
                        # Handle any errors that occurred during the request
                        local_error_message = f"Error when Identifying Medical Claims: {str(e)}"
                        error_message += f"\n{local_error_message}"
                        print(f"error_message: {local_error_message}\n{traceback.format_exc()}")
                        response_part = Part.from_function_response(
                            name=function_call_name,
                            response={
                                "error": local_error_message,
                                "instructions": "Error occurred while identifying medical claims. Please try again."},
                        )
                        response_parts.append(response_part)
                else:
                    # Unhandled function call
                    output_text = 'Could not resolve appropriate function and determine an answer.'
                    break

            if break_loop:
                break
            else:
                if loop_count == 0:
                    chat = imprecise_language_model_instance.start_chat(history=chat.history)
                    response = chat.send_message(response_parts)
                    loop_count += 1
                    pass
                else:
                    chat = output_response_model_instance.start_chat(history=chat.history, response_validation=False)
                    response = chat.send_message(response_parts)
    except Exception as e:
        print(f"Error occurred: {e}\n{traceback.format_exc()}")
        output_text = f"Please try again. An unexpected error occurred. {e}\n{traceback.format_exc()}"

    # Save chat history
    if save_session_history:
        save_chat_history(user_id, chat_history_id, chat.history)

    if error_message:
        output_text += error_message

    return output_text, chat_history_id, processed_claims, processed_imprecise_language_instances
