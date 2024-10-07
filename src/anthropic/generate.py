import os
from anthropic import AnthropicVertex
from tenacity import retry, wait_random_exponential, stop_after_attempt


@retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(3))
def generate(
    prompt,
    system_instruction: str = "",
    model_name: str = "claude-3-5-sonnet@20240620",
    max_output_tokens: int = 4096
):
    """Generate."""

    client = AnthropicVertex(region="us-east5",
                             project_id=os.getenv("GOOGLE_CLOUD_PROJECT"))

    message = client.messages.create(
        max_tokens=max_output_tokens,
        system=system_instruction,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
    )
    return message.content[0].text


@retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(3))
def stream(
    prompt,
    system_instruction: str = "",
    model_name: str = "claude-3-5-sonnet@20240620",
    max_output_tokens: int = 4096
):
    """Stream."""

    client = AnthropicVertex(region="us-east5",
                             project_id=os.getenv("GOOGLE_CLOUD_PROJECT"))

    with client.messages.stream(
        max_tokens=max_output_tokens,
        system=system_instruction,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
    ) as response:
        for text in response.text_stream:
            yield text
