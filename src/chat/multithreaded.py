import concurrent.futures
import threading
import time
from tqdm import tqdm
from typing import Any, List, Optional

import vertexai
import vertexai.preview.generative_models as generative_models


class TextGenerator:
    def __init__(self,
                 project_id: str,
                 location: str = "us-central1",
                 model_instance: Optional[Any] = None,
                 system_prompt: Optional[str] = "",
                 temperature: float = 0,
                 top_p: float = 0.8,
                 top_k: int = 40,
                 max_output_tokens: int = 8192,
                 verbose: bool = False,
                 max_calls_per_minute: int = 60):
        self.project_id = project_id
        self.location = location
        self.model_instance = model_instance
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.verbose = verbose
        self.max_calls_per_minute = max_calls_per_minute
        self.call_count = 0
        self.lock = threading.Lock()

    def generate_text(
        self,
        contents,
    ) -> str:
        """Generate text."""

        vertexai.init(project=self.project_id, location=self.location)

        # Query the model
        response = self.model_instance.generate_content(
            contents=contents,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            },
            safety_settings={
              generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
              generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
              generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
              generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            }, stream=False
        )

        if self.verbose:
            print(response.text)

        return response

    def process_text(self, contents):
        """Process text."""
        with self.lock:
            if self.call_count >= self.max_calls_per_minute:
                time.sleep(60)
                self.call_count = 0
            self.call_count += 1
        return self.generate_text(contents)

    def generate_texts(self, contents_list: List[str], num_threads: int = 5):
        """Generate texts."""
        results = [None] * len(contents_list)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_content = {executor.submit(self.process_text, content): (index, content) for index, content in enumerate(contents_list)}
            for future in tqdm(concurrent.futures.as_completed(future_to_content), total=len(future_to_content), desc="Processing texts"):
                index, content = future_to_content[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    print(f"Generated an exception for {content}: {e}")
        return results
