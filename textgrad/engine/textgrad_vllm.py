try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError(
        "If you'd like to use VLLM models, please install the vllm package by running `pip install vllm` or `pip install textgrad[vllm]."
    )

import os
import platformdirs
from .base import EngineLM, CachedEngine

from transformers import AutoTokenizer, AutoModelForCausalLM


class ChatVLLM(EngineLM, CachedEngine):
    # Default system prompt for VLLM models
    DEFAULT_SYSTEM_PROMPT = ""

    def __init__(
        self,
        model_string="meta-llama/Meta-Llama-3-8B-Instruct",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    ):
        root = platformdirs.user_cache_dir("textgrad")
        # root = "/home/chenmh/.ollama/models/manifests/registry.ollama.ai/library/"
        cache_path = os.path.join(root, f"{model_string}")
        super().__init__(cache_path=cache_path)

        self.model_string = model_string
        self.system_prompt = system_prompt

        # self._check_and_download_model(self.model_string)

        self.client = LLM(cache_path)
        self.tokenizer = self.client.get_tokenizer()

    def _check_and_download_model(self, model_string):
        """
        Check if the model exists locally, and if not, download it.
        """
        model_cache_dir = platformdirs.user_cache_dir("huggingface", "models")
        model_path = os.path.join(model_cache_dir, model_string)
        
        # If the model directory doesn't exist, download it.
        if not os.path.exists(model_path):
            print(f"Model '{model_string}' not found locally. Downloading...")
            os.makedirs(model_cache_dir, exist_ok=True)
            
            # Download the model and tokenizer using Hugging Face API
            try:
                AutoModelForCausalLM.from_pretrained(model_string, cache_dir=model_cache_dir)
                AutoTokenizer.from_pretrained(model_string, cache_dir=model_cache_dir)
                print(f"Model '{model_string}' downloaded successfully.")
            except Exception as e:
                print(f"Error downloading model '{model_string}': {e}")
        else:
            print(f"Model '{model_string}' found locally.")

    def generate(
        self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        # The chat template ignores the system prompt;
        conversation = []
        if sys_prompt_arg:
            conversation = [{"role": "system", "content": sys_prompt_arg}]

        conversation += [{"role": "user", "content": prompt}]
        chat_str = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=1
        )

        response = self.client.generate([chat_str], sampling_params)
        response = response[0].outputs[0].text

        self._save_cache(sys_prompt_arg + prompt, response)

        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
