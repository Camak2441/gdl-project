from openai import OpenAI
from dotenv import load_dotenv
import json
import os
from abc import ABC
from utils import *

load_dotenv()

openai_args = {
    "api_key",
    "organization",
    "project",
    "webhook_secret",
    "base_url",
    "websocket_base_url",
    "timeout",
    "max_retries",
    "default_headers",
    "default_query",
}

chat_completion_args = {
    "messages",
    "model",
    "max_completion_tokens",
    "reasoning_effort",
    "temperature",
    "timeout",
}

file_path = os.path.abspath(__file__)


models = {}
with open(os.path.join(os.path.dirname(file_path), "models.json")) as fp:
    models = json.load(fp)


class OpenAIClient:

    def __init__(self, **kwargs):
        self.client = OpenAI(
            **intersect_dict(openai_args, kwargs),
        )
        self.kwargs = kwargs
        self.chat_completion_args = chat_completion_args.copy()
        if kwargs.get("fixedTemp", False):
            self.chat_completion_args.remove("temperature")
        if not kwargs.get("reasoningModel", False):
            self.chat_completion_args.remove("reasoning_effort")
        if "maxCompletionTokens" in kwargs:
            if "max_completion_tokens" in kwargs:
                kwargs["max_completion_tokens"] = min(
                    max(0, kwargs["max_completion_tokens"]),
                    kwargs["maxCompletionTokens"],
                )

    def create_completion(self, messages, **kwargs):
        return self.client.chat.completions.create(
            messages=messages,
            **(
                add_dict(
                    intersect_dict(self.chat_completion_args, self.kwargs),
                    intersect_dict(self.chat_completion_args, kwargs),
                )
            ),
        )


def create_client(model, **kwargs):
    model_args = models[model]
    match model_args["source"]:
        case "ollama":
            return OpenAIClient(
                model=model,
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                **add_dict(kwargs, model_args),
            )
        case "openai":
            return OpenAIClient(
                model=model, api_key=os.environ.get("OPENAI_API_KEY"), **model_args
            )
