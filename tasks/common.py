import os
from llama_index.core.settings import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core.callbacks.schema import (
    CBEventType,
    BASE_TRACE_EVENT,
    EventPayload,
)
from typing import Generator


class TraceLangfuseDecorator:
    def __init__(self, func, name="", tags=[]):
        self.func = func
        self.name = name
        self.tags = tags
        self.init_langfuse_handler()

    def init_langfuse_handler(self):
        if os.environ.get('LANGFUSE_SECRET_KEY') is not None:
            self.original_method_ = LlamaIndexCallbackHandler._parse_metadata_from_event
            LlamaIndexCallbackHandler._parse_metadata_from_event = _fixed_parse_metadata_from_event
            self.langfuse_callback_handler = LlamaIndexCallbackHandler(
                public_key=os.environ.get('LANGFUSE_PUBLIC_KEY'),
                secret_key=os.environ.get('LANGFUSE_SECRET_KEY'),
                host=os.environ.get('LANGFUSE_HOST'),
            )
            self.langfuse_callback_handler.set_trace_params(
                name=self.name,
                tags=self.tags,
            )

            Settings.callback_manager = CallbackManager([self.langfuse_callback_handler])
        else:
            print('Warning: LANGFUSE_SECRET_KEY not set. Skipping Langfuse callback handler.')

    def __call__(self, *args, **kwargs):
        res = self.func(*args, **kwargs)
        if hasattr(self, 'langfuse_callback_handler'):
            self.langfuse_callback_handler.flush()
            LlamaIndexCallbackHandler._parse_metadata_from_event = self.original_method_
            Settings.callback_manager = CallbackManager([])
        return res


def trace_langfuse(name="", tags=None):
    if tags is None:
        tags = os.environ.get('LANGFUSE_TAGS', "")
        tags = tags.split(",") if tags else []
    def decorator(func):
        return TraceLangfuseDecorator(func, name, tags)

    return decorator


def _fixed_parse_metadata_from_event(self, event):
    if event.payload is None:
        return

    metadata = {}

    for key in event.payload.keys():
        if key not in [
            EventPayload.MESSAGES,
            EventPayload.QUERY_STR,
            EventPayload.PROMPT,
            EventPayload.COMPLETION,
            EventPayload.SERIALIZED,
            "additional_kwargs",
        ]:
            if key != EventPayload.RESPONSE:
                metadata[key] = event.payload[key]
            else:
                response = event.payload.get(EventPayload.RESPONSE)
                if hasattr(response, "__dict__"):
                    for res_key, value in vars(response).items():
                        if (
                                not res_key.startswith("_")
                                and res_key
                                not in [
                            "response",
                            "response_txt",
                            "message",
                            "additional_kwargs",
                            "delta",
                            "raw",
                        ]
                                and not isinstance(value, Generator)
                        ):
                            metadata[res_key] = value

    return metadata or None
