import os

_BASE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "llama2": {
        "model_path": os.environ.get(
            "SMOOTHLLM_LLAMA2_PATH", os.path.join(_BASE, "Llama-2-7b-chat-hf")
        ),
        "tokenizer_path": os.environ.get(
            "SMOOTHLLM_LLAMA2_PATH", os.path.join(_BASE, "Llama-2-7b-chat-hf")
        ),
        "conversation_template": "llama-2",
    },
    "vicuna": {
        "model_path": os.environ.get(
            "SMOOTHLLM_VICUNA_PATH", os.path.join(_BASE, "vicuna-13b-v1.5")
        ),
        "tokenizer_path": os.environ.get(
            "SMOOTHLLM_VICUNA_PATH", os.path.join(_BASE, "vicuna-13b-v1.5")
        ),
        "conversation_template": "vicuna",
    },
}
