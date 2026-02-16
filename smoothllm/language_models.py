import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM


def _get_default_device():
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LLM:
    """Forward pass through a LLM."""

    def __init__(self, model_path, tokenizer_path, conv_template_name, device=None):
        if device is None:
            device = _get_default_device()

        use_device_map = device.startswith("cuda")
        dtype = torch.bfloat16 if device != "cpu" else torch.float32

        # Language model
        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True,
            local_files_only=True,
        )
        if use_device_map:
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, **load_kwargs
        ).eval()

        if not use_device_map:
            self.model = self.model.to(device)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "left"
        if "llama-2" in tokenizer_path.lower():
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(conv_template_name)
        if self.conv_template.name == "llama-2":
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def __call__(self, batch, max_new_tokens=100):
        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, padding=True, truncation=False, return_tensors="pt"
        )

        batch_input_ids = batch_inputs["input_ids"].to(self.model.device)
        batch_attention_mask = batch_inputs["attention_mask"].to(self.model.device)

        # Forward pass through the LLM
        try:
            outputs = self.model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
            )
        except RuntimeError as e:
            print(f"A RuntimeError occurred during model.generate: {e}")
            return []

        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True))
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i] :] for i, output in enumerate(batch_outputs)
        ]

        return batch_outputs
