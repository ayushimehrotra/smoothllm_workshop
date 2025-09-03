import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback


class LLM:

    """Forward pass through a LLM."""

    def __init__(
        self, 
        model_path, 
        tokenizer_path, 
        conv_template_name,
        device
    ):

        # Language model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True,
            device_map="auto",
            local_files_only=True
        ).eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True
        )
        self.tokenizer.padding_side = 'left'
        if 'llama-2' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def __call__(self, batch, max_new_tokens=100):
        # Pass current batch through the tokenizer
        
#         print(f"[DEBUG] Batch = {batch}")
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        
        
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

#         print(f"[DEBUG] Inputs = {batch_input_ids}")
#         print(f"[DEBUG] Attention Mask = {batch_attention_mask}")

        # Forward pass through the LLM
        try:
#             print(f"[DEBUG] Model = {self.model}")
            outputs = self.model.generate(
                batch_input_ids, 
                attention_mask=batch_attention_mask, 
                max_new_tokens=max_new_tokens
            )
        except RuntimeError as e:
            print(f"A RuntimeError occurred during model.generate: {e}")
            traceback.print_exc()
            return []
            
#         print(f"[DEBUG] Output: {outputs}")
        
        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
#         print(f"[DEBUG] Decode: {batch_outputs}")
        
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        ]

        return batch_outputs