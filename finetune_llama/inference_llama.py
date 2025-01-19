import torch
import random

random.seed(42)
torch.manual_seed(42)


def genreate_response(prompt_text, model, tokenizer, max_length=30, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to('mps')

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2
    )

    responses =[]
    for response_id in output_sequences:
        response = tokenizer.decode(response_id, skip_special_tokens=True)
        responses.append(response)

    return responses


from transformers import LlamaTokenizer, LlamaForCausalLM
model_path = "openlm-research/open_llama_3b_v2"
tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)
base_model = LlamaForCausalLM.from_pretrained(model_path)

from peft import LoraConfig, PeftModel
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM")
model = PeftModel(base_model, lora_config, adapter_name="Shakespeare")

device = torch.device("mps")
model.to(device)


prompt_text = "Uneasy lies the head that wears a crown."
responses = genreate_response(prompt_text, model, tokenizer)
for response in responses:
    print(response)