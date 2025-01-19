import torch

from transformers import LlamaTokenizer, LlamaForCausalLM
model_path = "openlm-research/open_llama_3b_v2"
tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True, cache_dir="/Users/yaxiong/Documents/vscodes/gen_ai_experiments/finetune_llama/cache_models")
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




import os
import requests
file_name = "shakespeare.txt"
url = "http://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.isfile(file_name):
    data = requests.get(url)
    with open(file_name, 'w') as f:
        f.write(data.text)

from transformers import TextDataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=file_name,
    block_size=128)[:256]



from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
training_args = TrainingArguments(
    output_dir = "output",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    evaluation_strategy='no'
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)