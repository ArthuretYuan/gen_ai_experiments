import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

prompt = "Hello, my dog is cute"

# Encode the input text
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    input_ids,
    max_length=100,  # Maximum length of the generated text
    num_return_sequences=1,  # Number of outputs
    no_repeat_ngram_size=2,  # Prevent repetitive sequences
    top_k=50,  # Top-k sampling
    top_p=0.95,  # Top-p (nucleus) sampling
    temperature=1.0,  # Sampling temperature
    do_sample=True,  # Enable sampling
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)