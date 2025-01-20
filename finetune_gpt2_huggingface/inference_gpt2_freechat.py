from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "openai-community/gpt2"  # Choose "gpt2", "gpt2-medium", or other variants
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

print("Chat with GPT-2! Type 'exit' to end the conversation.")

# Infinite loop for the chat
while True:
    # User input
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Add the user input as a prompt
    prompt = user_input

    # Encode the input and generate a response
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate response
    output = model.generate(
        input_ids,
        max_length=100,  # Maximum length of the response
        num_return_sequences=1,  # Number of responses
        no_repeat_ngram_size=2,  # Prevent repetitive phrases
        top_k=50,  # Top-k sampling
        top_p=0.95,  # Top-p (nucleus) sampling
        temperature=0.7,  # Control creativity
        do_sample=True,  # Enable sampling
    )

    # Decode and print the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"GPT-2: {response[len(prompt):].strip()}")  # Remove the prompt from the response