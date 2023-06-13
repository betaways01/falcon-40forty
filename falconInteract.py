from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
# load tokenizer
custom_pretrained_model_path = "/workspace/falcon40b/falcon-40b"
model = AutoModelForCausalLM.from_pretrained(custom_pretrained_model_path)
tokenizer = AutoTokenizer.from_pretrained(custom_pretrained_model_path)

# Start interaction loop
# for as long as you like, until you type quit
while True:
    # Ask user input
    input_text = input("You: ")

    # Quit if the user types 'quit'
    if input_text.lower() == 'quit':
        break

    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate response from model
    output_ids = model.generate(input_ids)

    # Decode output IDs to get model's answer
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Print the model's answer
    print(f"Falcon-40b: {output_text}")
