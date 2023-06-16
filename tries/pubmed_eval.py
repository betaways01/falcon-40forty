from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Load model and tokenizer
custom_pretrained_model_path = "/workspace/falcon40b/falcon-40b"
model = AutoModelForCausalLM.from_pretrained(custom_pretrained_model_path)
tokenizer = AutoTokenizer.from_pretrained(custom_pretrained_model_path)

# Load ori_pqal.json data 
with open("ori_pqal.json", "r") as file:
    inp_data = json.load(file)

# Initialize list to store model's answers
model_answers = []

# Loop over input data
for example in inp_data:
    # Extract question and context
    question = example["question"]
    context = example["context"]

    # Encode question and context
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')

    # Make a prediction with falcon-40b model
    outputs = model.generate(inputs["input_ids"])
    
    # Decode the prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Store the model's answer
    model_answers.append({"id": example["id"], "answer": prediction})

# Save model's answers
with open("model_answers.json", "w") as file:
    json.dump(model_answers, file)

# TODO: Use the MMLU evaluation scripts to compute the benchmark metrics

# to run pipline:
# python mmlu_eval.py -- model_answers.json --ground_truth_file test_ground_truth.json
# python mmlu_eval.py --predictions_file model_answers.json --ground_truth_file test_ground_truth.json 