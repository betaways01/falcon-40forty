import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
import json

# let's do some logging so we know each step of the code run
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# Load the data
with open('train_set.json', 'r') as f:
    train_data = json.load(f)
with open('dev_set.json', 'r') as f:
    dev_data = json.load(f)

# Prepare the data for use with the model
# This may vary depending on how your JSON data is structured
train_texts = [item['input'] for item in train_data]
train_labels = [item['output'] for item in train_data]
dev_texts = [item['input'] for item in dev_data]
dev_labels = [item['output'] for item in dev_data]

# Create the tokenizer
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")

# Encode the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
dev_encodings = tokenizer(dev_texts, truncation=True, padding=True)

# Create a Dataset for use with the Trainer
class FalconDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = FalconDataset(train_encodings)
dev_dataset = FalconDataset(dev_encodings)

# Load the pre-trained model
# in your case:
# "/workspace/falcon40b/falcon-40b"
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b") # change this to "/workspace/falcon40b/falcon-40b"

# Create training arguments
# change these sections as it fits you
# set ther output folder
# set umber of training epochs
# set the batch sizes
# set the learning rate, etc.
# can also add other arguments as needed
training_args = TrainingArguments(
    output_dir='./results', 
    num_train_epochs=3, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64, 
    warmup_steps=500,
    weight_decay=0.01,  
    logging_dir='./logs',
)

# Create the Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)
trainer.train()

# save model
model.save_pretrained("./falcon-40b_med")
# save tokens
tokenizer.save_pretrained("./falcon-40b_med")

# reload model and tokenizer
# model = AutoModelForCausalLM.from_pretrained("./falcon-40b_med")
# tokenizer = AutoTokenizer.from_pretrained("./my_fine_tuned_model")
