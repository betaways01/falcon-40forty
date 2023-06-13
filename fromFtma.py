from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
import json
#from datasets import Dataset
import torch
from torch.utils.data import DataLoader, Dataset
#from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer, AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline



custom_pretrained_model_path = "/workspace/falcon40b/falcon-40b"



model = AutoModelForCausalLM.from_pretrained(custom_pretrained_model_path, trust_remote_code=True, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(custom_pretrained_model_path)



print("### loaded the model")