# gpt2_model.py
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "gpt2"  # This is the model name for the 124M parameter version of GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
