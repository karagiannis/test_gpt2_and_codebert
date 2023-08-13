from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # Model name for the largest GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
