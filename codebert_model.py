# codebert_model.py
from transformers import AutoTokenizer, AutoModel

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
