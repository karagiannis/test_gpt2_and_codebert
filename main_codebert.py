import torch
from codebert_model import tokenizer as codebert_tokenizer, model as codebert_model
from datasets import load_dataset
from transformers import RobertaForCausalLM



def get_dataset(prog_lang, nat_lang):
    test_data = load_dataset("blindsubmissions/M2CRB")

    test_data = test_data.filter(
        lambda example: example["docstring_language"] == nat_lang
        and example["language"] == prog_lang
    )
    return test_data

def generate_combined_output(prompt):
    # Tokenize the input prompt for the CodeBERT model
    input_ids_codebert = codebert_tokenizer(prompt, return_tensors="pt").input_ids
    attention_mask = torch.ones_like(input_ids_codebert)

    # Use RobertaForCausalLM for generation
    with torch.no_grad():
        codebert_output = codebert_model.generate(input_ids=input_ids_codebert, attention_mask=attention_mask)

    codebert_output_text = codebert_tokenizer.decode(codebert_output[0], skip_special_tokens=True)
    return codebert_output_text

def main():
    programming_language = "python"
    natural_language = "fr"

    m2crb_dataset = get_dataset(programming_language, natural_language)
    test_subset = m2crb_dataset["test"]  # Access the "test" subset

    for example in test_subset:  # Iterate through examples in the "test" subset
        prompt = example["docstring"]
        codebert_output = generate_combined_output(prompt)

        print("docstring:", prompt)
        print("CodeBERT Output:", codebert_output)
        print("-" * 40)

if __name__ == "__main__":
    main()

