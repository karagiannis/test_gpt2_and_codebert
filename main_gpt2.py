# main_gpt2.py
import torch
from gpt2_model import tokenizer as gpt2_tokenizer, model as gpt2_model
from datasets import load_dataset
from transformers import pipeline



def generate_combined_output(prompt):
    # Tokenize the input prompt using the GPT-2 tokenizer
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)


    # Generate output using the GPT-2 model
    with torch.no_grad():
        gpt2_output = gpt2_model.generate(input_ids=input_ids, attention_mask=attention_mask)


    gpt2_output_text = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)


    print("gpt2_output_text:", gpt2_output_text)
    return gpt2_output_text




combined_dataset = [
    {
        "prompt": "Write a Python program to calculate the factorial of a number.",
        "desired_output": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
    },
    {
        "prompt": "What is the purpose of the 'if __name__ == '__main__':' statement in Python?",
        "desired_output": "The 'if __name__ == '__main__':' statement is used to check whether the Python script is being run as the main program or being imported as a module."
    },
    {
        "prompt": "Show me how to read a file in Python and print its contents.",
        "desired_output": "with open('filename.txt', 'r') as file:\n    contents = file.read()\n    print(contents)"
    },
    # Add more examples here...
]


def get_dataset(prog_lang, nat_lang):
    test_data = load_dataset("blindsubmissions/M2CRB")
    test_data = test_data.filter(
        lambda example: example["docstring_language"] == nat_lang
        and example["language"] == prog_lang
    )
    return test_data

def main():
    # Your code here, using both GPT-2 and CodeBERT models
    # Specify your desired programming and natural languages
    programming_language = "python"  # Change this to the desired language
    natural_language = "en"  # Change this to the desired language

    # Get the M2CRB dataset filtered by your desired languages
    m2crb_dataset = get_dataset(programming_language, natural_language)

    # Generate combined outputs for each example in the combined_dataset
    for example in combined_dataset:
        prompt = example["prompt"]
        combined_output = generate_combined_output(prompt)

        # Print or store the combined output
        print("Prompt:", prompt)
        print("Combined Output:", combined_output)
        print("-" * 40)

if __name__ == "__main__":
    main()
