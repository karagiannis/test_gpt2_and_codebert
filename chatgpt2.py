#First run gpt2.py to created the gpt2 full model
#then run python3 chatgpt2.py --model_path ./fine_tuned_model


import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to GPT-2 model checkpoint')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Name of GPT-2 variant')
    args = parser.parse_args()

    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

    # Load the GPT-2 model from the specified checkpoint
    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    print("ChatBot Console - Type 'exit' to end the conversation")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("ChatBot: Goodbye!")
            break
        # Tokenize the user input and generate a response
        input_ids = tokenizer.encode(user_input, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # All tokens are attended to
        response_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=100)[0]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)


        print("ChatBot:", response)

if __name__ == "__main__":
    main()
