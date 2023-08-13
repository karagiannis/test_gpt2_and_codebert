# test_fine_tuned_gpt2.py
# This script uses a fine-tuned GPT-2 model to generate responses for a set of prompts and compares them with desired outputs.

# Import necessary libraries and modules
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from dataset import combined_dataset  # Import the custom dataset defined in 'dataset.py'

# Function to generate a response using the fine-tuned GPT-2 model
def generate_response(prompt, model_path):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Tokenize the input prompt and generate an attention mask
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    # Generate output using the fine-tuned model
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=100,              # Maximum length of the generated response in tokens
        num_return_sequences=1,     # Number of independent sequences to generate
        no_repeat_ngram_size=2,     # Prevent n-grams of size 2 from repeating in the response
        top_k=50,                   # Select from the top-k likelihood tokens at each step
        temperature=0.8,            # Controls randomness: higher values increase randomness
        pad_token_id=tokenizer.eos_token_id,  # ID of the padding token (end-of-sentence token)
    )

    # Decode the generated output and return it
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Function to write generated responses, prompts, and desired outputs to a file
def write_output(output_file_path, combined_dataset, model_path):
    try:
        with open(output_file_path, "w") as output_file:
            for example in combined_dataset:
                prompt = example["prompt"]
                desired_output = example["desired_output"]
                response = generate_response(prompt, model_path)

                # Write prompt, trained response, and desired output to the file
                output_file.write(f"Prompt: {prompt}\n")
                output_file.write(f"Trained Response: {response}\n")
                output_file.write(f"Desired Output: {desired_output}\n")
                output_file.write("\n")  # Add a line break between entries
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to run the testing process
def main():
    model_path = "./fine_tuned_model"  # Path to the directory where the trained model is saved
    output_file_path = "prompts_trainedResponses_and_desiredOutputs.txt"
    write_output(output_file_path, combined_dataset, model_path)

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
