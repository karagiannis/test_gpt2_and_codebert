from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from dataset import combined_dataset



def generate_response(prompt, model_path):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(model_path)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        temperature=0.8,  # Experiment with different temperature values
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def write_output(output_file_path, combined_dataset, model_path):
    try:
        with open(output_file_path, "w") as output_file:
            for example in combined_dataset:
                prompt = example["prompt"]
                desired_output = example["desired_output"]
                response = generate_response(prompt, model_path)

                output_file.write(f"Prompt: {prompt}\n")
                output_file.write(f"Trained Response: {response}\n")
                output_file.write(f"Desired Output: {desired_output}\n")
                output_file.write("\n")  # Add a line break between entries
    except Exception as e:
        print(f"An error occurred: {e}")




def main():

    model_path = "./fine_tuned_model"  # Path to the directory where your trained model is saved
    output_file_path = "prompts_trainedResponses_and_desiredOutputs.txt"
    write_output(output_file_path, combined_dataset, model_path)


if __name__ == "__main__":
    main()

