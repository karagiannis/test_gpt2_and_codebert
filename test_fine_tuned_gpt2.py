from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


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


def main():
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
    model_path = "./fine_tuned_model"  # Path to the directory where your trained model is saved
    with open("prompts_trainedResponses_and_desiredOutputs.txt", "w") as output_file:

        for example in combined_dataset:
            prompt = example["prompt"]
            desired_output = example["desired_output"]
            response = generate_response(prompt, model_path)
            print("prompt:", prompt)
            print("trained response:", response)
            print("desired output:", desired_output)

            output_file.write(f"Prompt: {prompt}\n")
            output_file.write(f"Trained Response: {response}\n")
            output_file.write(f"Desired Output: {desired_output}\n")
            output_file.write("\n")  # Add a line break between entries


if __name__ == "__main__":
    main()

