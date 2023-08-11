from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
import argparse
from torch.optim import AdamW


def train_gpt2(training_dataset, num_train_epochs, batch_size, save_limit, save_steps):
    # Load pre-trained GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # After loading the tokenizer, set a maximum length for tokenized examples
    max_length = 128  # Set the maximum length as needed

    # Tokenize the dataset
    tokenized_dataset = []
    for example in training_dataset:
        inputs = tokenizer(example["prompt"], max_length=max_length, return_tensors="pt", padding="max_length",
                           truncation=True)
        labels = tokenizer(example["desired_output"], max_length=max_length, return_tensors="pt", padding="max_length",
                           truncation=True)
        tokenized_dataset.append({"input_ids": inputs.input_ids, "labels": labels.input_ids})

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=5, #increase, prev. value 3
        per_device_train_batch_size=6, #increase, prev. value 4
        save_total_limit=4, #Increas, prev. value 2
        save_steps=12, #increase, prev. value 12
        logging_dir="./logs",
    )
    # Create AdamW optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        optimizers=(optimizer,None)
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")


def main():
    # Define your training dataset
    parser = argparse.ArgumentParser(description="Trains GPT-2 model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--save_limit", type=int, default=2, help="Save total limit")
    parser.add_argument("--save_steps", type=int, default=10, help="Save steps")

    args = parser.parse_args()
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

    # Call the training function
    train_gpt2(
        combined_dataset,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        save_limit=args.save_limit,
        save_steps=args.save_steps
    )


if __name__ == "__main__":
    main()
