# trainer_gpt2.py
# This script fine-tunes a pre-trained GPT-2 model on a custom dataset for text generation tasks.

# Import necessary libraries and modules
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
import argparse
from torch.optim import AdamW
from dataset import combined_dataset  # Import the custom dataset defined in 'dataset.py'

# Function to train the GPT-2 model on the provided dataset
def train_gpt2(training_dataset, num_train_epochs, batch_size, save_limit, save_steps):
    # Load pre-trained GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Set maximum length for tokenized examples
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
        output_dir="./output",              # Directory to save training outputs
        num_train_epochs=num_train_epochs,   # Number of training epochs
        per_device_train_batch_size=batch_size,  # Batch size for training
        save_total_limit=save_limit,        # Maximum number of models to keep
        save_steps=save_steps,              # Number of steps between saving models
        logging_dir="./logs",               # Directory to save training logs
    )

    # Create AdamW optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        optimizers=(optimizer, None)
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")  # Save the model in the 'fine_tuned_model' directory

# Main function to run the training process
def main():
    # Parse command line arguments for training parameters
    parser = argparse.ArgumentParser(description="Trains GPT-2 model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--save_limit", type=int, default=2, help="Save total limit")
    parser.add_argument("--save_steps", type=int, default=10, help="Save steps")
    args = parser.parse_args()

    # Call the training function with provided arguments
    train_gpt2(
        combined_dataset,                   # Custom training dataset
        num_train_epochs=args.epochs,       # Number of training epochs
        batch_size=args.batch_size,         # Batch size for training
        save_limit=args.save_limit,         # Maximum number of models to save
        save_steps=args.save_steps          # Number of steps between saving models
    )

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
