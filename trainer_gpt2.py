#trainer_gpt2.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
import argparse
from torch.optim import AdamW
from dataset import combined_dataset



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
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_total_limit=2,
        save_steps=10,
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
