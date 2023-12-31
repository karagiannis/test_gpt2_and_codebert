# optimizer.py
# This script performs hyperparameter optimization for the fine-tuned GPT-2 model using integer programming.

# Import necessary modules and functions
import string
import trainer_gpt2
import test_fine_tuned_gpt2
from dataset import combined_dataset
from scipy.optimize import differential_evolution

DEBUG = False  # Debug flag to control printing debug information

# Function to read lines from a file
def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

# Function to extract data (prompts, trained responses, desired outputs) from lines
def extract_data(lines):
    prompts, trained_responses, desired_outputs = [], [], []
    i = 0
    while i < len(lines):
        # Extract prompts
        if lines[i].startswith("Prompt:"):
            # Handle multi-line prompts
            prompt = lines[i][8:].strip()  # Extract the prompt text after "Prompt:"
            i += 1  # Move to the next line
            while i < len(lines) and not lines[i].startswith("Trained Response:"):
                prompt += " " + lines[i].strip()  # Append to the existing prompt
                i += 1
            prompts.append(prompt)

        # Extract trained responses
        elif lines[i].startswith("Trained Response:"):
            # Handle multi-line trained responses
            trained_response = lines[i][18:].strip()
            i += 1  # Move to the next line
            while i < len(lines) and not lines[i].startswith("Desired Output:"):
                trained_response += " " + lines[i].strip()  # Append to the existing trained response
                i += 1
            trained_responses.append(trained_response)

        # Extract desired outputs
        elif lines[i].startswith("Desired Output:"):
            # Handle multi-line desired outputs
            desired_output = lines[i][16:].strip()
            i += 1  # Move to the next line
            while i < len(lines) and lines[i].strip() != "":
                desired_output += " " + lines[i].strip()  # Append to the existing desired output
                i += 1
            desired_outputs.append(desired_output)

        else:
            i += 1

    return prompts, trained_responses, desired_outputs

# Function to subtract prompt from response
def subtract_prompt(prompt, response):
    return response[len(prompt):].strip()

# Function to calculate penalty score
def calculate_penalty(response, desired_output):
    length_error = abs(len(desired_output) - len(response))
    character_error = sum(abs(ord(d) - ord(r)) for d, r in zip(desired_output, response))
    total_error = length_error + character_error
    return total_error

# Objective function for differential evolution optimization
def objective_function(x):
    num_train_epochs, batch_size, save_limit, save_steps = x

    # Call trainer_gpt2 with provided hyperparameters
    trainer_gpt2.train_gpt2(combined_dataset, num_train_epochs, batch_size, save_limit, save_steps)

    # Call test_fine_tuned_gpt2 to get output file
    model_path = "./fine_tuned_model"
    output_file_path = "prompts_trainedResponses_and_desiredOutputs.txt"
    test_fine_tuned_gpt2.write_output(output_file_path, combined_dataset, model_path)

    # Extract data from output file
    lines = read_file(output_file_path)
    prompts, trained_responses, desired_outputs = extract_data(lines)

    # Calculate penalty scores
    penalty_scores = []
    for prompt, response, desired_output in zip(prompts, trained_responses, desired_outputs):
        response_without_prompt = subtract_prompt(prompt, response)
        penalty_score = calculate_penalty(response_without_prompt, desired_output)
        penalty_scores.append(penalty_score)
    total_penalty_score = sum(penalty_scores)
    return total_penalty_score

# Rounded objective function for integer programming
def rounded_objective(x):
    rounded_x = [int(val) for val in x]
    return objective_function(rounded_x)

# ...

def objective_function2(x):
    num_train_epochs, batch_size, save_limit, save_steps = x
    num_train_epochs_opt = 3
    batch_size_opt = 4
    save_limit_opt = 2
    save_steps_opt = 10
    min_total_penalty_score = 1000
    for i in range(2, 5):
        for j in range(3, 6):
            for k in range(1, 4):
                for l in range(9, 12):
                    # Call trainer_gpt2 with the provided hyperparameters
                    trainer_gpt2.train_gpt2(combined_dataset, num_train_epochs=i, batch_size=j, save_limit=k, save_steps=l)

                    # Call test_fine_tuned_gpt2 to get the output file
                    model_path = "./fine_tuned_model"  # Path to the directory where your trained model is saved
                    output_file_path = "prompts_trainedResponses_and_desiredOutputs.txt"
                    test_fine_tuned_gpt2.write_output(output_file_path, combined_dataset, model_path)

                    lines = read_file(output_file_path)
                    prompts, trained_responses, desired_outputs = extract_data(lines)
                    penalty_scores = []
                    for prompt, response, desired_output in zip(prompts, trained_responses, desired_outputs):
                        response_without_prompt = subtract_prompt(prompt, response)
                        penalty_score = calculate_penalty(response_without_prompt, desired_output)
                        penalty_scores.append(penalty_score)
                        total_penalty_score = sum(penalty_scores)
                        if total_penalty_score < min_total_penalty_score:
                            min_total_penalty_score = total_penalty_score
                            print("i,j,k,l:", i, j, k, l)
                            num_train_epochs_opt = i
                            batch_size_opt = j
                            save_limit_opt = k
                            save_steps_opt = l
    return num_train_epochs_opt, batch_size_opt, save_limit_opt, save_steps_opt

# Convert the objective function to a callable function
# Integer programming objective function
def integer_programming_objective(x):
    x_values = x
    return rounded_objective(x_values)

# Define parameter bounds for optimization
bounds = [(1, 10), (1, 16), (1, 10), (1, 20)]

# Main function
if __name__ == "__main__":
    output_file_path = "prompts_trainedResponses_and_desiredOutputs.txt"
    lines = read_file(output_file_path)
    prompts, trained_responses, desired_outputs = extract_data(lines)

    # Calculate penalty scores and total penalty score
    penalty_scores = []
    for prompt, response, desired_output in zip(prompts, trained_responses, desired_outputs):
        response_without_prompt = subtract_prompt(prompt, response)
        if DEBUG:
            print("response_without_prompt:", response_without_prompt)
        penalty_score = calculate_penalty(response_without_prompt, desired_output)
        penalty_scores.append(penalty_score)
    total_penalty_score = sum(penalty_scores)

    # Use differential_evolution for hyperparameter optimization
    result = differential_evolution(integer_programming_objective, bounds, seed=42)

    print("Optimal solution:", result.x)  # The optimal hyperparameters
    #print(objective_function2(initial_guess))

