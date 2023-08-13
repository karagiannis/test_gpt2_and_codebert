# Test GPT-2 and CodeBERT

This repository contains code and scripts to experiment with fine-tuning GPT-2 and CodeBERT models, as well as automating the training process and evaluating the model's responses.

## Contents

- `trainer_gpt2.py`: Script to fine-tune the GPT-2 model and save the fine-tuned model.
- `test_finetuned_gpt2.py`: Script to test the fine-tuned GPT-2 model on given prompts.
- `trainer_codebert.py`: Script to fine-tune the CodeBERT model and save the fine-tuned model.
- `test_finetuned_codebert.py`: Script to test the fine-tuned CodeBERT model on code-related prompts.
- `optimizer.py`: Script to automate parameter tuning using an optimizer based on response evaluations.
- 

## Getting Started

1. Install the required libraries and dependencies.
2. Fine-tune the models using the training scripts.
3. Test the fine-tuned models using the testing scripts.
4. Use the optimizer script to automate parameter tuning.

## Usage

1. Clone this repository.
2. Navigate to the directory of the desired script.
3. Run the script using Python: `python script_name.py`
4. Running this on vast.ai: 
Create an account and store your public ssh key (ends .pub) at your account page at 
https://www.vast.ai
Generate the key if you don't have one by running command
'ssh-keygen -t rsa' on your computer. Hit enter to accept all
the default values when questioned. Paste the public ssh key into your account page 
'Change SSH Key'.
Choose template 'cuda:12.0.1-devel-ubuntu20.04' for cuda support and rent a computer
when directed to the 'Search' page.
It takes a couple of minutes for the enviroment to be set up and when it is done
you are given the option of logging in via ssh or ssh via proxy.
Choose ssh without proxy if you do not know what you are doing.
The command given is 
```sh
ssh -p <remote port> root@<remote IP-adress> -L <your port number>:localhost:<their port number'>
```
It can look like this: 
```sh 
s   sh -p 40170 root@198.16.187.151 -L 8080:localhost:8080'
```
which means connecting from client port 8080 to the remote hosts port 40170 
which is set to be forwarded to their port 8080. Effectively communication
is done via ports 8080 on both sides.
Upload your files with command
```sh
scp -P <their remote port> <path to your file> root@<their IP adress>:/root/
```
It can look like this:
```sh
scp -P 40170 trainer_gpt2.py root@198.16.187.151:/root/'
```
Your upload is placed in the remote root folder.
To retrieve files one simple change the scp command like this
```sh
scp -P <their port number> root@<their IP adress>:<file path remote computer> 
<folder path local computer>
```
It can look like this:
```sh 
scp -P 40170 root@198.16.187.151:/root/trainer_gpt2.py ~/Desktop/
```
Install pip3 and then continue to install dependencies that python3 
is asking for. To install `pip3`, you can use the package manager for your system.
For example, on Debian-based systems, you can run:

```sh
sudo apt-get update
sudo apt-get install python3-pip

```


## Contributions

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please feel free to create an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).


