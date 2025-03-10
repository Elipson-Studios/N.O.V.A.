import sys
import requests
from bs4 import BeautifulSoup
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPU is available.")
else:
    print("Warning: No GPU found. The code will run on the CPU.")

incentive = 50
goal = 100
fail = 0 # Amount of incentive to destroy the network
filterSensitivity = 0.1 # The lower the value, the more perfection required.

def search(query, num_results=10):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(f"https://www.google.com/search?q={query}&num={num_results}", headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for g in soup.find_all(class_='BVG0Nb'):
        results.append(g.text)
    return results

class Endpoint:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # Ensure pad_token_id is set

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def interact(self):
        while True:
            value = input("you: ")
            if value.lower() == 'exit':
                print("Goodbye!")
                break
            response = self.generate_response(value)
            print(f"bot: {response}")

    def generate_response(self, input_text):
        input_text = self.preprocess_text(input_text)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(device)
        attention_mask = (inputs != self.tokenizer.pad_token_id).long().to(device)
        outputs = self.model.generate(
            inputs, 
            max_length=150, 
            num_return_sequences=1, 
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask,
            do_sample=True,  # Enable sampling to introduce randomness
            top_k=50,        # Consider only the top 50 tokens
            top_p=0.95       # Use nucleus sampling with probability 0.95
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    endpoint = Endpoint()
    endpoint.interact()