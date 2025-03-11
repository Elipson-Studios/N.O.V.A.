import os
import requests
from bs4 import BeautifulSoup
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
from heapq import nlargest

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'GPU' if device.type == 'cuda' else 'CPU'} is available.")

# Constants and config
incentive = 50
goal = 100
fail = 0
filterSensitivity = 0.1  # The lower the value, the more perfection required.

def search(query, num_results=5):
    """Performs a search and returns relevant results."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3' }
    
    # Avoid excessive requests. Limit the number of search calls and random queries.
    response = requests.get(f"https://www.google.com/search?q={query}&num={num_results}", headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Collect search result snippets
    results = [g.text for g in soup.find_all(class_='BVG0Nb')]
    return results

def summarize_text(texts, num_points=3):
    """Summarizes the given texts and returns the most important points."""
    combined_text = " ".join(texts)  # Combine all the results for processing
    word_freq = {}
    
    # Calculate word frequencies (basic method to find important words)
    for word in combined_text.split():
        word = word.lower()
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] += 1
    
    # Extract the most frequent words (basic key points)
    most_frequent = nlargest(num_points, word_freq, key=word_freq.get)
    
    # Return the top `num_points` words
    return most_frequent

class Endpoint:
    def __init__(self):
        # Load the on-site knowledge from the JSON file
        self.internal_knowledge = self.load_knowledge_base()  # Load any internal knowledge base
        self.config = self.load_config()  # Load config to check if search is allowed

        # Model and tokenizer initialization
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # Set pad_token_id once in constructor

    def load_knowledge_base(self):
        """Load or define the internal knowledge base from a JSON file."""
        knowledge_base = {}
        knowledge_file = os.path.join(os.path.dirname(__file__), 'onSiteData.json')
        
        if not os.path.exists(knowledge_file):
            print(f"Warning: {knowledge_file} does not exist.")
            return knowledge_base
        
        try:
            with open(knowledge_file, 'r') as f:
                knowledge_base = json.load(f)
        except Exception as e:
            print(f"Error loading onSiteData.json: {e}")
        return knowledge_base

    def load_config(self):
        """Load configuration settings to check if search is allowed."""  
        config = {"allowSearch": True}  # Default to allowing search if the config is not available
        config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        
        if not os.path.exists(config_file):
            print(f"Warning: {config_file} does not exist. Using default config.")
            return config
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config.json: {e}")
        return config

    def preprocess_text(self, text):
        """Preprocess the text to improve model's understanding and reduce noise."""
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return text

    def search_for_keywords(self, query):
        """Search for keywords in the internal knowledge base."""
        query = self.preprocess_text(query)
        matches = []

        for key, value in self.internal_knowledge.items():
            # If query is a substring or keyword within the key or value (case insensitive)
            if query in key.lower() or (isinstance(value, str) and query in value.lower()):
                matches.append(value)

        return matches

    def interact(self):
        """Main interaction loop with user."""
        print("You can start chatting now. Type 'exit' to end.")
        while True:
            value = input("you: ")
            if value.lower() == 'exit':
                print("Goodbye!")
                break
            
            response = self.generate_response(value)
            print(f"bot: {response}")

    def generate_response(self, input_text):
        """Generate a response from the model."""
        input_text = self.preprocess_text(input_text)

        # Check if the query is already in internal knowledge base (for keyword search)
        matches = self.search_for_keywords(input_text)
        if matches:
            return "\n".join(matches)

        # If search is not allowed, notify the user
        if not self.config.get("allowSearch", True):
            return "Searching is disabled. I can only use on-site data."

        # If search is allowed, perform a search
        print("I don't have that in my internal knowledge base. Let me look it up...")
        search_results = search(input_text)

        # Summarize and extrapolate main points from search results
        main_points = summarize_text(search_results)

        # Formulate a response based on both the summarized points and the model
        response = self.model_generate(input_text, main_points)
        return response

    def model_generate(self, input_text, main_points):
        """Generate response by combining summarized search points with model's knowledge."""
        context = " ".join(main_points)  # Combine the key points from the search results
        input_with_context = f"Query: {input_text}\nSummarized Points: {context}"

        inputs = self.tokenizer.encode(input_with_context, return_tensors="pt").to(device)
        attention_mask = (inputs != self.tokenizer.pad_token_id).long().to(device)

        outputs = self.model.generate(
            inputs, 
            max_length=150, 
            num_return_sequences=1, 
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask,
            do_sample=True,  # Enable randomness for more varied responses
            top_k=50,        # Use a smaller range for top_k
            top_p=0.9        # Adjust nucleus sampling
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    endpoint = Endpoint()
    endpoint.interact()
