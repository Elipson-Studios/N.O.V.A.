import os
import requests
from bs4 import BeautifulSoup
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.data = self.create_sequences()

    def create_sequences(self):
        sequences = []
        for i in range(0, len(self.text) - self.seq_length):
            seq = self.text[i:i + self.seq_length]
            label = self.text[i + self.seq_length]
            sequences.append((seq, label))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        seq_idx = [self.char_to_idx[char] for char in seq]
        label_idx = self.char_to_idx[label]
        return torch.tensor(seq_idx), torch.tensor(label_idx)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                  weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())
        return hidden

class Endpoint:
    def __init__(self):
        # Load the on-site knowledge from the JSON file
        self.internal_knowledge = self.load_knowledge_base()  # Load any internal knowledge base
        self.config = self.load_config()  # Load config to check if search is allowed

        # Model and tokenizer initialization
        self.seq_length = 100
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.num_layers = 2
        self.model = None
        self.dataset = None
        self.dataloader = None

    def load_knowledge_base(self):
        """Load or define the internal knowledge base from a JSON file."""
        knowledge_base = {}
        knowledge_file = os.path.join(os.path.dirname(__file__), 'Depend/json/onSiteData.json')
        
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

        inputs = torch.tensor([self.dataset.char_to_idx[char] for char in input_with_context], dtype=torch.long).unsqueeze(0).to(device)
        hidden = self.model.init_hidden(1)
        output, hidden = self.model(inputs, hidden)
        predicted_char_idx = torch.argmax(output, dim=1).item()
        predicted_char = self.dataset.idx_to_char[predicted_char_idx]
        return predicted_char

    def fine_tune_model(self, train_file, output_dir):
        """Fine-tune the LSTM model on a custom dataset."""
        # Load the dataset
        with open(train_file, 'r') as f:
            text = f.read()
        self.dataset = TextDataset(text, self.seq_length)
        self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)

        # Initialize the model
        self.model = LSTMModel(len(self.dataset.chars), self.embedding_dim, self.hidden_dim, self.num_layers).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model
        self.model.train()
        for epoch in range(10):  # Number of epochs
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                hidden = self.model.init_hidden(inputs.size(0))  # Initialize hidden state with the correct batch size
                hidden = tuple([each.data for each in hidden])
                self.model.zero_grad()
                output, hidden = self.model(inputs, hidden)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

if __name__ == "__main__":
    endpoint = Endpoint()
    # Fine-tune the model on a custom dataset
    endpoint.fine_tune_model(train_file="Depend/json/onSiteData.json", output_dir="./fine_tuned_model")
    endpoint.interact()