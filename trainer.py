import sys
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
import numpy as np
import re

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
        self.model = self.build_model()
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=10000, output_dim=64),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

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
        sequences = self.tokenizer.texts_to_sequences([input_text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
        prediction = self.model.predict(padded_sequences)
        return f"Predicted response: {prediction[0][0]}"

if __name__ == "__main__":
    endpoint = Endpoint()
    endpoint.interact()
