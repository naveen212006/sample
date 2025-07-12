"""
Simple Retrieval‑Based Chatbot
Usage: python chatbot.py
Type a message and press Enter. Type 'quit' to exit.
Libraries: nltk, scikit-learn
"""

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download tokenizer model
nltk.download('punkt', quiet=True)

# Bot memory
CORPUS = [
    "hi",
    "hello",
    "how are you",
    "i am fine",
    "what is your name",
    "i am a chatbot created with python",
    "quit"
]

# Function to respond based on input
def respond(user_input: str) -> str:
    tokens = nltk.sent_tokenize(" ".join(CORPUS + [user_input.lower()]))
    vectorizer = TfidfVectorizer().fit_transform(tokens)
    similarity = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
    idx = similarity.argmax()
    if similarity[idx] == 0:
        return "I didn't understand that."
    return tokens[idx]

# Main chat function
def chat():
    print("Chatbot: Hello! Type 'quit' to exit.")
    while True:
        user = input("You: ")
        if user == "quit":
            print("Chatbot: Bye!")
            break
        print("Chatbot:", respond(user))

# Start chatbot
if __name__ == "__main__":
    chat()
