import pandas as pd 
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Open raw text file 
with open("step-01-data-processing/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Before Cleaning:\n", raw_text[:200])


# Cleaning function
def clean_text_full(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # stopwords remove
    words = text.split()
    #words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Apply cleaning
cleaned_text = clean_text_full(raw_text)

print("\nAfter Cleaning:\n", cleaned_text[:200])


# Save cleaned text
with open("step-01-data-processing/cleaned.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)


