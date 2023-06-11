import nltk
import numpy as np
import pandas as pd

nltk.download('words')
from nltk.corpus import words

df = pd.read_csv('mixed_files.csv', header=None, on_bad_lines='skip')

# Rename your columns
df.columns = ['domain', 'classification']

# Create a set of English words
english_words = set(words.words())

# Feature extraction functions
def calculate_length(domain):
    return len(domain)

def calculate_entropy(domain):
    # Get probability of chars in domain
    prob = [float(domain.count(c)) / len(domain) for c in dict.fromkeys(list(domain))]
    # Calculate the entropy
    entropy = - sum([ p * np.log2(p) for p in prob])
    return entropy

def count_vowels(domain):
    return sum([1 for char in domain if char in 'aeiou'])

def count_consonants(domain):
    return sum([1 for char in domain if char in 'bcdfghjklmnpqrstvwxyz'])

def count_digits(domain):
    return sum([1 for char in domain if char.isdigit()])

def count_non_alphanumeric(domain):
    return sum([1 for char in domain if not char.isalnum()])

def count_english_words(domain):
    domain_words = domain.split('.')
    english_word_count = sum([1 for word in domain_words if word in english_words])
    return english_word_count

# Apply the functions to extract features
df['length'] = df['domain'].apply(calculate_length)
df['entropy'] = df['domain'].apply(calculate_entropy)
df['vowel_count'] = df['domain'].apply(count_vowels)
df['consonant_count'] = df['domain'].apply(count_consonants)
df['digit_count'] = df['domain'].apply(count_digits)
df['non_alphanumeric_count'] = df['domain'].apply(count_non_alphanumeric)
df['english_word_count'] = df['domain'].apply(count_english_words)

print(df)

# Save the DataFrame to a new txt file
df.to_csv('extracted_features.csv', sep='\t', index=False)
