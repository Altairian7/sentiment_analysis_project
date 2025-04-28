# data_augmentation.py
from nltk.corpus import wordnet
import random
import nltk

nltk.download('wordnet')

def synonym_replacement(sentence, n=2):
    words = sentence.split()
    new_words = words.copy()
    
    for _ in range(n):
        word = random.choice(words)
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if w == word else w for w in new_words]
    return ' '.join(new_words)

if __name__ == "__main__":
    text = input("Enter text: ")
    augmented = synonym_replacement(text)
    print(f"Augmented: {augmented}")
