import random

def reconstruct_sentence(sentence,replacements):
    for old, options in replacements.items():
        if old in sentence:
            sentence = sentence.replace(old, random.choice(options))
    
    return sentence