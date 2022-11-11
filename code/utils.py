from difflib import SequenceMatcher

def to_lower(x):
    y = []
    for word in x:
        y.append(word.lower())
    
    return y

def is_in(word, word_list, thre=0.8):
    for x in word_list:
        if SequenceMatcher(a=x, b=word).ratio() > thre:
            return x
    
    return None

