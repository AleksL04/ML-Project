import pandas as pd
import numpy as np
import spacy
import string
from gensim.models import KeyedVectors

def tokenize_df(df, nlp):
    df['text'] = df['text'].str.lower()
    tokens_list = []
    for doc in nlp.pipe(df['text'].astype(str), batch_size=1000):
        tokens_list.append([token.text for token in doc if not token.is_space])

    df['text_split'] = tokens_list
    return df

def add_embedding(df, word2vec_dict):
    np.random.seed(42)
    random_vec = np.random.rand(100)*2-1

    embedding_list = []
    for tokens in df['text_split']:
        embedding = []
        for token in tokens:
            if token in word2vec_dict:
                embedding.append(word2vec_dict[token])
            else:
                embedding.append(random_vec)
        embedding_list.append(embedding)
    df['embedding'] = embedding_list
    return df

def add_manual_features(df):
    # Calculate features
    df['quote_count'] = df['text'].astype(str).apply(lambda x: x.count("'") + x.count('"'))
    
    def get_density(text):
        text = str(text)
        if len(text) == 0: return 0.0
        punct_count = sum(1 for char in text if char in string.punctuation)
        return punct_count / len(text)
    
    df['punct_density'] = df['text'].apply(get_density)
    
    # Normalize features (Crucial for Dense layers)
    # Simple max scaling to keep them roughly 0-1
    if df['quote_count'].max() > 0:
        df['quote_count'] = df['quote_count'] / df['quote_count'].max()
    # punct_density is already a ratio 0-1
    
    return df

def fix_vector_length(df, max_len=50, vector_size=100):
    X = np.zeros((len(df), max_len, vector_size))

    for i, seq in enumerate(df['embedding']):
        length = min(len(seq), max_len)
        if length > 0:
            X[i, :length, :] = np.array(seq)[:length]
    return X

def process_df(df):
    model_path = 'word2vec_model/glove-wiki-gigaword-100.vectors'
    word2vec_dict = KeyedVectors.load(model_path)
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])

    df = df.copy()

    MAX_LEN = 50
    VECTOR_SIZE = 100

    df = tokenize_df(df, nlp)
    df = add_embedding(df, word2vec_dict)
    df = add_manual_features(df)

    # 1. Text Input (3D Array)
    X_text = fix_vector_length(df, MAX_LEN, VECTOR_SIZE)
    
    # 2. Features Input (2D Array)
    X_feat = df[['quote_count', 'punct_density']].values

    return X_text, X_feat