import pandas as pd
import spacy
import os

def get_keywords(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(str(text).lower())
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def tokenize_segments(dataFiles, inplace):
    nlp = spacy.load("en_core_web_sm")
    def get_keywords(text):
        doc = nlp(str(text).lower())
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    def generate_keywords_from_description(df):
        df["Keywords"] = df["Description"].apply(get_keywords)
        return df

    processed_dfs = []
    
    for file_path in dataFiles:
        print(f"Working on {file_path}...")
        df = pd.read_csv(file_path)
        df = generate_keywords_from_description(df)
        
        if inplace:
            df.to_csv(file_path, index=False)
        else:
            filename = os.path.basename(file_path)
            tokenized_name = f"tokenized-{filename}"
            df.to_csv(tokenized_name, index=False)
        
        processed_dfs.append(df)
        print(f"{file_path} is complete")

    return processed_dfs
