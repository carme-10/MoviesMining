import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SentenceBert:

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.data = self.data.reset_index(drop=True)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')     
        self.output_col: str = 'sentence_embeddings'

    def encode(self, col:str):
        tqdm.pandas()
        self.data[self.output_col] = self.data[col].progress_apply(lambda x: self.model.encode(x))
        joblib.dump(self.data, '../output/data_with_embedding.pkl')

    # similarità coseno tra un embedding e tutti gli altri
    def cosine_similarity(self, input_embedding):
        similarities = []
        for emb in self.data[self.output_col]:
            sim = cosine_similarity([input_embedding], [emb])[0][0]
            similarities.append(sim)
        return similarities
    
    def get_most_similar(self, input_index:int, n:int):

        input_embedding = self.data[self.output_col][input_index]
        similarities = np.array(self.cosine_similarity(input_embedding))

        most_similar_index = similarities.argsort() #ordina gli elementi restituendo i rispettivi indici
        most_similar_index = most_similar_index[most_similar_index != input_index]
        most_similar_index = most_similar_index[::-1] #inverto l'array per avere ordine decrescente
        most_similar_index = most_similar_index[:n]

        most_similar = self.data.iloc[most_similar_index].copy()
        most_similar['cosine_similarity'] = similarities[most_similar_index]
        return most_similar

    def get_indexes(self, col:str, value):
        return self.data[self.data[col] == value].index