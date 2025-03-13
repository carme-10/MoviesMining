import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class SentenceBert:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.model = SentenceTransformer('all-MiniLM-L6-v2')     
        self.output_col: str = 'sentence_embeddings'
        self.similarity_matrix = None

    def encode(self, col:str):
        tqdm.pandas()
        self.data[self.output_col] = self.data[col].progress_apply(lambda x: self.model.encode(x))
        joblib.dump(self.data, '../output/data_with_embedding.pkl')

    # funzione per calcolare la matrice delle similarità coseno tra gli embeddings
    def cosine_similarity_matrix(self, embeddings):
        similarity_matrix = cosine_similarity(embeddings)
        self.similarity_matrix = similarity_matrix
        joblib.dump(similarity_matrix, '../output/cosine_similarity_matrix.pkl')

    def run(self, col:str):
        self.encode(col)
        embeddings = self.data[self.output_col].to_list()
        self.cosine_similarity_matrix(embeddings)