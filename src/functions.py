from Preprocessing import Preprocessing
from Regressor import Regressor
from operations import operations_model, operations_sbert
from SentenceBert import SentenceBert

def regression(df, target: str):

    preprocessing = Preprocessing(df)
    preprocessing.run(operations_model)
    regressor = Regressor(preprocessing.data.drop(columns=target), preprocessing.data[target])
    regressor.run()

def sentences_encoding(df, col: str):

    preprocessing = Preprocessing(df)
    preprocessing.run(operations_sbert)
    sbert = SentenceBert(preprocessing.data)
    sbert.encode(col)
    return sbert

def reccomentation(sbert:SentenceBert, title: str):

    input_indexes = sbert.get_indexes('title', title)
    if len(input_indexes) == 0:
        print("Movie not found")
    for i in input_indexes:
        input_movie = sbert.data.loc[i]
        print()
        print(f"Index: {i}")
        print(f"Title: {input_movie['title']}")
        print(f"Overview: {input_movie['overview']}")
        output = sbert.get_most_similar(i, 10).drop(columns=['imdb_id', 'sentence_embeddings'])
        output.to_csv(f"../output/{title}_{i}_most_similar.csv", index=True)
