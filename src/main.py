from Preprocessing import Preprocessing
import pandas as pd
from Regressor import Regressor
from operations import operations_model, operations_sbert
from SentenceBert import SentenceBert

def main():

    path = open("dataset_path.txt", "r").read()
    df = pd.read_csv(path)

    preprocessing = Preprocessing(df)
    preprocessing.run(operations_model)
    target = 'vote_average'
    regressor = Regressor(preprocessing.data.drop(columns=target), preprocessing.data[target])
    regressor.run()

    preprocessing = Preprocessing(df)
    preprocessing.run(operations_sbert)
    sbert = SentenceBert(preprocessing.data)
    sbert.encode('overview')

    title = input("Insert the title of the movie: ")
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


main()