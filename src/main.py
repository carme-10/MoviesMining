from functions import regression, sentences_encoding, recommendation, yes_or_no
import pandas as pd

def main():

    path = open("dataset_path.txt", "r").read()
    df = pd.read_csv(path)

    answer = yes_or_no("Do you want to use the saved model? Type 'yes' or 'no' (Type 'no' the first time): ")
    regression(df, 'vote_average', answer)

    answer = yes_or_no("Do you want to use the saved embeddings? Type 'yes' or 'no' (Type 'no' the first time): ")
    sbert = sentences_encoding(df, 'overview', answer)
    while True:
        title = input("Insert the title of the movie: ")
        recommendation(sbert, title)

main()