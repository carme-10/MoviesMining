from functions import regression, sentences_encoding, reccomentation
import pandas as pd

def main():

    path = open("dataset_path.txt", "r").read()
    df = pd.read_csv(path)

    regression(df, 'vote_average')
    sbert = sentences_encoding(df, 'overview')

    while True:
        title = input("Insert the title of the movie: ")
        reccomentation(sbert, title)

main()