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
    sbert.run('overview')


main()