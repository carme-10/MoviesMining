from Preprocessing import Preprocessing
import pandas as pd
import operator
from Regressor import Regressor

def main():

    path = open("dataset_path.txt", "r").read()
    df = pd.read_csv(path)

    preprocessing = Preprocessing(df)
    columns_to_remove = ['title', 'backdrop_path', 'homepage',
       'imdb_id', 'original_title',
       'poster_path', 'tagline', 'budget',
       'revenue',
       'keywords']
    preprocessing.clean(columns_to_remove)
    preprocessing.filter('status', 'Released', operator.eq)
    preprocessing.filter('vote_count', 20, operator.ge)

    preprocessing.date_manipulation('release_date')
    
    preprocessing.one_hot_encoding_list('genres', True)
    preprocessing.frequency_encoding('original_language')
    preprocessing.frequency_encoding_list('production_companies', True)
    preprocessing.frequency_encoding_list('production_countries', True)
    preprocessing.frequency_encoding_list('spoken_languages', True)

    preprocessing.clean(['status', 'id', 'overview', 'popularity', 'vote_count'])

    print("Numero di righe e colonne dopo il preprocessing: ", preprocessing.data.shape)
    print("Nome delle colonne: ", preprocessing.data.columns)
    print("Primo record del dataset: ", preprocessing.data.head(1).to_string())

    target = 'vote_average'
    regressor = Regressor(preprocessing.data.drop(columns=target), preprocessing.data[target])
    regressor.run()


main()