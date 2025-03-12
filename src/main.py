from Preprocessing import Preprocessing
import pandas as pd
import numpy as np
import operator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib

def create_model(X, y):
    
    model = RandomForestRegressor(random_state=42)

    # parametri da testare
    param_grid = {
        'n_estimators': [50, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbose=3)
    grid_search.fit(X, y)

    mse = -grid_search.best_score_
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Varianza target: {np.var(y):.4f}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X) # predizioni sul dataset completo
    mse = mean_squared_error(y, y_pred)
    print(f"Errore quadratico medio (MSE) sul dataset completo: {mse}")

    joblib.dump(best_model, "model.pkl")

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
    
    preprocessing.string_to_list('genres')
    preprocessing.one_hot_encoding_list('genres')

    preprocessing.frequency_encoding('original_language')

    preprocessing.string_to_list('production_companies')
    preprocessing.frequency_encoding_list('production_companies')

    preprocessing.string_to_list('production_countries')
    preprocessing.frequency_encoding_list('production_countries')

    preprocessing.string_to_list('spoken_languages')
    preprocessing.frequency_encoding_list('spoken_languages')

    preprocessing.clean(['status', 'id', 'overview', 'popularity', 'vote_count'])

    print("Numero di righe e colonne dopo il preprocessing: ", preprocessing.data.shape)
    print("Nome delle colonne: ", preprocessing.data.columns)
    print("Primo record del dataset: ", preprocessing.data.head(1).to_string())

    target = 'vote_average'
    create_model(preprocessing.data.drop(columns=target), preprocessing.data[target])


main()