import pandas as pd
import numpy as np
import operator

class Preprocessing:

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def clean(self, col:list[str], drop:bool, duplicates_to_check:list[str]):
        if drop:
            self.data = self.data.drop(columns=col)
        else:
            self.data = self.data[col]
        self.data = self.data.dropna()
        if duplicates_to_check is None:
            duplicates_to_check = self.data.columns
        self.data = self.data.drop_duplicates(subset=duplicates_to_check)
        self.data = self.data.reset_index(drop=True)

    def filter(self, col:str, value, operator):
        self.data = self.data[operator(self.data[col], value)]
        self.data = self.data.reset_index(drop=True)

    def date_manipulation(self, col:str):
        self.data[col] = pd.to_datetime(self.data[col])
        self.data['day_of_week'] = self.data[col].dt.dayofweek
        self.data['day_of_the_year'] = self.data[col].dt.dayofyear
        self.data['year'] = self.data[col].dt.year
        self.data = self.data.drop(columns=[col])
        self.data = self.data.reset_index(drop=True)

    def string_to_list(self, col:str):
        self.data[col] = self.data[col].str.split(', ')

    def frequency_encoding(self, col:str):

        frequency = self.data[col].value_counts(normalize=True)
        self.data[col] = self.data[col].map(frequency)

    #frequency encoding su colonna che contiene liste di elementi
    def frequency_encoding_list(self, col:str, string_to_list:bool):
        if string_to_list:
            self.string_to_list(col)
        frequency = self.data[col].explode().value_counts(normalize=True)
        self.data[col] = self.data[col].apply(lambda x: np.mean([frequency[i] for i in x]))

    def one_hot_encoding_list(self, col:str, string_to_list:bool):

        if string_to_list:
            self.string_to_list(col)

        self.data = self.data.join(pd.get_dummies(self.data[col].explode()).groupby(level=0).sum())
        self.data = self.data.drop(columns=col)

    def run(self, operations: list[dict]):

        print("Numero di righe e colonne prima del preprocessing: ", self.data.shape)
        for operation in operations:
            method = operation["type"]
            method(self, **operation["params"])
        print("Numero di righe e colonne dopo il preprocessing: ", self.data.shape)
        print("Nome delle colonne: ", self.data.columns)


