import pandas as pd
import numpy as np
import operator

class Preprocessing:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean(self, col:list[str]):
        self.data = self.data.drop(columns=col)
        self.data = self.data.dropna()
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
    def frequency_encoding_list(self, col:str):
        frequency = self.data[col].explode().value_counts(normalize=True)
        self.data[col] = self.data[col].apply(lambda x: np.mean([frequency[i] for i in x]))

    def one_hot_encoding_list(self, col:str):

        self.data = self.data.join(pd.get_dummies(self.data[col].explode()).groupby(level=0).sum())
        self.data = self.data.drop(columns=col)

    # def standardize(self, ax=0):
    #     standardize = lambda x: (x - x.mean()) / x.std()
    #     self.data = self.data.apply(standardize, axis=ax)