from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from params import param_grid

class Regressor:

    def __init__(self, features_data, target_data):
        self.X = features_data
        self.y = target_data

        self.grid_search = None
        self.best_model_info = {}

    def best_parameters_computing(self):

        X = self.X
        y = self.y

        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbose=3)
        grid_search.fit(X, y)

        self.grid_search = grid_search
        joblib.dump(grid_search, "../output/grid_search.pkl")
 
    def evaluate(self):

        best_model = self.grid_search.best_estimator_
        X = self.X
        y = self.y

        mse = -self.grid_search.best_score_
        rmse = np.sqrt(mse)

        y_pred = best_model.predict(X) # predizioni sul dataset completo

        info = {
            "MSE on test set": mse,
            "RMSE on test set": rmse,
            "Parameters": self.grid_search.best_params_,
            "Variance target": np.var(y),
            "MSE on the whole dataset": mean_squared_error(y, y_pred)
        }

        self.best_model_info = info

    def save_output(self):
        with open("../output/best_model_info.txt", "w") as f:
            f.write(str(self.best_model_info))
    
    def run(self, use_saved_model:bool):
        if use_saved_model:
            self.grid_search = joblib.load("../output/grid_search.pkl")
        else:
            self.best_parameters_computing()
        self.evaluate()
        self.save_output()