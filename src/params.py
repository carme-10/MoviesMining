# parametri da testare
param_grid = {
    'n_estimators': [300],              
    'max_depth': [29, 30, 31],
    'min_samples_split': [5, 6, 7],  # Minimo numero di campioni per dividere un nodo
    'min_samples_leaf': [1, 2],  # Minimo numero di campioni in una foglia
    'max_features': ['sqrt']  # Numero massimo di feature considerate a ogni split
}