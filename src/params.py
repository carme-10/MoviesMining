# parametri da testare
param_grid = {
    'n_estimators': [200],              
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],  # Minimo numero di campioni per dividere un nodo
    'min_samples_leaf': [1, 2, 4],  # Minimo numero di campioni in una foglia
    'max_features': ['sqrt', 'log2', None]  # Numero massimo di feature considerate a ogni split
}