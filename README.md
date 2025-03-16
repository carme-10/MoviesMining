# MoviesMining

Il codice del progetto è composto dai file: 

- Preprocessing.py
- operations.py
- Regressor.py
- params.py
- SentenceBert.py
- functions.py
- main.py

Dataset used: [Movies Dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)

## Preprocessing.py

La classe Preprocessing permette di effettuare il preprocessing del dataset.
E' possibile utilizzare il metodo *run* per effettuare varie operazioni di preprocessing tramite una lista di configurazione.

## operations.py

Il file contiene le liste di configurazione che verranno passate 
al metodo *run* della classe Preprocessing.

## Regressor.py

La classe Regressor permette di effettuare la regressione utilizzando il 
RandomForestRegressor.
Viene utilizzata la GridSearch per trovare la combinazione dei parametri più adatti, a partire da quelli specificati nel file params.py.

## params.py

In questo file ci sono i parametri da utilizzare per la GridSearch.

## SentenceBert.py

La classe SentenceBert permette di effettuare il calcolo degli embeddings utilizzando il modello SentenceBert.
Tramite gli embeddings è possibile calcolare la similarità tra le corrispondenti frasi.

## functions.py

Il file contiene le funzioni che vengono utilizzate nel main.py e che utilizzano le classi di cui si è parlato sopra.

## main.py

Bisogna creare un file *dataset_path.txt* nella cartella *src*, in cui si specifica il percorso del dataset.
Nel main.py viene usato questo percorso per importare il dataset.
Poi vengono usate le funzioni presenti in functions.py per effettuare il preprocessing, la regressione sul *vote_average* e il calcolo degli embeddings
relativi alle trame dei film.
Dopo è possibile inserire il titolo di un film presente nel dataset, ed ottenere i film con la trama più simile.