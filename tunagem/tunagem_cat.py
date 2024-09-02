import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from tunagem import Tunagem
import pickle

# Carregando os dados

diretorio = 'dados'

with open(diretorio + '/dados_tratados.pkl', 'rb') as f:
    dados_tratados = pickle.load(f)

x_train = dados_tratados['x_train']
y_train = dados_tratados['y_train']

x_train = x_train.values

print('Dowload dos dados com sucesso!')

## Parametros random forest
parametros_random = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'max_depth': [1,2,60,70,80,90,100],
            'max_features': [1, 3, 4,5,6,8,10],
            'criterion':['gini','entropy'],
                 }
grid_random = RandomizedSearchCV(
            RandomForestClassifier(), parametros_random,verbose=0, scoring="recall"
        )
grid_random.fit(x_train, y_train)


with open(diretorio + '/random_forest_tunado.pkl', 'wb') as f:
    pickle.dump(grid_random, f)

print('Modelo Tunado com sucesso!')
