import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
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




## Parametros Cast Boost
parametros_cat = {"iterations": [2, 10, 20, 30, 50, 100, 200, 300, 400, 500],
                    "learning_rate": [0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1],
                    "depth": [1, 3, 5, 7, 9,20,40,50,60,70,80,90,100],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                }


grid_cat = RandomizedSearchCV(
            CatBoostClassifier(verbose=0), parametros_cat, verbose=0
        )
grid_cat.fit(x_train, y_train)


with open(diretorio + '/cat_boost_tunado.pkl', 'wb') as f:
    pickle.dump(grid_cat, f)

print('Modelo Tunado com sucesso!')
