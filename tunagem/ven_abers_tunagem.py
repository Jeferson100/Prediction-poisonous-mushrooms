import pandas as pd
import numpy as np
import sys
import os 
sys.path.append('../')
from tratamento import Tratamento
import pickle
import os 
from venn_abers import VennAbersCalibrator

# Carregando os dados

# Defina o caminho do arquivo
diretorio = os.path.join('/workspaces','binary_prediction_poisonous_mushrooms', 'dados', 'dados_tratados.pkl')

with open(diretorio, 'rb') as f:
    dados_tratados = pickle.load(f)

x_train = dados_tratados['x_train']
y_train = dados_tratados['y_train']

x_train = x_train.values
  
print('Dowload dos dados com sucesso!')


diretorio = 'modelos'


diretorio = os.path.join('/workspaces','binary_prediction_poisonous_mushrooms', 'modelos') # Modelos tunado

## fazendo dowload dos modelos

with open(diretorio + '/modelo_random_forest.pkl', 'rb') as f:
    modelo = pickle.load(f)
        

va_grad = VennAbersCalibrator(estimator=modelo.best_estimator_, inductive=False, n_splits=10)
va_grad.fit(x_train, y_train)
with open(diretorio + f'/gradiente_boost_base_ven.pkl', 'wb') as f:
      pickle.dump(va_grad, f)
      
      