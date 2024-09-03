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

# Constrói o caminho absoluto da raiz do projeto
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Constrói o caminho absoluto para o arquivo 'dados_tratados.pkl' dentro da pasta 'dados'
diretorio = os.path.join(root_dir, 'dados', 'dados_tratados.pkl')

try:
    with open(diretorio, 'rb') as f:
        dados_tratados = pickle.load(f)
except FileNotFoundError:
    print('Estamos dentro do GitHub Actions')
    with open('/home/runner/work/Prediction-poisonous-mushrooms/dados/dados_tratados.pkl', 'rb') as f:
        dados_tratados = pickle.load(f)    

x_train = dados_tratados['x_train']
y_train = dados_tratados['y_train']

x_train = x_train.values
  
print('Dowload dos dados com sucesso!')

diretorio = diretorio = os.path.join(root_dir, 'modelos')

## fazendo dowload dos modelos

try:
    with open(diretorio + '/modelo_random_forest.pkl', 'rb') as f:
        modelo = pickle.load(f)
except FileNotFoundError:
    print('Estamos dentro do GitHub Actions')
    with open('/home/runner/work/Prediction-poisonous-mushrooms/modelos/modelo_random_forest.pkl', 'rb') as f:
        dados_tratados = pickle.load(f) 
        
va_grad = VennAbersCalibrator(estimator=modelo.best_estimator_, inductive=False, n_splits=10)
va_grad.fit(x_train, y_train)

try:
    with open(diretorio + f'/gradiente_boost_base_ven.pkl', 'wb') as f:
        pickle.dump(va_grad, f)
except FileNotFoundError:
    print('Estamos dentro do GitHub Actions')
    with open('/home/runner/work/Prediction-poisonous-mushrooms/modelos/gradiente_boost_base_ven.pkl', 'wb') as f:
        pickle.dump(va_grad, f)
      
print('Modelo Calibrado com sucesso!')