# Predição de Cogumelos Venenosos
Este notebook foi desenvolvido como parte do [Desafio Kaggle para Predição de Cogumelos Venenosos](https://www.kaggle.com/competitions/playground-series-s4e8/overview). O objetivo deste desafio é classificar cogumelos como comestíveis ou venenosos com base em suas características físicas.

Para avaliar a eficácia dos modelos de classificação, será utilizada a métrica coeficiente de correlação de Matthews (MCC). Esta métrica é particularmente útil para tarefas de classificação binária, especialmente em cenários onde as classes podem estar desbalanceadas, como é o caso deste desafio.

# Poisonous Mushroom Prediction
This notebook was developed as part of the Kaggle Poisonous Mushroom Prediction Challenge. The objective of this challenge is to classify mushrooms as either edible or poisonous based on their physical characteristics.

To evaluate the effectiveness of the classification models, the Matthews correlation coefficient (MCC) metric will be used. This metric is particularly useful for binary classification tasks, especially in scenarios where the classes may be imbalanced, as is the case in this challenge.

# Índice

- [1 - Importando-Bibliotecas](#1_Importando_Bibliotecas)
- [2 - Importando Dados](#2_Importando_Dados)
- [3 - Criando Classes](#3_Criando_classes)
- [4 - Análise Exploratória de Dados](#4_Analise_Exploratoria-Dados)
- [5 - Tratamento dos Dados](#5_Tratamento_Dados)
- [6 - Treinamento dos Modelos](#6_Treinamento_Dos_Modelos)
- [7 - Calibrando os modelos](#7_Calibrando_os_modelos)
- [8 - Avaliando os modelos](#8_Avaliando_os_modelos)


```python
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("Está no google colab")
except:
    IN_COLAB = False
    print("Não está no google colab")
```

    Mounted at /content/drive
    Está no google colab


# 1_Importando_Bibliotecas


```python
if IN_COLAB:
    !pip install -r /content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/requirements.txt
```


```python
import pandas as pd
#import kaggle
#from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
import pickle
from pickle import UnpicklingError
import sys

# Tratamentos
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# Treinamento
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
import keras
import keras_tuner
from keras import layers
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV

##Avaliacao
from sklearn.metrics import classification_report,recall_score,roc_curve,confusion_matrix, precision_recall_curve, log_loss, brier_score_loss, accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict

## Calibracao

from venn_abers import VennAbersCalibrator
from sklearn.base import BaseEstimator, RegressorMixin
```


```python
import sys
sys.path.append('/home/vscode/.local/lib/python3.10/site-packages')
from feature_engine.transformation import LogCpTransformer, YeoJohnsonTransformer, BoxCoxTransformer

```


```python
plt.style.use("ggplot")
%matplotlib inline
warnings.filterwarnings("ignore")
```

# 2_Importando_Dados

Para acessar os dados da api do kaggle, acesse primeiro o seguinte [repositorio](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md) para informações de como configurar suas chaves de acesso.

As credenciais serão baixadas em um arquivo com o nome `kaggle.json`. Apos isso mova o arquivo para a seguinte pasta:

```bash
mv kaggle.json /home/vscode/.config/kaggle/
```
Va na pagina do desafio e copie o link para os dados do desafio

```bash
kaggle competitions download -c playground-series-s4e8
```

ou rode esse comando no python:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# Download the competition data
api.competition_download_cli(competition='playground-series-s4e8')
```




Vou uliizar os dados chamados `train` como dados de `train`, `test` e `validade`. Como o conjunto de dados é grande, dividirei em `80%` para treino, `20%` para teste.

O arquivo csv com o nome test, será nomeado `predict` e será usado só para o envio dos resultados das predicoes do modelo.


```python
if IN_COLAB:
    sys.path.append("/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/dados")

    # Verifica os arquivos dentro da pasta 'dados'
    arquivos = os.listdir("/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/dados")

    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/dados'


else:
    # Adiciona o caminho da pasta 'dados' ao sys.path
    sys.path.append("dados")

    # Verifica os arquivos dentro da pasta 'dados'
    arquivos = os.listdir("dados")

    diretorio = 'dados'

if "train_split.csv" not in arquivos or "test_split.csv" not in arquivos:
    dados = pd.read_csv(diretorio + "/train.csv")
    train, test  = train_test_split(dados, test_size=0.2)
    train.to_csv(diretorio + "/train_split.csv", index=False)
    test.to_csv(diretorio + "/test_split.csv", index=False)
    predict = pd.read_csv(diretorio + "/test.csv")

else:
    train = pd.read_csv(diretorio + "/train_split.csv")
    test = pd.read_csv(diretorio + "/test_split.csv")
    predict = pd.read_csv(diretorio + "/test.csv")
```


```python
train.shape, test.shape
```




    ((2493556, 22), (623389, 22))




```python
train.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>class</th>
      <th>cap-diameter</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>does-bruise-or-bleed</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stem-root</th>
      <th>stem-surface</th>
      <th>stem-color</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>has-ring</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>habitat</th>
      <th>season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1274731</td>
      <td>e</td>
      <td>5.68</td>
      <td>x</td>
      <td>s</td>
      <td>o</td>
      <td>f</td>
      <td>p</td>
      <td>NaN</td>
      <td>n</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>e</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>d</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2219508</td>
      <td>e</td>
      <td>3.04</td>
      <td>s</td>
      <td>d</td>
      <td>n</td>
      <td>t</td>
      <td>d</td>
      <td>c</td>
      <td>p</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>d</td>
      <td>u</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1332256</td>
      <td>e</td>
      <td>3.32</td>
      <td>x</td>
      <td>h</td>
      <td>n</td>
      <td>f</td>
      <td>NaN</td>
      <td>c</td>
      <td>k</td>
      <td>...</td>
      <td>NaN</td>
      <td>s</td>
      <td>w</td>
      <td>NaN</td>
      <td>w</td>
      <td>f</td>
      <td>f</td>
      <td>k</td>
      <td>d</td>
      <td>u</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2468935</td>
      <td>e</td>
      <td>5.38</td>
      <td>x</td>
      <td>h</td>
      <td>n</td>
      <td>f</td>
      <td>e</td>
      <td>NaN</td>
      <td>w</td>
      <td>...</td>
      <td>NaN</td>
      <td>s</td>
      <td>n</td>
      <td>u</td>
      <td>w</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>d</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1281172</td>
      <td>p</td>
      <td>4.10</td>
      <td>x</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>a</td>
      <td>NaN</td>
      <td>w</td>
      <td>...</td>
      <td>NaN</td>
      <td>y</td>
      <td>n</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>t</td>
      <td>r</td>
      <td>NaN</td>
      <td>d</td>
      <td>u</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-62638a5e-59fb-4fd0-ada9-586e08d97218')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

    <script>
      const buttonEl =
        document.querySelector('#df-62638a5e-59fb-4fd0-ada9-586e08d97218 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-62638a5e-59fb-4fd0-ada9-586e08d97218');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-0c87be96-468e-4ccc-8729-f343f0017550">
  <button class="colab-df-quickchart" onclick="quickchart('df-0c87be96-468e-4ccc-8729-f343f0017550')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>


  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-0c87be96-468e-4ccc-8729-f343f0017550 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
test.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>class</th>
      <th>cap-diameter</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>does-bruise-or-bleed</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stem-root</th>
      <th>stem-surface</th>
      <th>stem-color</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>has-ring</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>habitat</th>
      <th>season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2464561</td>
      <td>e</td>
      <td>1.44</td>
      <td>c</td>
      <td>g</td>
      <td>n</td>
      <td>f</td>
      <td>a</td>
      <td>NaN</td>
      <td>g</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>l</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1759666</td>
      <td>e</td>
      <td>5.81</td>
      <td>s</td>
      <td>NaN</td>
      <td>n</td>
      <td>f</td>
      <td>d</td>
      <td>c</td>
      <td>w</td>
      <td>...</td>
      <td>NaN</td>
      <td>s</td>
      <td>w</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>h</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2157233</td>
      <td>e</td>
      <td>5.62</td>
      <td>f</td>
      <td>g</td>
      <td>o</td>
      <td>f</td>
      <td>s</td>
      <td>NaN</td>
      <td>y</td>
      <td>...</td>
      <td>NaN</td>
      <td>t</td>
      <td>o</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>d</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2900264</td>
      <td>e</td>
      <td>8.18</td>
      <td>x</td>
      <td>d</td>
      <td>w</td>
      <td>f</td>
      <td>s</td>
      <td>c</td>
      <td>w</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>w</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>g</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2146802</td>
      <td>p</td>
      <td>3.48</td>
      <td>f</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>e</td>
      <td>c</td>
      <td>w</td>
      <td>...</td>
      <td>NaN</td>
      <td>k</td>
      <td>w</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>t</td>
      <td>e</td>
      <td>NaN</td>
      <td>d</td>
      <td>u</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a47d2958-5774-4f6b-97cd-4c54d270e98d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

 

    <script>
      const buttonEl =
        document.querySelector('#df-a47d2958-5774-4f6b-97cd-4c54d270e98d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a47d2958-5774-4f6b-97cd-4c54d270e98d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-dbfb3bfb-1e83-4a1b-9213-dde488dc45b2">
  <button class="colab-df-quickchart" onclick="quickchart('df-dbfb3bfb-1e83-4a1b-9213-dde488dc45b2')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-dbfb3bfb-1e83-4a1b-9213-dde488dc45b2 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




# 3_Criando_classes


```python
class Tratamento:
    def retirando_ids(self, dados):
        """
        Remove a coluna `id` de um DataFrame.

        - **Parâmetros:**
        - `dados` (pd.DataFrame): DataFrame que contém a coluna `id`.

        - **Retorna:**
        - `dados` (pd.DataFrame): DataFrame sem a coluna `id`.
        - `id_dados` (pd.Series): Série contendo os IDs removidos.
        """

        id_dados = dados["id"]
        dados.drop("id", axis=1, inplace=True)
        return dados, id_dados

    def transformando_classe(self, dados):
        """
        Transforma a classe alvo em binária, onde 'e' (comestível) é 1 e todas as outras classes são 0.

        - **Parâmetros:**
        - `dados` (pd.Series): Série contendo as classes alvo.

        - **Retorna:**
        - `dados` (np.ndarray): Array com as classes transformadas.
        """
        dados = np.where(dados == 'e', 1, 0)
        return dados

    def drop_columns_nan(self,dados,columns_drop):
        """
        Remove colunas específicas de um DataFrame.

        - **Parâmetros:**
        - `dados` (pd.DataFrame): DataFrame do qual as colunas serão removidas.
        - `columns_drop` (list): Lista de colunas a serem removidas.

        - **Retorna:**
        - `dados` (pd.DataFrame): DataFrame sem as colunas especificadas.
        """
        dados.drop(columns_drop, axis=1, inplace=True)
        return dados

    def imputar_dados_faltantes_numericos(
        self, dados, modelo_imputer=None, strategy="median"
    ):
        """
        Imputa valores faltantes em um DataFrame usando uma estratégia especificada.

        Parâmetros:
        - train_x (pd.DataFrame): DataFrame com valores faltantes a serem imputados.
        - modelo_imputer (SimpleImputer, opcional): Modelo de imputação a ser utilizado. Se não for fornecido, um novo modelo será criado.
        - strategy (str, opcional): Estratégia de imputação a ser utilizada. Padrão é 'median'.

        Retorna:
        - train_x (pd.DataFrame): DataFrame com valores faltantes imputados.
        - modelo_imputer (SimpleImputer, opcional): Modelo de imputação utilizado.
        """
        float_cols = dados.select_dtypes(include=["float64", "int64"]).columns

        if modelo_imputer is None:
            imputer = SimpleImputer(strategy=strategy)
            dados[float_cols] = imputer.fit_transform(dados[float_cols])
            return dados, imputer
        else:
            dados[float_cols] = modelo_imputer.transform(dados[float_cols])
            return dados

    def criando_onhot_nan(self, dados, colunas):
        """
        Cria colunas indicadoras (`onhot`) para indicar a presença de valores faltantes em colunas específicas.

        - **Parâmetros:**
        - `dados` (pd.DataFrame): DataFrame com valores faltantes.
        - `colunas` (list): Lista de colunas nas quais criar as colunas indicadoras.

        - **Retorna:**
        - `dados` (pd.DataFrame): DataFrame com as novas colunas indicadoras.
        """
        for col in colunas:
            dados[col+'_nan'] = np.where(pd.isna(dados[col]), 1, 0)
        return dados

    ## imputando valores faltantes nas colunas categoricas
    def imputar_dados_faltantes_categoricos(self,dados, modelo_imputer=None, strategy="most_frequent"):
        """
        Imputa valores faltantes em colunas categóricas usando uma estratégia especificada.

        - **Parâmetros:**
        - `dados` (pd.DataFrame): DataFrame com valores faltantes a serem imputados.
        - `modelo_imputer` (SimpleImputer, opcional): Modelo de imputação a ser utilizado. Se não for fornecido, um novo modelo será criado.
        - `strategy` (str, opcional): Estratégia de imputação (padrão: 'most_frequent').

        - **Retorna:**
        - `dados` (pd.DataFrame): DataFrame com valores faltantes imputados.
        - `modelo_imputer` (SimpleImputer, opcional): Modelo de imputação utilizado.
        """

        colunas_categoricas = dados.select_dtypes(include=["object", "category"]).columns

        if modelo_imputer is None:
            imputer = SimpleImputer(strategy=strategy)
            dados[colunas_categoricas] = imputer.fit_transform(dados[colunas_categoricas])
            return dados, imputer
        else:
            dados[colunas_categoricas] = modelo_imputer.transform(dados[colunas_categoricas])
            return dados

    def tipos_variaveis_numericas(self, dados, tipo=None):
        """
        Aplica transformação numérica aos dados.

        Parâmetros:
        - dados (array-like): Dados a serem transformados.
        - tipo (str, opicional): Tipo de transformação a ser aplicada. Pode ser 'StandardScaler', 'MinMaxScaler', 'BoxCoxTransformer', 'LogCpTransformer' ou 'YeoJohnsonTransformer'. Se não for fornecido, será utilizado 'StandardScaler' como padrão.

        Retorno:
        - dados_transformados (array-like): Dados transformados.
        - transformador (objeto): Instância do transformador utilizado.
        """

        # Define os transformadores disponíveis
        transformadores = {
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'BoxCoxTransformer': BoxCoxTransformer,
            'LogCpTransformer': LogCpTransformer,
            'YeoJohnsonTransformer': YeoJohnsonTransformer
        }

        # Verifica se o tipo de transformação é válido
        if tipo is None:
            tipo = 'StandardScaler'
        elif tipo not in transformadores:
            raise ValueError("Tipo de transformação inválido")

        # Instancia o transformador
        transformador = transformadores[tipo]()

        # Aplica a transformação
        dados_transformados = transformador.fit_transform(dados)

        return dados_transformados, transformador

    def transformacao_variaveis_numericas(self, dados, col_transformada, tipo=None, modelo_imputer=None):
        """
        Aplica transformação numérica a colunas específicas de DataFrames .

        - **Parâmetros:**
        - `dados` (pd.DataFrame): Conjunto de dados.
        - `col_numeric` (list): Lista de colunas numéricas a serem transformadas.
        - `tipo` (str, opcional): Tipo de transformação a ser aplicada (padrão: 'StandardScaler').
        - `modelo_imputer` (dict, opcional): Dicionário de modelos de imputação para o conjunto de teste.

        - **Retorna:**
        - `train_x` ou `test_x` (pd.DataFrame): DataFrame transformado.
        - `dic_transform` (dict): Dicionário de transformadores utilizados.
        """

        if modelo_imputer is None:
            dic_transform = {}
            for col in col_transformada:

                dados[col], transformador_num = self.tipos_variaveis_numericas(dados[col].values.reshape(-1, 1), tipo=tipo)

                dic_transform[col] = transformador_num

            return dados, dic_transform
        else:
            for col in col_transformada:
                dados[col] = modelo_imputer[col].transform(dados[col].values.reshape(-1, 1))
                return dados

    def categorizar_outros(self, dados, col_cat_categorical):
        """
        - **Parâmetros:**
        - `dados` (pd.DataFrame): DataFrame com as colunas categóricas a serem reclassificadas.
        - `col_cat_categorical` (dict): Dicionário onde as chaves são os nomes das colunas e os valores são as listas de categorias a serem mantidas.

        - **Retorna:**
        - `dados` (pd.DataFrame): DataFrame com as categorias reclassificadas.
        """
        for col, categorias in col_cat_categorical.items():
            # Certifique-se de que 'categorias' seja uma lista
            if isinstance(categorias, str):
                categorias = [categorias]
            dados[col] = np.where(dados[col].isin(categorias), dados[col], 'outros_' + col)
        return dados

    def custom_combiner(self, feature, category):
        """
        Combina o nome da feature e da categoria em uma string.

        - **Parâmetros:**
        - `feature` (str): Nome da feature.
        - `category` (str): Nome da categoria.

        - **Retorna:**
        - `str`: String combinada da feature e categoria.
        """
        return str(feature) + "_"  + str(category)

    def criando_onehot(self, dados, col_categorical, model_onehot=None):
        """
        Aplica `OneHotEncoder` às colunas categóricas especificadas.

        - **Parâmetros:**
        - `dados` (pd.DataFrame): DataFrame com as colunas categóricas.
        - `col_categorical` (list): Lista de colunas categóricas a serem transformadas.
        - `model_onehot` (dict, opcional): Dicionário de modelos `OneHotEncoder` a serem utilizados.

        - **Retorna:**
        - `dados` (pd.DataFrame): DataFrame transformado com as colunas `OneHot` adicionadas.
        - `modelos_onehots` (dict, opcional): Dicionário de modelos `OneHotEncoder` utilizados.
        """

        modelos_onehots = {}

        if model_onehot is None:
            for col in col_categorical:
                model_onehot = OneHotEncoder(feature_name_combiner=self.custom_combiner,
                                            handle_unknown='infrequent_if_exist',
                                            min_frequency=0.05)
                model_onehot.fit(dados[[col]])
                dados_onehot = model_onehot.transform(dados[[col]])
                dados_onehot_df = pd.DataFrame(dados_onehot.toarray(),
                                            columns=model_onehot.get_feature_names_out(),
                                            index=dados.index)
                dados = dados.join(dados_onehot_df)
                modelos_onehots[col] = model_onehot
                dados.drop(col, axis=1, inplace=True)

            return dados, modelos_onehots

        else:
            for col in col_categorical:
                dados_onehot = model_onehot[col].transform(dados[[col]])
                dados_onehot_df = pd.DataFrame(dados_onehot.toarray(),
                                            columns=model_onehot[col].get_feature_names_out(),
                                            index=dados.index)
                dados = dados.join(dados_onehot_df)
                dados.drop(col, axis=1, inplace=True)

            return dados

```


```python
class AvaliacaoModelos:
    def calibracao_plot(self, y_test, predictions, diretorio=None, colors=None,salvar=False):
        # Calibration Plot
        if colors is None:
            colors = ['blue', 'orange', 'green', 'red','purple','black','cyan','magenta']
        plt.figure(figsize=(12, 9))
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", color='gray')
        for color, (name, preds) in zip(colors, predictions.items()):
            frac, mean = calibration_curve(y_test, preds, n_bins=10)
            plt.plot(mean, frac, "s-", label=name.replace("_", " ").title(), color=color)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Plot")
        plt.legend()
        plt.grid(True)
        if salvar:
            plt.savefig(diretorio, dpi=300, bbox_inches='tight')
        plt.show()

    def matriz_confusao_matplotlib(self, predicoes, y_teste, diretorio=None, salvar=False, threshold=0.5):
        for kay, y_proba in predicoes.items():
            y_pred = (y_proba > threshold).astype(int).squeeze()
            plt.figure(figsize=(10,6))
            sns.heatmap(confusion_matrix(y_teste, y_pred),annot=True,fmt='g')
            plt.xlabel('Predicao',fontsize=18)
            plt.ylabel('Classes Reais',fontsize=18);
            plt.title(f'Matriz de confusao {kay.replace("_", " ").title()}',fontsize=18)
            if salvar:
                plt.savefig(f'imagens/{diretorio}_{kay}.png')
            plt.show()
            print('-----------------------------------------------------------------')

    def curva_roc(self, modelos,x_treino,y_treino):
        dict_recal = {}
        for kay, model in modelos.items():
            if 'redes_neural' in kay:
                y_proba = model.predict(x_treino)
                y_scores = (y_proba > 0.50).astype(int).squeeze()
                fpr, tpr, thresholds = roc_curve(y_treino,y_scores)
            else:
                y_scores = cross_val_predict(model, x_treino, y_treino, cv=3)
                fpr, tpr, thresholds = roc_curve(y_treino,y_scores)
            dict_recal[kay+'_fpr'], dict_recal[kay+'_tpr'], dict_recal[kay+'_thresholds'] = fpr, tpr, thresholds
        return dict_recal

    def compute_ece(self, y_true, y_prob, n_bins=10, strategy='uniform'):
        true_frequencies, predicted_probabilities = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
        bin_edges = np.linspace(0, 1, n_bins+1)
        bin_width = 1.0 / n_bins
        bin_centers = np.linspace(bin_width/2, 1.0 - bin_width/2, n_bins)
        weights, _ = np.histogram(y_prob, bins=bin_edges, range=(0, 1))
        ece = np.sum(weights * np.abs(predicted_probabilities - bin_centers)) / len(y_prob)
        return ece

    def calibrations_metrics(self, modelos, x_teste, y_teste):
        predictions = {}
        for name, model in modelos.items():
            if '_ven' in name:
                predictions[name] = model.predict_proba(x_teste)[:, 1]
            elif 'redes_neural' in name:
                predictions[name] = model.predict(x_teste).squeeze()
            else:
                predictions[name] = model.predict_proba(x_teste)[:, 1]

        results = {
            "Model": [],
            "Log Loss": [],
            "Brier Loss": [],
            "ECE": [],
            "Accuracy": [],
            "Recall": [],
            "Precision": [],
            "F1 Score": [],
            "MCC": []
        }

        for name, preds in predictions.items():
            results["Model"].append(name)
            results["Log Loss"].append(log_loss(y_teste, preds))
            results["Brier Loss"].append(brier_score_loss(y_teste, preds))
            try:
                results["ECE"].append(self.compute_ece(y_teste, preds))
            except ValueError:
                results["ECE"].append(np.nan)

            # Calculate other metrics assuming a threshold of 0.5 for binary classification
            preds_binary = (preds >= 0.5).astype(int)
            results["Accuracy"].append(accuracy_score(y_teste, preds_binary))
            results["Recall"].append(recall_score(y_teste, preds_binary))
            results["Precision"].append(precision_score(y_teste, preds_binary))
            results["F1 Score"].append(f1_score(y_teste, preds_binary))
            results["MCC"].append(matthews_corrcoef(y_teste, preds_binary))

        results_df = pd.DataFrame(results)
        return results_df, predictions
```


```python
class KerasTrainedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        # Este método não será chamado, pois o modelo já está treinado
        pass

    def predict(self, X):
        return self.model.predict(X).squeeze()

    #transformar para o formato do predict_proba do sklearn
    def predict_proba(self,X):
        probs = self.model.predict(X)
        # Transforme as previsões para o formato o
        probs_transformed = np.hstack((1 - probs, probs))
        return probs_transformed
```

# 4_Analise_Exploratoria_Dados

Retirando o `id` dados de test e train


```python
id_train = train["id"]
id_test = test["id"]
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)
print(train.head())
test.head()
```

**Criando os dados train_p e train_e, a partir da classe 'p' e 'e' para verificar a diferenca entre elas**


```python
train_p = train[train["class"] == "p"]
train_e = train[train["class"] == "e"]
```


```python
train.shape
```




    (2493556, 21)




```python
test.shape
```




    (623389, 21)




```python
print(train.columns)
test.columns
```

    Index(['class', 'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
           'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
           'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
           'veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color',
           'habitat', 'season'],
          dtype='object')





    Index(['class', 'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
           'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
           'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
           'veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color',
           'habitat', 'season'],
          dtype='object')




```python
test.head()
```





  <div id="df-ad689d47-8c9f-419a-80e3-98b5d15009b0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-diameter</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>does-bruise-or-bleed</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-color</th>
      <th>stem-height</th>
      <th>...</th>
      <th>stem-root</th>
      <th>stem-surface</th>
      <th>stem-color</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>has-ring</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>habitat</th>
      <th>season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>e</td>
      <td>1.44</td>
      <td>c</td>
      <td>g</td>
      <td>n</td>
      <td>f</td>
      <td>a</td>
      <td>NaN</td>
      <td>g</td>
      <td>4.77</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>l</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>5.81</td>
      <td>s</td>
      <td>NaN</td>
      <td>n</td>
      <td>f</td>
      <td>d</td>
      <td>c</td>
      <td>w</td>
      <td>4.99</td>
      <td>...</td>
      <td>NaN</td>
      <td>s</td>
      <td>w</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>h</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>5.62</td>
      <td>f</td>
      <td>g</td>
      <td>o</td>
      <td>f</td>
      <td>s</td>
      <td>NaN</td>
      <td>y</td>
      <td>8.76</td>
      <td>...</td>
      <td>NaN</td>
      <td>t</td>
      <td>o</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>d</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>e</td>
      <td>8.18</td>
      <td>x</td>
      <td>d</td>
      <td>w</td>
      <td>f</td>
      <td>s</td>
      <td>c</td>
      <td>w</td>
      <td>6.45</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>w</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>f</td>
      <td>NaN</td>
      <td>g</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>p</td>
      <td>3.48</td>
      <td>f</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>e</td>
      <td>c</td>
      <td>w</td>
      <td>5.24</td>
      <td>...</td>
      <td>NaN</td>
      <td>k</td>
      <td>w</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>t</td>
      <td>e</td>
      <td>NaN</td>
      <td>d</td>
      <td>u</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ad689d47-8c9f-419a-80e3-98b5d15009b0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ad689d47-8c9f-419a-80e3-98b5d15009b0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ad689d47-8c9f-419a-80e3-98b5d15009b0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-142f7a8d-fc56-4e24-99c3-5d509462e351">
  <button class="colab-df-quickchart" onclick="quickchart('df-142f7a8d-fc56-4e24-99c3-5d509462e351')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-142f7a8d-fc56-4e24-99c3-5d509462e351 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2493556 entries, 0 to 2493555
    Data columns (total 21 columns):
     #   Column                Dtype  
    ---  ------                -----  
     0   class                 object 
     1   cap-diameter          float64
     2   cap-shape             object 
     3   cap-surface           object 
     4   cap-color             object 
     5   does-bruise-or-bleed  object 
     6   gill-attachment       object 
     7   gill-spacing          object 
     8   gill-color            object 
     9   stem-height           float64
     10  stem-width            float64
     11  stem-root             object 
     12  stem-surface          object 
     13  stem-color            object 
     14  veil-type             object 
     15  veil-color            object 
     16  has-ring              object 
     17  ring-type             object 
     18  spore-print-color     object 
     19  habitat               object 
     20  season                object 
    dtypes: float64(3), object(18)
    memory usage: 399.5+ MB



```python
train.describe()
```





  <div id="df-2068b4f0-1b69-40de-9372-1fbe072bbb97" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cap-diameter</th>
      <th>stem-height</th>
      <th>stem-width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.493553e+06</td>
      <td>2.493556e+06</td>
      <td>2.493556e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.307293e+00</td>
      <td>6.348078e+00</td>
      <td>1.115112e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.656455e+00</td>
      <td>2.699522e+00</td>
      <td>8.098856e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000e-02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.320000e+00</td>
      <td>4.670000e+00</td>
      <td>4.970000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.750000e+00</td>
      <td>5.880000e+00</td>
      <td>9.640000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.240000e+00</td>
      <td>7.410000e+00</td>
      <td>1.562000e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.446000e+01</td>
      <td>8.872000e+01</td>
      <td>1.024800e+02</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2068b4f0-1b69-40de-9372-1fbe072bbb97')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2068b4f0-1b69-40de-9372-1fbe072bbb97 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2068b4f0-1b69-40de-9372-1fbe072bbb97');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-ff38ba45-8890-4d1d-ba53-cfdb32d5b3e4">
  <button class="colab-df-quickchart" onclick="quickchart('df-ff38ba45-8890-4d1d-ba53-cfdb32d5b3e4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-ff38ba45-8890-4d1d-ba53-cfdb32d5b3e4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




**Primeiro verificamos a quantida de valores faltantes por coluna.**

Temos colunas com muitos valores faltantes:

As colunas `veil-type`, `spore-print-color`, `stem-root`, `veil-color`, `stem-surface` contém mais de `50%` de dados faltantes. Já a coluna a `gill-spacing` contém  mais de `40%`.


```python
# Calcula a porcentagem de valores nulos
valor_nul = (train.isnull().sum() / train.shape[0]).sort_values(ascending=False)
valor_nul_e = (train_e.isnull().sum() / train_e.shape[0]).sort_values(ascending=False)
valor_nul_p = (train_p.isnull().sum() / train_p.shape[0]).sort_values(ascending=False)

# Cria os subplots
fig, axes = plt.subplots(3, 1, figsize=(30, 20))

# Subplot para 'train'
axes[0].bar(valor_nul.index, valor_nul.values)
axes[0].set_title("Porcentagem de Valores Nulos (train)")
axes[0].tick_params(axis="x", rotation=90)
for index, value in enumerate(valor_nul.values):
    axes[0].text(index, value, f"{value:.2f}", ha="center", va="bottom")

# Subplot para 'train_e'
axes[1].bar(valor_nul_e.index, valor_nul_e.values)
axes[1].set_title("Porcentagem de Valores Nulos (train_e)")
axes[1].tick_params(axis="x", rotation=90)
for index, value in enumerate(valor_nul_e.values):
    axes[1].text(index, value, f"{value:.2f}", ha="center", va="bottom")

# Subplot para 'train_p'
axes[2].bar(valor_nul_p.index, valor_nul_p.values)
axes[2].set_title("Porcentagem de Valores Nulos (train_p)")
axes[2].tick_params(axis="x", rotation=90)
for index, value in enumerate(valor_nul_p.values):
    axes[2].text(index, value, f"{value:.2f}", ha="center", va="bottom")

# Ajusta o layout para evitar sobreposição
plt.tight_layout()

# Exibe os gráficos
plt.show()
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_30_0.png)
    


**Selecionando as colunas com mais valores faltantes**

Selecionando as colunas com mais de `40%` de valores faltantes. Olhando as linhas, verificamos que as colunas juntas tem 5 valores faltantes em `46%` das vezes, `27%` com 4 valores nulos e todas as 6 linhas nulas `17%`.

Isso demonstra que `90%` das vezes, essas colunas na mesma linha tem mais de 4 valores nulos. Por terem muitos valores nulos e com pouca informação, essas colunas serão excluidas:

- veil-type
- spore-print-color
- stem-root
- veil-color
- stem-surface
- gill-spacing'


```python
# A maioria das colunas que tem mais de 40% de valores nulos, essa ocorrencia ocorre juntamente na mesma linha
valores_nulos_por_linha = (
    train[valor_nul[valor_nul > 0.4].index.values].isnull().sum(axis=1).value_counts()
)
valores_nulos_por_linha_e = (
    train_e[valor_nul_e[valor_nul_e > 0.4].index.values]
    .isnull()
    .sum(axis=1)
    .value_counts()
)
valores_nulos_por_linha_p = (
    train_p[valor_nul_p[valor_nul_p > 0.4].index.values]
    .isnull()
    .sum(axis=1)
    .value_counts()
)

## transfoemando em porcentagem
valores_nulos_por_linha_porcent = round(valores_nulos_por_linha / train.shape[0], 2)
valores_nulos_por_linha_porcent_e = round(
    valores_nulos_por_linha_e / train_e.shape[0], 2
)
valores_nulos_por_linha_porcent_p = round(
    valores_nulos_por_linha_p / train_p.shape[0], 2
)

## criando o grafico
fig, axes = plt.subplots(3, 1, figsize=(20, 12))
axes[0].bar(
    valores_nulos_por_linha_porcent.index, valores_nulos_por_linha_porcent.values
)
axes[0].set_title(
    "Valores Nulos por Linhas nas Colunas com mais de 40% de Valores Nulos"
)
axes[0].tick_params(axis="x", rotation=90)


axes[1].bar(
    valores_nulos_por_linha_porcent_e.index, valores_nulos_por_linha_porcent_e.values
)
axes[1].set_title(
    "Valores Nulos por Linhas nas Colunas com mais de 40% de Valores Nulos na classe e"
)
axes[1].tick_params(axis="x", rotation=90)


axes[2].bar(
    valores_nulos_por_linha_porcent_p.index, valores_nulos_por_linha_porcent_p.values
)
axes[2].set_title("Porcentagem de Valores Nulos (train_p)")
axes[2].tick_params(axis="x", rotation=90)

plt.tight_layout()

plt.show()
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_32_0.png)
    



```python
# Remove as colunas 'veil-type', 'spore-print-color', 'stem-root', 'veil-color', 'stem-surface', 'gill-spacing'
columns_drop = [
    "veil-type",
    "spore-print-color",
    "stem-root",
    "veil-color",
    "stem-surface",
    "gill-spacing",
]

train.drop(columns_drop, axis=1, inplace=True)
```

**Verificando quantos valores unicos tem em cada coluna**

As 3 primeiras colunas são numericas não importando a quantidade de valores unicos. Ja algumas colunas categoricas tem muitas classes, `cap-surface` por exemplo tem 70 valores unicos.


```python
train_nunique = train.nunique().sort_values(ascending=False)
train_nunique_e = train_e.nunique().sort_values(ascending=False)
train_nunique_p = train_p.nunique().sort_values(ascending=False)

fig, axes = plt.subplots(3, 1, figsize=(20, 20))

axes[0].bar(train_nunique.index, train_nunique.values)
axes[0].set_title("Contagem de Valores Unicos por Colunas")
axes[0].tick_params(axis="x", rotation=90)
for index, value in enumerate(train_nunique.values):
    axes[0].text(index, value, f"{value:.2f}", ha="center", va="bottom")

axes[1].bar(train_nunique_e.index, train_nunique_e.values)
axes[1].set_title("Contagem de Valores Unicos por Colunas na classe e")
axes[1].tick_params(axis="x", rotation=90)
for index, value in enumerate(train_nunique_e.values):
    axes[1].text(index, value, f"{value:.2f}", ha="center", va="bottom")


axes[2].bar(train_nunique_p.index, train_nunique_p.values)
axes[2].set_title("Contagem de Valores Unicos por Colunas na Classe p")
axes[2].tick_params(axis="x", rotation=90)
for index, value in enumerate(train_nunique_p.values):
    axes[2].text(index, value, f"{value:.2f}", ha="center", va="bottom")
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_35_0.png)
    


**Verificando valores numericos**

Primeiro plotamos o histograma e depois um boxplot.

Com o histograma verificamos que as distribuições das colunas numericas não são normalmente distribuidas. Elas tem uma distribuição assimétrica para a esquerda.

Analisando o boxplot, observamos que as caixas são bastante largas, indicando que os valores das colunas numéricas estão concentrados entre o 1º e o 3º quartil. Além disso, percebemos que a média e a mediana não diferem muito nas colunas `cap-diameter` e `stem-height`. No entanto, na coluna `stem-width`, há uma diferença maior entre a média e a mediana, com uma discrepância de 13%. Essa diferença é ainda mais acentuada na classe `p`, onde chega a 26,95%. Isso pode indicar que a classe `p` tem uma maior quantidade de outliers.

```
cap-diameter: Diferença entre media e mediana dados train: 8.84 %, Diferença entre media e mediana na classe e: 8.70 %, Diferença entre media e mediana na classe p: 16.86 %
stem-height: Diferença entre media e mediana dados train: 7.37 %, Diferença entre media e mediana na classe e: 6.70 %, Diferença entre media e mediana na classe p: 9.57 %
stem-width: Diferença entre media e mediana dados train: 13.55 %, Diferença entre media e mediana na classe e: 5.02 %, Diferença entre media e mediana na classe p: 26.95 %
```

Analisando a porcentagem de outliers por coluna, observamos que a coluna com mais outliers é `stem-height`, com 4,63%. Esses valores extremos podem ser um preditor para a classe `p`, indicando que, quanto mais extremos os valores, mais venenosos podem ser.

```
Explorando a porcentagem de outliers na coluna cap-diameter
cap-diameter: 2.44%
cap-diameter_e: 1.91%
cap-diameter_p: 3.09%
-------------------------------------------------------------
Explorando a porcentagem de outliers na coluna stem-height
stem-height: 4.25%
stem-height_e: 4.74%
stem-height_p: 3.58%
-------------------------------------------------------------
Explorando a porcentagem de outliers na coluna stem-width
stem-width: 2.16%
stem-width_e: 1.06%
stem-width_p: 4.63%
```


```python
## plot histograma dos valores numericos
for col in train.columns:
    plt.figure(figsize=(10, 10))
    if train[col].dtype == "float64":
        train[col].hist(bins=20)
        train_e[col].hist(bins=20)
        train_p[col].hist(bins=20)
        plt.title(col)
        plt.legend(["train", "train_e", "train_p"])
        plt.show()
        plt.close()
```


    <Figure size 1000x1000 with 0 Axes>



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_37_1.png)
    



    <Figure size 1000x1000 with 0 Axes>



    <Figure size 1000x1000 with 0 Axes>



    <Figure size 1000x1000 with 0 Axes>



    <Figure size 1000x1000 with 0 Axes>



    <Figure size 1000x1000 with 0 Axes>



    <Figure size 1000x1000 with 0 Axes>



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_37_8.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_37_9.png)
    



    <Figure size 1000x1000 with 0 Axes>



    <Figure size 1000x1000 with 0 Axes>



    <Figure size 1000x1000 with 0 Axes>



    <Figure size 1000x1000 with 0 Axes>



    <Figure size 1000x1000 with 0 Axes>



```python
for col in train.columns:
    if train[col].dtype == "float64":
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        mean_value = train[col].mean()
        mean_value_e = train_e[col].mean()
        mean_value_p = train_p[col].mean()
        median_value = train[col].median()
        median_value_e = train_e[col].median()
        median_value_p = train_p[col].median()

        sns.boxplot(train[col], ax=axes[0], showfliers=True)
        axes[0].set_title(f"{col} - Train")
        axes[0].axhline(
            y=mean_value, color="red", linestyle="--", label=f"Média ({mean_value:.2f})"
        )
        axes[0].axhline(
            y=median_value,
            color="blue",
            linestyle="-",
            label=f"Mediana ({median_value:.2f})",
        )
        axes[0].legend()

        sns.boxplot(train_e[col], ax=axes[1], showfliers=True)
        axes[1].set_title(f"{col} - Train_e")
        axes[1].axhline(
            y=mean_value_e,
            color="red",
            linestyle="--",
            label=f"Média ({mean_value_e:.2f})",
        )
        axes[1].axhline(
            y=median_value_e,
            color="blue",
            linestyle="-",
            label=f"Mediana ({median_value_e:.2f})",
        )
        axes[1].legend()

        sns.boxplot(train_p[col], ax=axes[2], showfliers=True)
        axes[2].set_title(f"{col} - Train_p")
        axes[2].axhline(
            y=mean_value_p,
            color="red",
            linestyle="--",
            label=f"Média ({mean_value_p:.2f})",
        )
        axes[2].axhline(
            y=median_value_p,
            color="blue",
            linestyle="-",
            label=f"Mediana ({median_value_p:.2f})",
        )
        axes[2].legend()

        plt.show()
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_38_0.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_38_1.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_38_2.png)
    


**Visualizando 5 transformacoes para as colunas numericas**

- StandardScaler()
- MinMaxScaler()
- BoxCoxTransformer()
- LogCpTransformer()
- YeoJohnsonTransformer()


```python
train_numerical_sem_nan = train.copy()
float_cols = train_numerical_sem_nan.select_dtypes(include=["float64", "int64"]).columns
imputer = SimpleImputer(strategy="median")
train_numerical_sem_nan[float_cols] = imputer.fit_transform(train_numerical_sem_nan[float_cols])
col_numeric = ['cap-diameter', 'stem-height', 'stem-width']
for col in col_numeric:
    stand = StandardScaler()
    norm = MinMaxScaler()
    box = BoxCoxTransformer()
    log = LogCpTransformer()
    jonson = YeoJohnsonTransformer()

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    axes[0][0].hist(train_numerical_sem_nan[col], label=col)
    axes[0][0].set_title(f'Distribuições coluna {col}')
    axes[0][0].legend()

    axes[0][1].hist(stand.fit_transform(train_numerical_sem_nan[[col]]).squeeze(), label=f'{col}_stand')
    axes[0][1].set_title(f'{col} transformada por StandardScaler')

    axes[0][2].hist(norm.fit_transform(train_numerical_sem_nan[[col]]).squeeze(), label=f'{col}_norm')
    axes[0][2].set_title(f'{col} transformada por MinMaxScaler')

    axes[1][0].hist(log.fit_transform(train_numerical_sem_nan[[col]]), label=f'{col}_log')
    axes[1][0].set_title(f'{col} transformada por LogCpTransformer')

    try:
        axes[1][1].hist(box.fit_transform(train_numerical_sem_nan[[col]]).squeeze(), label=f'{col}_box')
        axes[1][1].set_title(f'{col} transformada por BoxCoxTransformer')

    except ValueError:
        axes[1][1].set_title('Valores negativos BoxCoxTransformer não transforma.')

    axes[1][2].hist(jonson.fit_transform(train_numerical_sem_nan[[col]]).squeeze(), label=f'{col}_jonson')
    axes[1][2].set_title(f'{col} transformada por YeoJohnsonTransformer')

    plt.tight_layout()
    plt.show()
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_40_0.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_40_1.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_40_2.png)
    



```python
# difernca percentual entre a media e a mediana
for col in train.columns:
    if train[col].dtype == "float64":
        mean_value = train[col].mean()
        mean_value_e = train_e[col].mean()
        mean_value_p = train_p[col].mean()

        median_value = train[col].median()
        median_value_e = train_e[col].median()
        median_value_p = train_p[col].median()

        diff_mean_median = (mean_value - median_value) / mean_value * 100
        diff_mean_median_e = (mean_value_e - median_value_e) / mean_value_e * 100
        diff_mean_median_p = (mean_value_p - median_value_p) / mean_value_p * 100

        print(
            f"{col}: Diferença entre media e mediana dados train: {diff_mean_median:.2f} %, Diferença entre media e mediana na classe e: {diff_mean_median_e:.2f} %, Diferença entre media e mediana na classe p: {diff_mean_median_p:.2f} %"
        )
```

    cap-diameter: Diferença entre media e mediana dados train: 8.84 %, Diferença entre media e mediana na classe e: 8.70 %, Diferença entre media e mediana na classe p: 16.86 %
    stem-height: Diferença entre media e mediana dados train: 7.37 %, Diferença entre media e mediana na classe e: 6.70 %, Diferença entre media e mediana na classe p: 9.57 %
    stem-width: Diferença entre media e mediana dados train: 13.55 %, Diferença entre media e mediana na classe e: 5.02 %, Diferença entre media e mediana na classe p: 26.95 %



```python
# Porcentagem de outliers
for col in train.columns:
    if train[col].dtype == "float64":
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        upper_bound, lower_bound
        out_train = train[
            (train[col] < lower_bound) | (train[col] > upper_bound)
        ].shape[0]

        Q1_e = train_e[col].quantile(0.25)
        Q3_e = train_e[col].quantile(0.75)
        IQR_e = Q3_e - Q1_e
        lower_bound_e = Q1_e - 1.5 * IQR_e
        upper_bound_e = Q3_e + 1.5 * IQR_e
        out_train_e = train_e[
            (train_e[col] < lower_bound_e) | (train_e[col] > upper_bound_e)
        ].shape[0]

        Q1_p = train_p[col].quantile(0.25)
        Q3_p = train_p[col].quantile(0.75)
        IQR_p = Q3_p - Q1_p
        lower_bound_p = Q1_p - 1.5 * IQR_p
        upper_bound_p = Q3_p + 1.5 * IQR_p
        out_train_p = train_p[
            (train_p[col] < lower_bound_p) | (train_p[col] > upper_bound_p)
        ].shape[0]

        print(f"Explorando a porcentagem de outliers na coluna {col}")
        print(f"{col}: {out_train/train.shape[0]:.2%}")
        print(f"{col}_e: {out_train_e/train_e.shape[0]:.2%}")
        print(f"{col}_p: {out_train_p/train_p.shape[0]:.2%}")

        print("-------------------------------------------------------------")
```

    Explorando a porcentagem de outliers na coluna cap-diameter
    cap-diameter: 2.44%
    cap-diameter_e: 1.91%
    cap-diameter_p: 3.09%
    -------------------------------------------------------------
    Explorando a porcentagem de outliers na coluna stem-height
    stem-height: 4.25%
    stem-height_e: 4.74%
    stem-height_p: 3.58%
    -------------------------------------------------------------
    Explorando a porcentagem de outliers na coluna stem-width
    stem-width: 2.16%
    stem-width_e: 1.06%
    stem-width_p: 4.63%
    -------------------------------------------------------------


**Verificando as categorias em cada coluna**

Ao analisar as colunas categóricas, observamos que muitas variáveis aparecem com frequência nas mesmas colunas. Por exemplo, na coluna `cap-shape`, a variável X ocorre em 46% das vezes. A coluna `does-bruise-or-bleed` possui a variável `f` como a mais frequente, aparecendo em 82% dos dados de treino.

Além disso, algumas colunas apresentam uma mudança na variável mais frequente dependendo da classe. Por exemplo, na coluna `stem-color`, a variável mais frequente na classe `e` é a variável `w`, ocorrendo em 49% das vezes, enquanto na classe `p`, a variável predominante é `n`, com `36%`.

Verificando os valores acumulados das porcentagens, notamos que em quase todas as colunas, 5 ou 6 variáveis representam `90%` dos valores. Uma das únicas colunas com uma distribuição mais uniforme entre as variáveis é `cap-surface`, onde as 10 primeiras variáveis representam `78%` dos dados.


```python
for col in train.columns:
    if train[col].dtype != "float64":
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        train_class = (
            round(train[col].value_counts() / train.shape[0], 2)
            .head(10)
            .sort_values(ascending=False)
        )
        train_class_e = (
            round(train_e[col].value_counts() / train_e.shape[0], 2)
            .head(10)
            .sort_values(ascending=False)
        )
        train_class_p = (
            round(train_p[col].value_counts() / train_p.shape[0], 2)
            .head(10)
            .sort_values(ascending=False)
        )

        axes[0].bar(train_class.index, train_class.values, color="blue", label="train")
        axes[0].set_title(f"{col} na classe train")
        axes[0].tick_params(axis="x", rotation=90)
        for index, value in enumerate(train_class.values):
            axes[0].text(index, value, f"{value:.2f}", ha="center", va="bottom")

        axes[1].bar(
            train_class_e.index, train_class_e.values, color="red", label="train_e"
        )
        axes[1].set_title(f"{col} na classe train_e")
        axes[1].tick_params(axis="x", rotation=90)
        for index, value in enumerate(train_class_e.values):
            axes[1].text(index, value, f"{value:.2f}", ha="center", va="bottom")

        axes[2].bar(
            train_class_p.index, train_class_p.values, color="green", label="train_p"
        )
        axes[2].set_title(f"{col} na classe train_p")
        axes[2].tick_params(axis="x", rotation=90)
        for index, value in enumerate(train_class_p.values):
            axes[2].text(index, value, f"{value:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.show()
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_0.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_1.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_2.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_3.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_4.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_5.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_6.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_7.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_8.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_9.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_10.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_44_11.png)
    


**Nos proximos graficos faremos a soma acumulada das porcentagens das categorias**


```python
for col in train.columns:
    if train[col].dtype != "float64":
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        train_cumsum = (
            round(train[col].value_counts() / train.shape[0], 2)
            .sort_values(ascending=False)
            .cumsum()
            .head(10)
        )
        train_cumsum_e = (
            round(train_e[col].value_counts() / train_e.shape[0], 2)
            .sort_values(ascending=False)
            .cumsum()
            .head(10)
        )
        train_cumsum_p = (
            round(train_p[col].value_counts() / train_p.shape[0], 2)
            .sort_values(ascending=False)
            .cumsum()
            .head(10)
        )

        axes[0].bar(
            train_cumsum.index, train_cumsum.values, color="blue", label="train"
        )
        axes[0].set_title(f"{col} train cumsum")
        axes[0].tick_params(axis="x", rotation=90)
        for index, value in enumerate(train_cumsum.values):
            axes[0].text(index, value, f"{value:.2f}", ha="center", va="bottom")

        axes[1].bar(
            train_cumsum_e.index, train_cumsum_e.values, color="red", label="train_e"
        )
        axes[1].set_title(f"{col} train_e cumsum")
        axes[1].tick_params(axis="x", rotation=90)
        for index, value in enumerate(train_cumsum_e.values):
            axes[1].text(index, value, f"{value:.2f}", ha="center", va="bottom")

        axes[2].bar(
            train_cumsum_p.index, train_cumsum_p.values, color="green", label="train_p"
        )
        axes[2].set_title(f"{col} train_p cumsum")
        axes[2].tick_params(axis="x", rotation=90)
        for index, value in enumerate(train_cumsum_p.values):
            axes[2].text(index, value, f"{value:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.show()
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_0.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_1.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_2.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_3.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_4.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_5.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_6.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_7.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_8.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_9.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_10.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_46_11.png)
    


**Nos proximos gráficos anasalimos a media e mediana nas colunas numericas pelas classes e pelas categorias das colunas categoricas**


```python
train.groupby(["cap-shape", "class"])[['cap-diameter', 'stem-height', 'stem-width']].agg({
    'cap-diameter': ['mean', 'median','count'],
    'stem-height': ['mean','median', 'count'],
    'stem-width': ['mean', 'median','count']
}).sort_values(by=('cap-diameter', 'count'), ascending=False).head(12)
```





  <div id="df-81bde26b-9cd0-4850-90d7-2b12e7cfb320" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">cap-diameter</th>
      <th colspan="3" halign="left">stem-height</th>
      <th colspan="3" halign="left">stem-width</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
    </tr>
    <tr>
      <th>cap-shape</th>
      <th>class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">x</th>
      <th>p</th>
      <td>5.801369</td>
      <td>5.09</td>
      <td>586048</td>
      <td>6.408587</td>
      <td>5.77</td>
      <td>586048</td>
      <td>10.310832</td>
      <td>7.91</td>
      <td>586048</td>
    </tr>
    <tr>
      <th>e</th>
      <td>6.831617</td>
      <td>6.59</td>
      <td>562770</td>
      <td>6.708666</td>
      <td>6.37</td>
      <td>562771</td>
      <td>12.515798</td>
      <td>11.91</td>
      <td>562771</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">f</th>
      <th>p</th>
      <td>6.147670</td>
      <td>4.86</td>
      <td>277200</td>
      <td>6.106435</td>
      <td>5.47</td>
      <td>277201</td>
      <td>9.844380</td>
      <td>6.71</td>
      <td>277201</td>
    </tr>
    <tr>
      <th>e</th>
      <td>7.144165</td>
      <td>6.97</td>
      <td>263715</td>
      <td>6.301611</td>
      <td>5.85</td>
      <td>263715</td>
      <td>12.689022</td>
      <td>12.30</td>
      <td>263715</td>
    </tr>
    <tr>
      <th>b</th>
      <th>p</th>
      <td>3.457981</td>
      <td>3.15</td>
      <td>197019</td>
      <td>6.954017</td>
      <td>5.92</td>
      <td>197019</td>
      <td>5.248591</td>
      <td>3.97</td>
      <td>197019</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">s</th>
      <th>p</th>
      <td>6.960287</td>
      <td>6.52</td>
      <td>160218</td>
      <td>5.075420</td>
      <td>4.84</td>
      <td>160219</td>
      <td>11.179819</td>
      <td>9.20</td>
      <td>160219</td>
    </tr>
    <tr>
      <th>e</th>
      <td>7.871823</td>
      <td>7.50</td>
      <td>131848</td>
      <td>5.785798</td>
      <td>5.86</td>
      <td>131848</td>
      <td>15.078627</td>
      <td>16.13</td>
      <td>131848</td>
    </tr>
    <tr>
      <th>o</th>
      <th>p</th>
      <td>5.761289</td>
      <td>4.42</td>
      <td>58841</td>
      <td>3.591321</td>
      <td>3.65</td>
      <td>58841</td>
      <td>17.901514</td>
      <td>16.43</td>
      <td>58841</td>
    </tr>
    <tr>
      <th>b</th>
      <th>e</th>
      <td>3.336758</td>
      <td>3.18</td>
      <td>57858</td>
      <td>5.347697</td>
      <td>5.40</td>
      <td>57858</td>
      <td>3.712206</td>
      <td>3.08</td>
      <td>57858</td>
    </tr>
    <tr>
      <th>p</th>
      <th>e</th>
      <td>7.910718</td>
      <td>8.18</td>
      <td>46277</td>
      <td>9.047026</td>
      <td>7.89</td>
      <td>46277</td>
      <td>14.484625</td>
      <td>14.81</td>
      <td>46277</td>
    </tr>
    <tr>
      <th>c</th>
      <th>p</th>
      <td>4.309586</td>
      <td>4.66</td>
      <td>45360</td>
      <td>6.126992</td>
      <td>6.41</td>
      <td>45360</td>
      <td>7.677243</td>
      <td>7.38</td>
      <td>45360</td>
    </tr>
    <tr>
      <th>p</th>
      <th>p</th>
      <td>5.771435</td>
      <td>6.55</td>
      <td>39421</td>
      <td>9.437283</td>
      <td>8.40</td>
      <td>39421</td>
      <td>12.752324</td>
      <td>13.07</td>
      <td>39421</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-81bde26b-9cd0-4850-90d7-2b12e7cfb320')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-81bde26b-9cd0-4850-90d7-2b12e7cfb320 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-81bde26b-9cd0-4850-90d7-2b12e7cfb320');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-eef521da-5a82-4d72-a7bc-cf37fb4a4995">
  <button class="colab-df-quickchart" onclick="quickchart('df-eef521da-5a82-4d72-a7bc-cf37fb4a4995')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-eef521da-5a82-4d72-a7bc-cf37fb4a4995 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
for col in train.columns:
    if train[col].dtype == 'object':
        dados = train.groupby([col, "class"])[['cap-diameter', 'stem-height', 'stem-width']].agg({
            'cap-diameter': ['mean', 'median','count'],
            'stem-height': ['mean','median', 'count'],
            'stem-width': ['mean', 'median','count']
        }).sort_values(by=('cap-diameter', 'count'), ascending=False).head(12)

        fig, axs = plt.subplots(1,3, figsize=(50, 15))

        # Gráfico de barras para cap-diameter
        sns.barplot(x=dados.index.get_level_values(0), y=dados[('cap-diameter', 'mean')], hue=dados.index.get_level_values(1), ax=axs[0],palette=['#9C27B0', '#C5CAE9'])
        sns.barplot(x=dados.index.get_level_values(0), y=dados[('cap-diameter', 'median')], hue=dados.index.get_level_values(1), ax=axs[0], palette=['#3F51B5', '#66CCCC'])
        handles = [plt.Line2D([], [], marker='o', color='#3F51B5', label='Mediana_p'),
                plt.Line2D([], [], marker='o', color='#66CCCC', label='Mediana_e'),
                plt.Line2D([], [], marker='o', color='#9C27B0', label='Media_p'),
                plt.Line2D([], [], marker='o', color='#C5CAE9', label='Media_e'),]
        axs[0].legend(title='Legenda', loc='upper right', bbox_to_anchor=(1.05, 1), handles=handles)
        axs[0].set_title(f'Comparação entre média e mediana nas colunas numéricas por classe e categorias de {col}')
        axs[0].set_ylabel('Valores')
        axs[0].set_xlabel(col)

        # Gráfico de barras para stem-height
        sns.barplot(x=dados.index.get_level_values(0), y=dados[('stem-height', 'mean')], hue=dados.index.get_level_values(1), ax=axs[1], palette=['#9C27B0', '#C5CAE9'])
        sns.barplot(x=dados.index.get_level_values(0), y=dados[('stem-height', 'median')], hue=dados.index.get_level_values(1), ax=axs[1], palette=['#3F51B5', '#66CCCC'])
        handles = [plt.Line2D([], [], marker='o', color='#3F51B5', label='Mediana_p'),
                plt.Line2D([], [], marker='o', color='#66CCCC', label='Mediana_e'),
                plt.Line2D([], [], marker='o', color='#9C27B0', label='Media_p'),
                plt.Line2D([], [], marker='o', color='#C5CAE9', label='Media_e'),]
        axs[1].legend(title='Legenda', loc='upper right', bbox_to_anchor=(1.05, 1), handles=handles)
        axs[1].set_title(f'Comparação entre média e mediana nas colunas numéricas por classe e categorias de {col}')
        axs[1].set_ylabel('Valores')
        axs[1].set_xlabel(col)

        # Gráfico de barras para stem-width
        sns.barplot(x=dados.index.get_level_values(0), y=dados[('stem-width', 'mean')], hue=dados.index.get_level_values(1), ax=axs[2], palette=['#9C27B0', '#C5CAE9'])
        sns.barplot(x=dados.index.get_level_values(0), y=dados[('stem-width', 'median')], hue=dados.index.get_level_values(1), ax=axs[2], palette=['#3F51B5', '#66CCCC'])
        handles = [plt.Line2D([], [], marker='o', color='#3F51B5', label='Mediana_p'),
                plt.Line2D([], [], marker='o', color='#66CCCC', label='Mediana_e'),
                plt.Line2D([], [], marker='o', color='#9C27B0', label='Media_p'),
                plt.Line2D([], [], marker='o', color='#C5CAE9', label='Media_e'),]
        axs[2].legend(title='Legenda', loc='upper right', bbox_to_anchor=(1.05, 1), handles=handles)
        axs[2].set_title(f'Comparação entre média e mediana nas colunas numéricas por classe e categorias de {col}')
        axs[2].set_ylabel('Valores')
        axs[0].set_xlabel(col)

        # Mostre o gráfico
        plt.tight_layout()
        plt.show()
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_0.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_1.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_2.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_3.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_4.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_5.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_6.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_7.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_8.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_9.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_10.png)
    



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_49_11.png)
    


# 5_Tratamento_Dados

**Foi aplicado os seguintes tratamentos ao conjunto de dados**

1. **Colocando as classes como numéricas**
    - Colocando as classe `e=comestivel` como `1` e a classe `p=venenosa` como `0`.    

2. **Retirando IDs**:
   - Remove a coluna `id` de `train_x` e `test_x`.
  
3. **Dropando colunas**:
   - Remove as seguintes colunas que comtém muitos valores nan: `veil-type`, `spore-print-color`, `stem-root`, `veil-color`, `stem-surface`, `gill-spacing`.
  
4. **Criando colunas de NaN (OneHot)**:
   - Algumas outras colunas tambem tem menos de 20% de nan, mantive elas mas crie colunas onhot para os dados faltantes, criando as seguintes colunas:`cap-surface_nan`, `gill-attachment_nan`, `ring-type_nan`.
  
5. **Imputação de Dados Faltantes Categoricas**:
   - ImputEI valores faltantes nas colunas Categoricas, escolhendo o metodo de imputação a `most_frequent`.

6. **Imputação de Dados Faltantes Numericos**:
   - ImputEI valores faltantes nas colunas Numericas, escolhendo o metodo de imputação a `mediam`.
  
7. **Escalonamento de Dados Numéricos**:
   - Escalona as variáveis numéricas utilizando a técnica de escalonamento `StandardScaler`.
  
8. **OneHot Encoding com Frequência Mínima**:
   - Aplica `OneHotEncoder` com uma frequência mínima de 5%, criando uma coluna `infrequent_if_exist` para categorias com menos de 5%.

Após todo os tratamentos os conjuntos de dados ficaram com os seguintes tamanhos:

- treino: `(2493556, 59)`

- teste: `(623389, 59)`



```python
if IN_COLAB:
    sys.path.append("/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/dados")
    # Verifica os arquivos dentro da pasta 'dados'
    arquivos = os.listdir("/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/dados")
    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/dados'
else:
    # Adiciona o caminho da pasta 'dados' ao sys.path
    sys.path.append("dados")
    # Verifica os arquivos dentro da pasta 'dados'
    arquivos = os.listdir("dados")
    diretorio = 'dados'

tratar_dados = False

if "dados_tratados.pkl" not in arquivos or tratar_dados == True:
    ## separando as x e y
    train_x = train.drop("class", axis=1)
    train_y = train["class"]

    test_x = test.drop("class", axis=1)
    test_y = test["class"]

    ## colocando as classe e=comestivel como 1 e a classe p=venenosa como 0.

    train_y = np.where(train_y == 'e', 1, 0)
    test_y = np.where(test_y == 'e', 1, 0)


    ## criando tratamento
    trat = Tratamento()

    ## retirando ids
    train_x_sem_id, train_x_id = trat.retirando_ids(train_x)
    test_x_sem_id, test_x_id = trat.retirando_ids(test_x)


    ## Dropando colunas 'veil-type', 'spore-print-color', 'stem-root', 'veil-color', 'stem-surface', 'gill-spacing'
    columns_drop = [
        "veil-type",
        "spore-print-color",
        "stem-root",
        "veil-color",
        "stem-surface",
        "gill-spacing",
    ]

    train_x_drop = trat.drop_columns_nan(train_x_sem_id, columns_drop)
    test_x_drop = trat.drop_columns_nan(test_x_sem_id, columns_drop)


    ## Criando uma coluna com 1 onde tem nan e 0 onde não tem nan, nas colunas 'cap-surface', 'gill-attachment', 'ring-type'
    colunas_onhot_nan = ['cap-surface', 'gill-attachment', 'ring-type']

    train_x_nan_onhot = trat.criando_onhot_nan(train_x_drop, colunas_onhot_nan)
    test_x_nan_onhot = trat.criando_onhot_nan(test_x_drop, colunas_onhot_nan)


    ## Imputando em valores faltantes nas colunas categoricas

    train_x_imput_categorica, modelo_input_cat = trat.imputar_dados_faltantes_categoricos(train_x_nan_onhot)
    test_x_imput_categorica = trat.imputar_dados_faltantes_categoricos(test_x_nan_onhot,modelo_input_cat)


    ## Imputando em valores faltantes nas colunas numericas

    train_x_imput_numerical, modelo_input_num = trat.imputar_dados_faltantes_numericos(train_x_imput_categorica)
    test_x_imput_numerical = trat.imputar_dados_faltantes_numericos(test_x_imput_categorica,modelo_input_num)


    ## escalonando os dados numericos

    col_numeric = ['cap-diameter', 'stem-height', 'stem-width']

    train_x_stard_numerical, modelo_stard = trat.transformacao_variaveis_numericas(train_x_imput_numerical,col_transformada=col_numeric)
    test_x_stard_numerical = trat.transformacao_variaveis_numericas(test_x_imput_numerical,col_transformada=col_numeric, modelo_imputer=modelo_stard)


    ## Passando OneHotEncoder com os parametros minimo de 5% de frequencia onde sera criada a coluna infrequent_if_exist se nao atingir esse limite, tentando dimunir o numero de categorias para algumas colunas

    col_categorical = ['cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed', 'gill-attachment', 'gill-color', 'stem-color', 'has-ring', 'ring-type', 'habitat', 'season']

    x_train_onehot, model_onehot = trat.criando_onehot(train_x_stard_numerical, col_categorical)

    x_test_onhot = trat.criando_onehot(test_x_stard_numerical, col_categorical, model_onehot=model_onehot)



    print(x_train_onehot.shape, x_test_onhot.shape)
    print('Dados tratados!')

    dados_tratados = {}
    dados_tratados['x_train'] = x_train_onehot
    dados_tratados['x_test'] = x_test_onhot
    dados_tratados['y_train'] = train_y
    dados_tratados['y_test'] = test_y
    dados_tratados['modelo_input_cat'] = modelo_input_cat
    dados_tratados['modelo_input_num'] = modelo_input_num
    dados_tratados['modelo_stard'] = modelo_stard
    dados_tratados['model_onehot'] = model_onehot
    dados_tratados['columns_drop'] = columns_drop
    dados_tratados['colunas_onhot_nan'] = colunas_onhot_nan
    dados_tratados['col_numeric'] = col_numeric
    dados_tratados['col_categorical'] = col_categorical
    with open(diretorio + '/dados_tratados.pkl', 'wb') as f:
        pickle.dump(dados_tratados, f)

else:
    with open(diretorio + '/dados_tratados.pkl', 'rb') as f:
        dados_tratados = pickle.load(f)

    x_train = dados_tratados['x_train']
    x_test = dados_tratados['x_test']
    y_train = dados_tratados['y_train']
    y_test = dados_tratados['y_test']
    modelo_input_cat = dados_tratados['modelo_input_cat']
    modelo_input_num = dados_tratados['modelo_input_num']
    modelo_stard = dados_tratados['modelo_stard']
    model_onehot = dados_tratados['model_onehot']
    columns_drop = dados_tratados['columns_drop']
    colunas_onhot_nan =  dados_tratados['colunas_onhot_nan']
    col_numeric = dados_tratados['col_numeric']
    col_categorical = dados_tratados['col_categorical']


    print('X_treino:',x_train.shape, 'X_teste:',x_test.shape,  'Y_train:',y_train.shape, 'Y_teste:',y_test.shape,)
    print('Dowload dos dados com sucesso!')

```

    X_treino: (2493556, 59) X_teste: (623389, 59) Y_train: (2493556,) Y_teste: (623389,)
    Dowload dos dados com sucesso!



```python
x_train_colunas = x_train.copy()
x_test_colunas = x_test.copy()

x_train = x_train.values
x_test = x_test.values
```

# 6_Treinamento_Dos_Modelos

Primeiramente, utilizei sete algoritmos para o treinamento com seus parâmetros padrão:

- `LogisticRegression()`
- `RandomForestClassifier()`
- `GradientBoostingClassifier()`
- `CatBoostClassifier()`
- `keras.Sequential()`
- `SGDClassifier(loss='log_loss')`
- `GaussianNB()`
--------------------------------------------------------------------------
Os melhores algoritmos com os parâmetros padrão, observando o `MCC`, foram o Random Forest com MCC de 0.87, seguido pelo Cat Boost com 0.58. Utilizando outras métricas, o Random Forest continua sendo o melhor. Sob a ótica da calibração, ele apresenta o menor Log Loss, o menor Brier Loss, e o menor ECE.


| Model                | Log Loss | Brier Loss | ECE  | MCC  |
|----------------------|----------|------------|------|------|
| Random Forest        | 0.35     | 0.098      | 0.005|0.866 |
| Gradient Boost       | 0.50     | 0.17       | 0.005| 0.56 |
| Cat Boost            | 0.52     | 0.16       | 0.02 | 0.58 |
| SGD Classifier       | 1.01     | 0.32       | 0.01 | 0.15 |
| Regressão Logística  | 1.01     | 0.32       | 0.01 | 0.14 |
| Redes Neurais        | 2.50     | 0.38       | 0.03 | 0.19 |
| Naive Bayes          | 4.61     | 0.38       | 0.04 | 0.17 |

-----------------------------------------------------------------------
Em termos de acurácia, o Random Forest obteve 93.4% de acerto, sendo o melhor. O próximo passo será a tunagem dos parâmetros, onde escolherei três algoritmos: Random Forest, Cat Boost, e Redes Neurais.

As tunagens dos parâmetros ocorreram em outros arquivos, com as seguintes configurações selecionadas:

- `CatBoostClassifier({'depth': 9, 'learning_rate': 0.15, 'l2_leaf_reg': 1, 'iterations': 500})`
- `RandomForestClassifier(criterion='entropy', max_depth=90, max_features=8, n_estimators=70)`
- `Redes Neurais`

      | Layer (type)   | Output Shape | Param # |
      |----------------|--------------|---------|
      | dense (Dense)  | (None, 8)    | 480     |
      | dense_1 (Dense)| (None, 8)    | 72      |
      | dense_2 (Dense)| (None, 8)    | 72      |
      | dense_3 (Dense)| (None, 8)    | 72      |
      | dense_4 (Dense)| (None, 8)    | 72      |
      | dense_5 (Dense)| (None, 1)    | 9       |



## Treinando modelos com parametros padrões


```python
## criar a redes_neural

model_rede = keras.Sequential()
model_rede.add(layers.Dense(units=40,activation='relu',input_shape=(x_train.shape[1],)))
#model.add(layers.Dropout(0.20))
model_rede.add(layers.Dense(units=60,activation='relu'))
#model.add(layers.Dropout(0.20))
model_rede.add(layers.Dense(units=40,activation='relu'))
#model.add(layers.Dropout(0.20))
model_rede.add(layers.Dense(units=1,activation='sigmoid'))
model_rede.summary()
opt = keras.optimizers.SGD(learning_rate=0.01)
model_rede.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy', 'Recall'])
#epochs_hist = model.fit(x_treino, y_treino.values, epochs=100,batch_size=500)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">2,400</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">60</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">2,460</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">40</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">2,440</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │              <span style="color: #00af00; text-decoration-color: #00af00">41</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">7,341</span> (28.68 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">7,341</span> (28.68 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
if IN_COLAB:
    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/modelos'
else:
    diretorio = 'modelos'

modelos = {
    #'logistic_regression': LogisticRegression(),
    #'random_forest': RandomForestClassifier(),
    #'gradient_boost': GradientBoostingClassifier(),
    #'cat_boost': CatBoostClassifier(),
    #'redes_neural': model_rede,
    'sgd_classifier': SGDClassifier(loss='log_loss'),
    'naive_bayes': GaussianNB()
}


for key,model in modelos.items():
    if key == 'redes_neural':
        epochs_hist = model.fit(x_train, y_train, epochs=100,batch_size=500,verbose=0)
        print('O modelo treinado é o:', key)
        with open(diretorio + f'/modelo_base_{key}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print('----------------------------------------------------')
    else:
        model = model
        if key == 'cat_boost':
            model_treinado = model.fit(x_train, y_train,verbose=0)
        else:
            model_treinado = model.fit(x_train, y_train)
        print('O modelo treinado e de:', key)
        with open(diretorio + f'/modelo_base_{key}.pkl', 'wb') as f:
            pickle.dump(model, f)

        print('----------------------------------------------------')
```

    O modelo treinado e de: sgd_classifier
    ----------------------------------------------------
    O modelo treinado e de: naive_bayes
    ----------------------------------------------------



```python
if IN_COLAB:
    sys.path.append("/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/modelos")
    # Verifica os arquivos dentro da pasta 'dados'
    arquivos = os.listdir("/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/modelos")
    modelos_base = [arquivo for arquivo in arquivos if 'modelo_base' in arquivo]
    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/modelos'
else:
    sys.path.append("modelos")
    # Verifica os arquivos dentro da pasta 'dados'
    arquivos = os.listdir("modelos")
    modelos_base = [arquivo for arquivo in arquivos if 'modelo_base' in arquivo]
    diretorio = 'modelos'


dic_modelos_padrao = {}
for index, model in enumerate(modelos_base):
    with open(diretorio+'/'+ model, 'rb') as f:
        dic_modelos_padrao[modelos_base[index].split('.')[0].split('modelo_base_')[1]] = pickle.load(f)

matthews = {}
for key,model in dic_modelos_padrao.items():
    if key == 'redes_neural':
        print('O modelo treinado é o:', key)
        y_pred = model.predict(x_test).squeeze()
        y_pred_redes = (y_pred > 0.5).astype(int)
        matthews[key] = matthews_corrcoef(y_test,y_pred_redes)
        print(classification_report(y_test, y_pred_redes))
        print('----------------------------------------------------')

    else:
        y_pred = model.predict(x_test).squeeze()
        print('O modelo treinado e de:', key)
        print(key, model.score(x_test, y_test))
        print(classification_report(y_test, model.predict(x_test)))
        matthews[key] = matthews_corrcoef(y_test,model.predict(x_test))
        print('----------------------------------------------------')
```

    O modelo treinado e de: logistic_regression
    logistic_regression 0.5661841963846009
                  precision    recall  f1-score   support
    
               0       0.56      1.00      0.72    341025
               1       0.93      0.05      0.09    282364
    
        accuracy                           0.57    623389
       macro avg       0.74      0.52      0.40    623389
    weighted avg       0.73      0.57      0.43    623389
    
    ----------------------------------------------------
    O modelo treinado e de: random_forest
    random_forest 0.9336497756617457
                  precision    recall  f1-score   support
    
               0       0.94      0.94      0.94    341025
               1       0.93      0.93      0.93    282364
    
        accuracy                           0.93    623389
       macro avg       0.93      0.93      0.93    623389
    weighted avg       0.93      0.93      0.93    623389
    
    ----------------------------------------------------
    O modelo treinado e de: gradient_boost
    gradient_boost 0.7665181772536891
                  precision    recall  f1-score   support
    
               0       0.88      0.66      0.76    341025
               1       0.69      0.89      0.78    282364
    
        accuracy                           0.77    623389
       macro avg       0.78      0.78      0.77    623389
    weighted avg       0.79      0.77      0.77    623389
    
    ----------------------------------------------------
    O modelo treinado e de: cat_boost
    cat_boost 0.7814302145209492
                  precision    recall  f1-score   support
    
               0       0.88      0.69      0.78    341025
               1       0.71      0.89      0.79    282364
    
        accuracy                           0.78    623389
       macro avg       0.79      0.79      0.78    623389
    weighted avg       0.80      0.78      0.78    623389
    
    ----------------------------------------------------
    O modelo treinado é o: redes_neural
    [1m19481/19481[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 1ms/step
                  precision    recall  f1-score   support
    
               0       0.70      0.34      0.46    341025
               1       0.51      0.83      0.63    282364
    
        accuracy                           0.56    623389
       macro avg       0.61      0.58      0.54    623389
    weighted avg       0.62      0.56      0.53    623389
    
    ----------------------------------------------------
    O modelo treinado e de: sgd_classifier
    sgd_classifier 0.5668675578170291
                  precision    recall  f1-score   support
    
               0       0.56      1.00      0.72    341025
               1       0.93      0.05      0.09    282364
    
        accuracy                           0.57    623389
       macro avg       0.75      0.52      0.40    623389
    weighted avg       0.73      0.57      0.43    623389
    
    ----------------------------------------------------
    O modelo treinado e de: naive_bayes
    naive_bayes 0.592642796071153
                  precision    recall  f1-score   support
    
               0       0.58      0.92      0.71    341025
               1       0.67      0.20      0.30    282364
    
        accuracy                           0.59    623389
       macro avg       0.63      0.56      0.51    623389
    weighted avg       0.62      0.59      0.53    623389
    
    ----------------------------------------------------



```python
pd.DataFrame(list(matthews.items()), columns=['Model', 'MCC']).sort_values(by='MCC', ascending=False)
```





  <div id="df-1cb51901-4d0c-481f-9741-379b263c459a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>random_forest</td>
      <td>0.866100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat_boost</td>
      <td>0.583948</td>
    </tr>
    <tr>
      <th>2</th>
      <td>gradient_boost</td>
      <td>0.560476</td>
    </tr>
    <tr>
      <th>4</th>
      <td>redes_neural</td>
      <td>0.187552</td>
    </tr>
    <tr>
      <th>6</th>
      <td>naive_bayes</td>
      <td>0.172149</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sgd_classifier</td>
      <td>0.147736</td>
    </tr>
    <tr>
      <th>0</th>
      <td>logistic_regression</td>
      <td>0.144209</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1cb51901-4d0c-481f-9741-379b263c459a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1cb51901-4d0c-481f-9741-379b263c459a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1cb51901-4d0c-481f-9741-379b263c459a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4a446a75-023d-445a-b5c9-deb26a726e1c">
  <button class="colab-df-quickchart" onclick="quickchart('df-4a446a75-023d-445a-b5c9-deb26a726e1c')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4a446a75-023d-445a-b5c9-deb26a726e1c button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
avali = AvaliacaoModelos()
```


```python
results_df, predictions = avali.calibrations_metrics(dic_modelos_padrao, x_test, y_test)
```

    [1m19481/19481[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m28s[0m 1ms/step



```python
results_df.sort_values(by='Log Loss')
```





  <div id="df-dbf35967-74e4-43ab-9a16-ebcfb42d2ce2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Log Loss</th>
      <th>Brier Loss</th>
      <th>ECE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>random_forest</td>
      <td>0.354361</td>
      <td>0.098235</td>
      <td>0.005220</td>
    </tr>
    <tr>
      <th>2</th>
      <td>gradient_boost</td>
      <td>0.502887</td>
      <td>0.165515</td>
      <td>0.005129</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cat_boost</td>
      <td>0.518413</td>
      <td>0.156549</td>
      <td>0.020920</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sgd_classifier</td>
      <td>1.006149</td>
      <td>0.320425</td>
      <td>0.009015</td>
    </tr>
    <tr>
      <th>0</th>
      <td>logistic_regression</td>
      <td>1.007042</td>
      <td>0.320729</td>
      <td>0.009224</td>
    </tr>
    <tr>
      <th>4</th>
      <td>redes_neural</td>
      <td>2.496958</td>
      <td>0.381925</td>
      <td>0.026467</td>
    </tr>
    <tr>
      <th>6</th>
      <td>naive_bayes</td>
      <td>4.601501</td>
      <td>0.376481</td>
      <td>0.036788</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-dbf35967-74e4-43ab-9a16-ebcfb42d2ce2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-dbf35967-74e4-43ab-9a16-ebcfb42d2ce2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-dbf35967-74e4-43ab-9a16-ebcfb42d2ce2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-d0a82ccc-81b4-410d-8674-2fb3c42b4d97">
  <button class="colab-df-quickchart" onclick="quickchart('df-d0a82ccc-81b4-410d-8674-2fb3c42b4d97')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-d0a82ccc-81b4-410d-8674-2fb3c42b4d97 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
avali.calibracao_plot(y_test, predictions)
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_64_0.png)
    


# 7_Calibrando_os_modelos

Nessa parte sera utilizado o algoritimo Venn-ABERS para calibrar os modelos. A calibração Venn-Abers é uma técnica avançada usada para melhorar a confiança das previsões probabilísticas de modelos de classificação. É uma extensão dos métodos de calibração de probabilidade tradicionais, como Platt Scaling e Isotonic Regression. A calibração Venn-Abers é especialmente útil quando se deseja obter intervalos de confiança para as previsões, em vez de apenas uma única probabilidade.


```python
if IN_COLAB:
    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/modelos'
else:
    diretorio = 'modelos'


selecionando_modelos = ['modelo_base_random_forest.pkl', 'modelo_base_gradient_boost.pkl', 'modelo_base_cat_boost.pkl', 'modelo_base_redes_neural.pkl', ## Modelos base
                        'modelo_cat_boost.pkl','modelo_random_forest.pkl', 'modelo_redes_neural.h5'] # Modelos tunado

## fazendo dowload dos modelos
dic_modelos = {}
for index, model in enumerate(selecionando_modelos):
    print(selecionando_modelos[index])
    with open(diretorio+'/'+ model, 'rb') as f:
        if '_base_' in selecionando_modelos[index]:
            dic_modelos[selecionando_modelos[index].split('.pkl')[0]] = pickle.load(f)
        else:
            try:
                dic_modelos[selecionando_modelos[index].split('.pkl')[0] + '_tunado'] = pickle.load(f)
            except UnpicklingError:
                dic_modelos[selecionando_modelos[index].split('.h5')[0] + '_tunado'] = keras.models.load_model(diretorio + '/modelo_redes_neural.h5')

```

    modelo_base_random_forest.pkl
    modelo_base_gradient_boost.pkl
    modelo_base_cat_boost.pkl
    modelo_base_redes_neural.pkl
    modelo_cat_boost.pkl
    modelo_random_forest.pkl
    modelo_redes_neural.h5


    WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.



```python
##  Venn-ABERS calibration no modelo base random forest
va_random = VennAbersCalibrator(estimator=dic_modelos['modelo_base_random_forest'], inductive=False, n_splits=10)
va_random.fit(x_train, y_train)
with open(diretorio + f'/random_forest_base_ven.pkl', 'wb') as f:
      pickle.dump(va_random, f)
```


```python
## Venn-ABERS calibration no  modelo base Cat boost
va_cat = VennAbersCalibrator(estimator=dic_modelos['modelo_base_cat_boost'], inductive=False, n_splits=10)
va_cat.fit(x_train, y_train)
with open(diretorio + f'/cat_boost_base_ven.pkl', 'wb') as f:
      pickle.dump(va_cat, f)
```


```python
## Venn-ABERS calibration no  modelo base redes neurais
va_redes = VennAbersCalibrator(estimator=KerasTrainedRegressor(dic_modelos['modelo_base_redes_neural']), inductive=False, n_splits=10)
va_redes.fit(x_train, y_train)
with open(diretorio + f'/redes_neural_base_ven.pkl', 'wb') as f:
      pickle.dump(va_redes, f)
```

    [1m38962/38962[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 1ms/step
    [1m38962/38962[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m58s[0m 1ms/step
    [1m13637/13637[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 1ms/step



```python
## Venn-ABERS calibration no  modelo base Gradiente Boost
va_grad = VennAbersCalibrator(estimator=dic_modelos['modelo_base_gradient_boost'], inductive=False, n_splits=10)
va_grad.fit(x_train, y_train)
with open(diretorio + f'/gradiente_boost_base_ven.pkl', 'wb') as f:
      pickle.dump(va_grad, f)
```


```python
##  Venn-ABERS calibration no modelo tunado random forest
va_random_tunado = VennAbersCalibrator(estimator=dic_modelos['modelo_random_forest_tunado'].best_estimator_, inductive=False, n_splits=10)
va_random_tunado.fit(x_train, y_train)
with open(diretorio + f'/random_forest_tunado_ven.pkl', 'wb') as f:
      pickle.dump(va_random_tunado, f)
```


```python
## Venn-ABERS calibration no  modelo base Cat boost
va_cat_tunado = VennAbersCalibrator(estimator=dic_modelos['modelo_cat_boost_tunado'], inductive=False, n_splits=10)
va_cat_tunado.fit(x_train, y_train)
with open(diretorio + f'/cat_boost_tunado_ven.pkl', 'wb') as f:
      pickle.dump(va_cat_tunado, f)
```


```python
## Venn-ABERS calibration no  modelo base redes neurais
va_redes = VennAbersCalibrator(estimator=KerasTrainedRegressor(dic_modelos['modelo_redes_neural_tunado']), inductive=False, n_splits=10)
va_redes.fit(x_train, y_train)
with open(diretorio + f'/redes_neural_tunado_ven.pkl', 'wb') as f:
      pickle.dump(va_redes, f)
```

    [1m38962/38962[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 1ms/step
    [1m38962/38962[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m55s[0m 1ms/step
    [1m13637/13637[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m19s[0m 1ms/step


# 8_Avaliando_os_ modelos

## Análise dos Modelos de Classificação

Com base nos resultados fornecidos para diferentes modelos de classificação, podemos analisar o desempenho de cada um em termos de métricas como **Log Loss**, **Brier Loss**, **ECE (Expected Calibration Error)**, **Accuracy**, **Recall**, **Precision**, **F1 Score**, e **MCC (Matthews Correlation Coefficient)**.

### 1. Melhor Modelo: `modelo_random_forest_tunado`
- **Log Loss**: 0.315282 (baixo, indicando boa confiança nas previsões)
- **Brier Loss**: 0.084112 (baixo, sugerindo boa calibração de probabilidade)
- **ECE**: 0.006145 (baixo, indicando excelente calibração)
- **Accuracy**: 94.44% (alta, indicando bom desempenho geral)
- **F1 Score**: 0.939083 (alto, equilíbrio entre precisão e recall)
- **MCC**: 0.887996 (alto, indicando forte correlação entre previsões e rótulos reais)

**Conclusão:** O modelo Random Forest tunado (`modelo_random_forest_tunado`) se destaca como o melhor modelo em geral, com excelente desempenho em todas as métricas, especialmente em Log Loss, Brier Loss, ECE, e MCC.

### 2. Modelos com Venn-Abers Calibration
- **`random_forest_tunado_ven`**: Um modelo que utiliza calibração Venn-Abers, melhorando a **Log Loss** para 0.236318 e mantendo a **Accuracy** em 93.42%. Embora tenha um **ECE** ligeiramente maior que o modelo base tunado, ainda oferece uma boa calibração com um **F1 Score** de 0.928331.
- **`random_forest_base_ven`**: Com uma **Log Loss** de 0.251590 e **Accuracy** de 92.78%, este modelo mostra melhorias em relação ao modelo base, mas ainda não alcança o desempenho do modelo tunado.

**Conclusão:** A calibração Venn-Abers melhora a confiabilidade das previsões em termos de probabilidade (Log Loss), especialmente no caso do modelo `random_forest_tunado_ven`. No entanto, o modelo base tunado sem a calibração ainda supera em termos de desempenho geral.

### 3. Modelos Base
- **Random Forest (`modelo_base_random_forest`)**: Mostra uma **Log Loss** de 0.354361, **Accuracy** de 93.40% e **MCC** de 0.867110. Embora seja um modelo forte, ele é superado pela versão tunada.
- **CatBoost (`modelo_base_cat_boost`)** e **Gradient Boosting (`modelo_base_gradient_boost`)**: Estes modelos apresentam desempenho inferior com **Log Loss** acima de 0.5 e **MCC** abaixo de 0.6, sugerindo que são menos eficazes para esta tarefa.

**Conclusão:** Entre os modelos base, o Random Forest ainda é o melhor, mas é claramente superado quando tunado.

### 4. Modelos Tunados de CatBoost e Redes Neurais
- **CatBoost Tunado (`modelo_cat_boost_tunado`)** e sua versão calibrada (`cat_boost_tunado_ven`) têm **Log Loss** mais altas (0.381284 e 0.448885, respectivamente) e **MCC** mais baixos, sugerindo que o tuning não foi tão eficaz para este modelo.
- **Redes Neurais** (tanto a base quanto a tunada) apresentam **Log Loss** muito elevadas (>1.2), indicando previsões altamente incertas. Além disso, os valores de **Accuracy** e **MCC** são os mais baixos entre todos os modelos.

**Conclusão:** O tuning não trouxe benefícios significativos para os modelos de CatBoost e Redes Neurais, com ambos apresentando um desempenho inferior quando comparados ao Random Forest.

### Conclusão Geral do melhor modelo `random_forest_tunado_ven`

O **Random Forest Tunado** (`modelo_random_forest_tunado`) destacou-se como o melhor modelo em termos de precisão, calibração e desempenho geral. A calibração Venn-Abers aplicada ao modelo tunado (`random_forest_tunado_ven`) melhorou a confiabilidade das previsões probabilísticas, tornando-as mais calibradas. Embora o `random_forest_tunado_ven` não tenha obtido os melhores resultados em **Accuracy** e **F1 Score**, ele ficou muito próximo do melhor modelo, com apenas 1% de diferença tanto em **Accuracy** quanto em **F1 Score**, e 2% em **MCC**.

Dado que o `random_forest_tunado_ven` apresentou a melhor calibração e ficou próximo dos melhores resultados nas outras métricas, ele foi escolhido como o algoritmo ideal para esta tarefa. Portanto, para essa tarefa específica, o **Random Forest Tunado com Calibração Venn-Abers** (`random_forest_tunado_ven`) foi selecionado como a melhor escolha, equilibrando excelente calibração com um desempenho robusto nas principais métricas.






| Model                           | Log Loss | Brier Loss | ECE    | Accuracy | Recall  | Precision | F1 Score | MCC      |
|---------------------------------|----------|------------|--------|----------|---------|-----------|----------|----------|
| modelo_random_forest_tunado     | 0.315282 | 0.084112   | 0.006145 | 0.944369 | 0.946679 | 0.931607  | 0.939083 | 0.887996 |
| random_forest_tunado_ven        | 0.236318 | 0.062010   | 0.009572 | 0.934279 | 0.939723 | 0.917212  | 0.928331 | 0.867880 |
| modelo_base_random_forest       | 0.354361 | 0.098235   | 0.005220 | 0.934009 | 0.935063 | 0.920503  | 0.927726 | 0.867110 |
| random_forest_base_ven          | 0.251590 | 0.067133   | 0.006276 | 0.927767 | 0.924810 | 0.916478  | 0.920625 | 0.854388 |
| cat_boost_base_ven              | 0.439283 | 0.127614   | 0.022030 | 0.837674 | 0.941788 | 0.758314  | 0.840151 | 0.695645 |
| modelo_cat_boost_tunado         | 0.381284 | 0.118446   | 0.017859 | 0.831715 | 0.938151 | 0.751824  | 0.834716 | 0.684560 |
| cat_boost_tunado_ven            | 0.448885 | 0.135861   | 0.018358 | 0.815796 | 0.944936 | 0.728808  | 0.822918 | 0.661052 |
| modelo_base_cat_boost           | 0.518413 | 0.156549   | 0.020920 | 0.781430 | 0.886668 | 0.706012  | 0.786094 | 0.583948 |
| modelo_base_gradient_boost      | 0.502887 | 0.165515   | 0.005129 | 0.766518 | 0.890432 | 0.686885  | 0.775525 | 0.560476 |
| gradiente_boost_base_ven        | 0.587717 | 0.180181   | 0.018253 | 0.761106 | 0.815826 | 0.703862  | 0.755720 | 0.529928 |
| redes_neural_base_ven           | 1.892259 | 0.378667   | 0.027182 | 0.572719 | 0.814661 | 0.518018  | 0.633324 | 0.205695 |
| modelo_base_redes_neural        | 2.496958 | 0.381925   | 0.026467 | 0.559716 | 0.828601 | 0.508582  | 0.630297 | 0.187552 |
| modelo_redes_neural_tunado      | 1.205225 | 0.328330   | 0.016000 | 0.602014 | 0.421970 | 0.583966  | 0.489924 | 0.183589 |
| redes_neural_tunado_ven         | 1.197619 | 0.341152   | 0.012570 | 0.600280 | 0.430437 | 0.579046  | 0.493803 | 0.180481 |



```python
##Local do diretorio
if IN_COLAB:
    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/modelos'
else:
    diretorio = 'modelos'
##Pegando os amodelos salvos
selecionando_modelos = ['modelo_base_random_forest.pkl', 'modelo_base_gradient_boost.pkl', 'modelo_base_cat_boost.pkl', 'modelo_base_redes_neural.pkl', ## Modelos base
                       'random_forest_base_ven.pkl', 'gradiente_boost_base_ven.pkl', 'cat_boost_base_ven.pkl', 'redes_neural_base_ven.pkl', ## Modelos Calibrados com VennAbersCalibrator
                        'modelo_cat_boost.pkl','modelo_random_forest.pkl','modelo_redes_neural.h5',  # Modelos tunado
                        'redes_neural_tunado_ven.pkl', 'random_forest_tunado_ven.pkl', 'cat_boost_tunado_ven.pkl'] # Modelos tunado com VennAbersCalibrator
dic_modelos = {}
for index, model in enumerate(selecionando_modelos):
    with open(diretorio+'/'+ model, 'rb') as f:
        if '_base_' in selecionando_modelos[index]:
            dic_modelos[selecionando_modelos[index].split('.pkl')[0]] = pickle.load(f)
        elif '_ven' in selecionando_modelos[index]:
            dic_modelos[selecionando_modelos[index].split('.pkl')[0]] = pickle.load(f)
        else:
            try:
                dic_modelos[selecionando_modelos[index].split('.pkl')[0] + '_tunado'] = pickle.load(f)
            except UnpicklingError:
                dic_modelos[selecionando_modelos[index].split('.h5')[0] + '_tunado'] = keras.models.load_model(diretorio + '/modelo_redes_neural.h5')
#
```

    WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.



```python
dic_modelos
```




    {'modelo_base_random_forest': RandomForestClassifier(),
     'modelo_base_gradient_boost': GradientBoostingClassifier(),
     'modelo_base_cat_boost': <catboost.core.CatBoostClassifier at 0x7fe98333a710>,
     'modelo_base_redes_neural': <Sequential name=sequential, built=True>,
     'random_forest_base_ven': <venn_abers.venn_abers.VennAbersCalibrator at 0x7fe8c68805e0>,
     'gradiente_boost_base_ven': <venn_abers.venn_abers.VennAbersCalibrator at 0x7fe8c67f9870>,
     'cat_boost_base_ven': <venn_abers.venn_abers.VennAbersCalibrator at 0x7fe8c3ea1690>,
     'redes_neural_base_ven': <venn_abers.venn_abers.VennAbersCalibrator at 0x7fe8c3eb0070>,
     'modelo_cat_boost_tunado': <catboost.core.CatBoostClassifier at 0x7fe8c3f22500>,
     'modelo_random_forest_tunado': RandomizedSearchCV(estimator=RandomForestClassifier(),
                        param_distributions={'criterion': ['gini', 'entropy'],
                                             'max_depth': [1, 2, 60, 70, 80, 90,
                                                           100],
                                             'max_features': [1, 3, 4, 5, 6, 8, 10],
                                             'n_estimators': [10, 20, 30, 40, 50, 60,
                                                              70, 80, 90, 100]},
                        scoring=make_scorer(matthews_corrcoef, response_method='predict'),
                        verbose=1),
     'modelo_redes_neural_tunado': <Sequential name=sequential, built=True>,
     'redes_neural_tunado_ven': <venn_abers.venn_abers.VennAbersCalibrator at 0x7fe8c3eb0130>,
     'random_forest_tunado_ven': <venn_abers.venn_abers.VennAbersCalibrator at 0x7fe8c3dab580>,
     'cat_boost_tunado_ven': <venn_abers.venn_abers.VennAbersCalibrator at 0x7fe8c3dab550>}




```python
avali = AvaliacaoModelos()
```


```python
results_df, predictions = avali.calibrations_metrics(dic_modelos, x_test, y_test)
```

    [1m19481/19481[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 2ms/step
    [1m19481/19481[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 2ms/step
    [1m19481/19481[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m35s[0m 2ms/step
    [1m19481/19481[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 2ms/step



```python
results_df.sort_values(by='MCC', ascending=False)
```





  <div id="df-1d9a89c1-5a5f-422b-8598-0320a71a5d92" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Log Loss</th>
      <th>Brier Loss</th>
      <th>ECE</th>
      <th>Accuracy</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1 Score</th>
      <th>MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>modelo_random_forest_tunado</td>
      <td>0.315282</td>
      <td>0.084112</td>
      <td>0.006145</td>
      <td>0.944369</td>
      <td>0.946679</td>
      <td>0.931607</td>
      <td>0.939083</td>
      <td>0.887996</td>
    </tr>
    <tr>
      <th>12</th>
      <td>random_forest_tunado_ven</td>
      <td>0.236318</td>
      <td>0.062010</td>
      <td>0.009572</td>
      <td>0.934279</td>
      <td>0.939723</td>
      <td>0.917212</td>
      <td>0.928331</td>
      <td>0.867880</td>
    </tr>
    <tr>
      <th>0</th>
      <td>modelo_base_random_forest</td>
      <td>0.354361</td>
      <td>0.098235</td>
      <td>0.005220</td>
      <td>0.934009</td>
      <td>0.935063</td>
      <td>0.920503</td>
      <td>0.927726</td>
      <td>0.867110</td>
    </tr>
    <tr>
      <th>4</th>
      <td>random_forest_base_ven</td>
      <td>0.251590</td>
      <td>0.067133</td>
      <td>0.006276</td>
      <td>0.927767</td>
      <td>0.924810</td>
      <td>0.916478</td>
      <td>0.920625</td>
      <td>0.854388</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cat_boost_base_ven</td>
      <td>0.439283</td>
      <td>0.127614</td>
      <td>0.022030</td>
      <td>0.837674</td>
      <td>0.941788</td>
      <td>0.758314</td>
      <td>0.840151</td>
      <td>0.695645</td>
    </tr>
    <tr>
      <th>8</th>
      <td>modelo_cat_boost_tunado</td>
      <td>0.381284</td>
      <td>0.118446</td>
      <td>0.017859</td>
      <td>0.831715</td>
      <td>0.938151</td>
      <td>0.751824</td>
      <td>0.834716</td>
      <td>0.684560</td>
    </tr>
    <tr>
      <th>13</th>
      <td>cat_boost_tunado_ven</td>
      <td>0.448885</td>
      <td>0.135861</td>
      <td>0.018358</td>
      <td>0.815796</td>
      <td>0.944936</td>
      <td>0.728808</td>
      <td>0.822918</td>
      <td>0.661052</td>
    </tr>
    <tr>
      <th>2</th>
      <td>modelo_base_cat_boost</td>
      <td>0.518413</td>
      <td>0.156549</td>
      <td>0.020920</td>
      <td>0.781430</td>
      <td>0.886668</td>
      <td>0.706012</td>
      <td>0.786094</td>
      <td>0.583948</td>
    </tr>
    <tr>
      <th>1</th>
      <td>modelo_base_gradient_boost</td>
      <td>0.502887</td>
      <td>0.165515</td>
      <td>0.005129</td>
      <td>0.766518</td>
      <td>0.890432</td>
      <td>0.686885</td>
      <td>0.775525</td>
      <td>0.560476</td>
    </tr>
    <tr>
      <th>5</th>
      <td>gradiente_boost_base_ven</td>
      <td>0.587717</td>
      <td>0.180181</td>
      <td>0.018253</td>
      <td>0.761106</td>
      <td>0.815826</td>
      <td>0.703862</td>
      <td>0.755720</td>
      <td>0.529928</td>
    </tr>
    <tr>
      <th>7</th>
      <td>redes_neural_base_ven</td>
      <td>1.892259</td>
      <td>0.378667</td>
      <td>0.027182</td>
      <td>0.572719</td>
      <td>0.814661</td>
      <td>0.518018</td>
      <td>0.633324</td>
      <td>0.205695</td>
    </tr>
    <tr>
      <th>3</th>
      <td>modelo_base_redes_neural</td>
      <td>2.496958</td>
      <td>0.381925</td>
      <td>0.026467</td>
      <td>0.559716</td>
      <td>0.828601</td>
      <td>0.508582</td>
      <td>0.630297</td>
      <td>0.187552</td>
    </tr>
    <tr>
      <th>10</th>
      <td>modelo_redes_neural_tunado</td>
      <td>1.205225</td>
      <td>0.328330</td>
      <td>0.016000</td>
      <td>0.602014</td>
      <td>0.421970</td>
      <td>0.583966</td>
      <td>0.489924</td>
      <td>0.183589</td>
    </tr>
    <tr>
      <th>11</th>
      <td>redes_neural_tunado_ven</td>
      <td>1.197619</td>
      <td>0.341152</td>
      <td>0.012570</td>
      <td>0.600280</td>
      <td>0.430437</td>
      <td>0.579046</td>
      <td>0.493803</td>
      <td>0.180481</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1d9a89c1-5a5f-422b-8598-0320a71a5d92')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-1d9a89c1-5a5f-422b-8598-0320a71a5d92 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1d9a89c1-5a5f-422b-8598-0320a71a5d92');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-7ed6963c-f4a0-4a77-bdff-88605c19c070">
  <button class="colab-df-quickchart" onclick="quickchart('df-7ed6963c-f4a0-4a77-bdff-88605c19c070')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-7ed6963c-f4a0-4a77-bdff-88605c19c070 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
for kay, y_proba in predictions.items():
    y_pred = (y_proba > 0.5).astype(int).squeeze()
    print('O modelo treinado é o:',key.replace('_', ' ').title())
    print(classification_report(y_test, y_pred))
    print('----------------------------------------------------')

```

    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.94      0.94      0.94    341025
               1       0.93      0.93      0.93    282364
    
        accuracy                           0.93    623389
       macro avg       0.93      0.93      0.93    623389
    weighted avg       0.93      0.93      0.93    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.88      0.66      0.76    341025
               1       0.69      0.89      0.78    282364
    
        accuracy                           0.77    623389
       macro avg       0.78      0.78      0.77    623389
    weighted avg       0.79      0.77      0.77    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.88      0.69      0.78    341025
               1       0.71      0.89      0.79    282364
    
        accuracy                           0.78    623389
       macro avg       0.79      0.79      0.78    623389
    weighted avg       0.80      0.78      0.78    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.70      0.34      0.46    341025
               1       0.51      0.83      0.63    282364
    
        accuracy                           0.56    623389
       macro avg       0.61      0.58      0.54    623389
    weighted avg       0.62      0.56      0.53    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.94      0.93      0.93    341025
               1       0.92      0.92      0.92    282364
    
        accuracy                           0.93    623389
       macro avg       0.93      0.93      0.93    623389
    weighted avg       0.93      0.93      0.93    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.82      0.72      0.77    341025
               1       0.70      0.82      0.76    282364
    
        accuracy                           0.76    623389
       macro avg       0.76      0.77      0.76    623389
    weighted avg       0.77      0.76      0.76    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.94      0.75      0.84    341025
               1       0.76      0.94      0.84    282364
    
        accuracy                           0.84    623389
       macro avg       0.85      0.85      0.84    623389
    weighted avg       0.86      0.84      0.84    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.71      0.37      0.49    341025
               1       0.52      0.81      0.63    282364
    
        accuracy                           0.57    623389
       macro avg       0.61      0.59      0.56    623389
    weighted avg       0.62      0.57      0.55    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.94      0.74      0.83    341025
               1       0.75      0.94      0.83    282364
    
        accuracy                           0.83    623389
       macro avg       0.84      0.84      0.83    623389
    weighted avg       0.85      0.83      0.83    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.95      0.95      0.95    341025
               1       0.94      0.94      0.94    282364
    
        accuracy                           0.94    623389
       macro avg       0.94      0.94      0.94    623389
    weighted avg       0.94      0.94      0.94    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.61      0.75      0.67    341025
               1       0.58      0.42      0.49    282364
    
        accuracy                           0.60    623389
       macro avg       0.60      0.59      0.58    623389
    weighted avg       0.60      0.60      0.59    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.61      0.74      0.67    341025
               1       0.58      0.43      0.49    282364
    
        accuracy                           0.60    623389
       macro avg       0.60      0.59      0.58    623389
    weighted avg       0.60      0.60      0.59    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.95      0.93      0.94    341025
               1       0.92      0.94      0.93    282364
    
        accuracy                           0.93    623389
       macro avg       0.93      0.93      0.93    623389
    weighted avg       0.93      0.93      0.93    623389
    
    ----------------------------------------------------
    O modelo treinado é o: Cat Boost Tunado Ven
                  precision    recall  f1-score   support
    
               0       0.94      0.71      0.81    341025
               1       0.73      0.94      0.82    282364
    
        accuracy                           0.82    623389
       macro avg       0.83      0.83      0.82    623389
    weighted avg       0.84      0.82      0.81    623389
    
    ----------------------------------------------------



```python
avali.calibracao_plot(y_test, predictions)

```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_83_0.png)
    



```python
ultimos_arrays = dict(list(predictions.items())[8:])
avali.calibracao_plot(y_test,ultimos_arrays)
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_84_0.png)
    



```python
avali.matriz_confusao_matplotlib(predictions,y_test)
```


    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_0.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_2.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_4.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_6.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_8.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_10.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_12.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_14.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_16.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_18.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_20.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_22.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_24.png)
    


    -----------------------------------------------------------------



    
![png](predicao_poisonous_mushrooms_files/predicao_poisonous_mushrooms_85_26.png)
    


    -----------------------------------------------------------------


# 9_Submissao


```python
## Tratando os dados para predicao

if IN_COLAB:
    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/dados'
else:
    diretorio = 'dados'

predict = pd.read_csv(diretorio + "/test.csv")

trat = Tratamento()

## retirando ids
predict_sem_id, predict_id = trat.retirando_ids(predict)

## Dropando colunas 'veil-type', 'spore-print-color', 'stem-root', 'veil-color', 'stem-surface', 'gill-spacing'
columns_drop = [
        "veil-type",
        "spore-print-color",
        "stem-root",
        "veil-color",
        "stem-surface",
        "gill-spacing",
    ]
predict_drop = trat.drop_columns_nan(predict_sem_id, columns_drop)

## Criando uma coluna com 1 onde tem nan e 0 onde não tem nan, nas colunas 'cap-surface', 'gill-attachment', 'ring-type'
colunas_onhot_nan = ['cap-surface', 'gill-attachment', 'ring-type']

predict_nan_onhot = trat.criando_onhot_nan(predict_drop, colunas_onhot_nan)

## Imputando em valores faltantes nas colunas categoricas

predict_imput_categorica = trat.imputar_dados_faltantes_categoricos(predict_nan_onhot,modelo_input_cat)

## Imputando em valores faltantes nas colunas numericas

predict_imput_numerical = trat.imputar_dados_faltantes_numericos(predict_imput_categorica,modelo_input_num)

## escalonando os dados numericos
col_numeric = ['cap-diameter', 'stem-height', 'stem-width']

predict_stard_numerical = trat.transformacao_variaveis_numericas(predict_imput_numerical,col_transformada=col_numeric, modelo_imputer=modelo_stard)

## Passando OneHotEncoder com os parametros minimo de 5% de frequencia onde sera criada a coluna infrequent_if_exist se nao atingir esse limite, tentando dimunir o numero de categorias para algumas colunas

col_categorical = ['cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed', 'gill-attachment', 'gill-color', 'stem-color', 'has-ring', 'ring-type', 'habitat', 'season']

predict_onhot = trat.criando_onehot(predict_stard_numerical, col_categorical, model_onehot=model_onehot)

dados_predict = predict_onhot.values

predict_onhot.head()
```





  <div id="df-9ccfbe25-8702-4d1e-82b5-b0c10ba0e4ec" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cap-diameter</th>
      <th>stem-height</th>
      <th>stem-width</th>
      <th>cap-surface_nan</th>
      <th>gill-attachment_nan</th>
      <th>ring-type_nan</th>
      <th>cap-shape_b</th>
      <th>cap-shape_f</th>
      <th>cap-shape_s</th>
      <th>cap-shape_x</th>
      <th>...</th>
      <th>ring-type_f</th>
      <th>ring-type_infrequent_sklearn</th>
      <th>habitat_d</th>
      <th>habitat_g</th>
      <th>habitat_l</th>
      <th>habitat_infrequent_sklearn</th>
      <th>season_a</th>
      <th>season_u</th>
      <th>season_w</th>
      <th>season_infrequent_sklearn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.500963</td>
      <td>11.13</td>
      <td>17.12</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.127288</td>
      <td>1.27</td>
      <td>10.75</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.925016</td>
      <td>6.18</td>
      <td>3.14</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.609325</td>
      <td>4.98</td>
      <td>8.51</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.029484</td>
      <td>6.73</td>
      <td>13.70</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9ccfbe25-8702-4d1e-82b5-b0c10ba0e4ec')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9ccfbe25-8702-4d1e-82b5-b0c10ba0e4ec button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9ccfbe25-8702-4d1e-82b5-b0c10ba0e4ec');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-471d95f1-90b8-4799-961e-fbfe9b70af99">
  <button class="colab-df-quickchart" onclick="quickchart('df-471d95f1-90b8-4799-961e-fbfe9b70af99')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-471d95f1-90b8-4799-961e-fbfe9b70af99 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
## Modelo  random_forest_tunado_ven
if IN_COLAB:
    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/modelos'
else:
    diretorio = 'modelos'

with open(diretorio + '/random_forest_tunado_ven.pkl', 'rb') as f:
    model = pickle.load(f)

```


```python
# fazendo a predicao
resultado_predict = (model.predict(dados_predict)[:, 1] > 0.5).astype(int)
```


```python
predict_id = pd.DataFrame(predict_id)
```


```python
predict_id['class'] = np.where(resultado_predict == 1, 'e', 'p')
```


```python
predict_id.head()
```





  <div id="df-f7fc6708-6a1f-4f7f-a8cc-085d7d26ddbf" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3116945</td>
      <td>e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3116946</td>
      <td>p</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3116947</td>
      <td>p</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3116948</td>
      <td>p</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3116949</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f7fc6708-6a1f-4f7f-a8cc-085d7d26ddbf')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f7fc6708-6a1f-4f7f-a8cc-085d7d26ddbf button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f7fc6708-6a1f-4f7f-a8cc-085d7d26ddbf');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f6338ea7-695d-4641-837d-757e79baa8f5">
  <button class="colab-df-quickchart" onclick="quickchart('df-f6338ea7-695d-4641-837d-757e79baa8f5')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f6338ea7-695d-4641-837d-757e79baa8f5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
if IN_COLAB:
    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/dados/'
else:
    diretorio = 'dados'
predict_id.to_csv(diretorio + 'submission.csv', index=False)
```


```python
## Modelo  random_forest_tunado_ven
if IN_COLAB:
    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/modelos'
else:
    diretorio = 'modelos'

with open(diretorio + '/modelo_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# fazendo a predicao
resultado_predict = (model.predict(dados_predict) > 0.5).astype(int)
predict_id = pd.DataFrame(predict_id)
predict_id['class'] = np.where(resultado_predict == 1, 'e', 'p')

if IN_COLAB:
    diretorio = '/content/drive/Othercomputers/Meu-laptop/binary_prediction_poisonous_mushrooms/dados/'
else:
    diretorio = 'dados'
predict_id.to_csv(diretorio + 'submission_2.csv', index=False)
```


```python
predict_id
```





  <div id="df-9d55e14e-bcc2-4191-8a0f-91fd32baf4f1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3116945</td>
      <td>e</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3116946</td>
      <td>p</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3116947</td>
      <td>p</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3116948</td>
      <td>p</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3116949</td>
      <td>e</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2077959</th>
      <td>5194904</td>
      <td>p</td>
    </tr>
    <tr>
      <th>2077960</th>
      <td>5194905</td>
      <td>e</td>
    </tr>
    <tr>
      <th>2077961</th>
      <td>5194906</td>
      <td>p</td>
    </tr>
    <tr>
      <th>2077962</th>
      <td>5194907</td>
      <td>e</td>
    </tr>
    <tr>
      <th>2077963</th>
      <td>5194908</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
<p>2077964 rows × 2 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9d55e14e-bcc2-4191-8a0f-91fd32baf4f1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9d55e14e-bcc2-4191-8a0f-91fd32baf4f1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9d55e14e-bcc2-4191-8a0f-91fd32baf4f1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-12ad02ef-9729-4dba-bdf5-ad4335f459d4">
  <button class="colab-df-quickchart" onclick="quickchart('df-12ad02ef-9729-4dba-bdf5-ad4335f459d4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-12ad02ef-9729-4dba-bdf5-ad4335f459d4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_0c3e4ae7-9f3b-4935-87f6-b6840505b194">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('predict_id')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_0c3e4ae7-9f3b-4935-87f6-b6840505b194 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('predict_id');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python

```
