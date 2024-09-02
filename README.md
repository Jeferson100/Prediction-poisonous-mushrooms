# Desafio Kaggle para Predição de Cogumelos Venenosos

## Introdução

Este repositório contém o trabalho desenvolvido para o desafio **Binary Prediction of Poisonous Mushrooms** do Kaggle. O objetivo deste desafio é prever se um cogumelo é comestível ou venenoso com base em suas características físicas.

Você pode acessar a página do desafio através deste [link](https://www.kaggle.com/competitions/playground-series-s4e8/overview).


## Detalhes do Desafio

### Objetivo
O objetivo principal é classificar cada cogumelo como comestível (`e`) ou venenoso (`p`) utilizando um conjunto de dados que contém informações sobre suas características físicas. 

### Métrica de Avaliação
As submissões serão avaliadas utilizando o **coeficiente de correlação de Matthews (MCC)**, uma métrica que leva em consideração todos os elementos da matriz de confusão e é especialmente útil em problemas de classificação binária com classes desbalanceadas.
O coeficiente de correlação de Matthews é usado em aprendizado de máquina como uma medida da qualidade de classificações binárias e multiclasse. Ele leva em conta verdadeiros e falsos positivos e negativos e é geralmente considerado uma medida balanceada que pode ser usada mesmo se as classes forem de tamanhos muito diferentes. O MCC é, em essência, um valor de coeficiente de correlação entre -1 e +1. Um coeficiente de +1 representa uma previsão perfeita, 0 uma previsão aleatória média e -1 uma previsão inversa. A estatística também é conhecida como coeficiente phi.

### Formato de Submissão
Os arquivos de submissão devem seguir o formato abaixo, contendo um cabeçalho e duas colunas: `id` e `class`:

```	
id,class
3116945,e
3116946,p
3116947,e
```

# Etapas

## Preparação dos Dados

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

## Treinamento

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




# Citation
Walter Reade, Ashley Chow. (2024). Binary Prediction of Poisonous Mushrooms. Kaggle. https://kaggle.com/competitions/playground-series-s4e8

