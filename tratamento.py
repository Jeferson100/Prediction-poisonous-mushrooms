import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
sys.path.append('/home/vscode/.local/lib/python3.10/site-packages')
from feature_engine.transformation import LogCpTransformer, YeoJohnsonTransformer, BoxCoxTransformer
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
