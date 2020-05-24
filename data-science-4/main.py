#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[42]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.

""" Removendo espaços em branco
"""
countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()

""" Selecionando colunas númericas que estão como object
"""
columns_with_comma = list(set(list(countries.select_dtypes('object').columns)) - set(['Region', 'Country']))

""" Substituindo vírgula por ponto e convertendo para inteiro
"""
countries[columns_with_comma] = countries[columns_with_comma].apply(lambda x: x.str.replace(',', '.'))
countries[columns_with_comma] = countries[columns_with_comma].astype(float)

countries.head(5)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[71]:


def q1():
    # Retorne aqui o resultado da questão 1.
    regions = countries['Region'].unique().tolist()
    regions.sort()
    
    return regions
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[50]:


def q2():
    # Retorne aqui o resultado da questão 2.
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    pop_density = discretizer.fit_transform(countries[['Pop_density']])

    return int(sum(pop_density[:,0] == 9))
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[72]:


countries[['Region', 'Climate']].fillna(0).nunique().sum()


# In[51]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return int(countries[['Region', 'Climate']].fillna(0).nunique().sum())
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[11]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[12]:


test_country = pd.DataFrame([test_country], columns=countries.columns)

numeric_columns = countries.select_dtypes([int, float]).columns.tolist()

pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])
pipeline.fit_transform(countries[numeric_columns])

dict(zip(numeric_columns, pipeline.transform(test_country[numeric_columns])[0]))


# In[13]:


def q4():
    # Retorne aqui o resultado da questão 4.
    tranformed_test = dict(zip(numeric_columns, pipeline.transform(test_country[numeric_columns])[0]))

    return float(round(tranformed_test['Arable'], 3))
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[73]:


quantile_1 = countries['Net_migration'].quantile(q=0.25)
quantile_3 = countries['Net_migration'].quantile(q=0.75)
iqr = quantile_3 - quantile_1
iqr
outliers_baixo = len(countries.loc[countries['Net_migration'] < (quantile_1 - (1.5*iqr))])
outliers_acima = len(countries.loc[countries['Net_migration'] > (quantile_3 + (1.5*iqr))])


# In[74]:


def q5():
    # Retorne aqui o resultado da questão 4.
    return (int(outliers_baixo), int(outliers_acima), False)
q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[63]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(newsgroup.data)
X.getcol(vectorizer.vocabulary_.get(u'phone')).sum()
# newsgroup.data


# In[64]:


def q6():
    # Retorne aqui o resultado da questão 4.
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(newsgroup.data)

    return int(X.getcol(vectorizer.vocabulary_.get(u'phone')).sum())
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[67]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroup.data)
X.getcol(vectorizer.vocabulary_.get(u'phone')).sum()


# In[68]:


def q7():
    # Retorne aqui o resultado da questão 4.
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(newsgroup.data)
    return float(X.getcol(vectorizer.vocabulary_.get(u'phone')).sum().round(3))
q7()

