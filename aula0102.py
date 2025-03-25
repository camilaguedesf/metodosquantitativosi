# Instalação das bibliotecas necessárias (caso ainda não estejam instaladas)
# !pip install pandas matplotlib seaborn wooldridge

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wooldridge import data

# Carregar a base de dados 'bwght'
bwght = data("bwght")

# Exibir as primeiras linhas da base para conhecer os dados
print(bwght.head())

# Exibir as colunas disponíveis na base de dados
print(bwght.columns)

# Estatísticas descritivas da variável cigs
print("Soma de cigs:", bwght["cigs"].sum())
print("Mínimo de cigs:", bwght["cigs"].min())
print("Máximo de cigs:", bwght["cigs"].max())

# Estatísticas descritivas da variável bwght
print("Média de bwght:", bwght["bwght"].mean())
print("Mediana de bwght:", bwght["bwght"].median())
print("Quantis de bwght:\n", bwght["bwght"].quantile([0.25, 0.5, 0.75]))
print("Variância de bwght:", bwght["bwght"].var())
print("Desvio padrão de bwght:", bwght["bwght"].std())

# Correlação e covariância entre cigs e bwght
print("Correlação entre cigs e bwght:", bwght["cigs"].corr(bwght["bwght"]))
print("Covariância entre cigs e bwght:", bwght["cigs"].cov(bwght["bwght"]))

# Gráficos
# Histograma de cigs
plt.figure(figsize=(8,6))
sns.histplot(bwght["cigs"], bins=20, kde=True)
plt.title("Histograma de Cigs")
plt.xlabel("Cigarros fumados por dia")
plt.ylabel("Frequência")
plt.show()

# Gráfico de dispersão entre cigs e bwght
plt.figure(figsize=(8,6))
sns.scatterplot(x=bwght["cigs"], y=bwght["bwght"])
plt.title("Diagrama de Dispersão: Cigs vs Bwght")
plt.xlabel("Cigarros fumados por dia")
plt.ylabel("Peso ao nascer")
plt.show()

# Boxplot da variável bwght
plt.figure(figsize=(8,6))
sns.boxplot(y=bwght["bwght"])
plt.title("Boxplot de Bwght")
plt.ylabel("Peso ao nascer")
plt.show()
