# 1a lista

'''
Nome da Tarefa:
1ª lista de exercícios: estatísticas descritivas (capítulos 2 e 3)

Descrição:
Exercícios dos capítulos 2 e 3 do livro Estatística Aplicada a Administração e Economia.
Cap. 2: 10, 18, 37, 52.
Cap. 3: 12, 29, 34, 36, 44, 48, 56, 59.
Descrição de gráfico

Período:
Inicia em 19/03/2025 às 00h00 e finaliza em 31/03/2025 às 23h59
'''

# Cap 2 - 10

import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do Excel
arquivo = r"C:\Users\camil\Documents\repo\metodosquantitativosi\data_files\SBE13e_DATAfiles_CH02\HotelRatings.xlsx"
df = pd.read_excel(arquivo)

# Selecionar a primeira coluna (assumindo que contém as avaliações) e garantir que os valores sejam strings
avaliacoes = df.iloc[:, 0].astype(str).str.strip()

# Definir a ordem correta das categorias
ordem_avaliacoes = ["Terrible", "Poor", "Average", "Very Good", "Excellent"]

# Contar as ocorrências corretamente sem reindexamento inicial
frequencia = avaliacoes.value_counts()

# Reindexar garantindo que todas as categorias apareçam, preenchendo ausências com zero
frequencia = frequencia.reindex(ordem_avaliacoes, fill_value=0)

print("Distribuição de Frequência:")
print(frequencia)

# Distribuição de frequência percentual ordenada
frequencia_percentual = (frequencia / frequencia.sum()) * 100
print("\nDistribuição de Frequência Percentual:")
print(frequencia_percentual)

# Gráfico de barras da distribuição de frequência percentual
plt.figure(figsize=(8, 6))
plt.bar(frequencia_percentual.index, frequencia_percentual.values, color='blue', alpha=0.7)
plt.xlabel("Avaliação")
plt.ylabel("Frequência Percentual (%)")
plt.title("Distribuição de Frequência Percentual das Avaliações")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Comentários sobre as avaliações
mapa_avaliacoes = {"Terrible": 1, "Poor": 2, "Average": 3, "Very Good": 4, "Excellent": 5}
avaliacoes_numericas = avaliacoes.map(mapa_avaliacoes)

media = avaliacoes_numericas.mean()
mediana = avaliacoes_numericas.median()
moda = avaliacoes.mode()[0]

print("\nComentários sobre as avaliações:")
print(f"- Média das avaliações: {media:.2f}")
print(f"- Mediana das avaliações: {mediana}")
print(f"- Moda das avaliações: {moda}")

if media >= 4:
    print("- A maioria dos hóspedes avaliou o hotel positivamente.")
elif media >= 3:
    print("- As avaliações indicam uma experiência mediana, com opiniões variadas.")
else:
    print("- A avaliação geral indica uma experiência insatisfatória para a maioria dos hóspedes.")


# Criar um DataFrame com os dados do Disney’s Grand Californian
dados_disney = pd.DataFrame({
    "Classificação": ["Excelente", "Muito bom", "Regular", "Ruim", "Péssima"],
    "Frequência": [807, 521, 200, 107, 44]
})

# Garantir a mesma ordem de categorias para comparação
dados_disney["Classificação"] = pd.Categorical(dados_disney["Classificação"], 
                                               categories=["Péssima", "Ruim", "Regular", "Muito bom", "Excelente"], 
                                               ordered=True)

# Calcular distribuição percentual
dados_disney["Frequência Percentual"] = (dados_disney["Frequência"] / dados_disney["Frequência"].sum()) * 100

# Exibir os resultados
print("\nDistribuição de Frequência - Disney’s Grand Californian:")
print(dados_disney)

# Criar gráfico comparativo
plt.figure(figsize=(8, 6))
plt.bar(dados_disney["Classificação"], dados_disney["Frequência Percentual"], color='red', alpha=0.6, label="Disney’s Grand Californian")
plt.bar(frequencia_percentual.index, frequencia_percentual.values, color='blue', alpha=0.6, label="Sheraton Anaheim Hotel")

plt.xlabel("Avaliação")
plt.ylabel("Frequência Percentual (%)")
plt.title("Comparação das Avaliações dos Hotéis")
plt.legend()
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
