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

## Cap 2 - 10

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

# Criar DataFrame para Disney’s Grand Californian
dados_disney = pd.DataFrame({
    "Classificação": ["Péssima", "Ruim", "Regular", "Muito bom", "Excelente"],
    "Frequência": [44, 107, 200, 521, 807]
})

# Criar DataFrame para Sheraton Anaheim Hotel (frequencia já calculada anteriormente)
dados_sheraton = pd.DataFrame({
    "Classificação": ["Péssima", "Ruim", "Regular", "Muito bom", "Excelente"],
    "Frequência": [41, 62, 107, 252, 187]
})

# Definir a mesma ordem categórica para os dois conjuntos de dados
ordem_classes = ["Péssima", "Ruim", "Regular", "Muito bom", "Excelente"]
dados_disney["Classificação"] = pd.Categorical(dados_disney["Classificação"], categories=ordem_classes, ordered=True)
dados_sheraton["Classificação"] = pd.Categorical(dados_sheraton["Classificação"], categories=ordem_classes, ordered=True)

# Calcular distribuições percentuais
dados_disney["Frequência Percentual"] = (dados_disney["Frequência"] / dados_disney["Frequência"].sum()) * 100
dados_sheraton["Frequência Percentual"] = (dados_sheraton["Frequência"] / dados_sheraton["Frequência"].sum()) * 100

# Criar gráfico comparativo de barras lado a lado
largura_barra = 0.4  # Definir largura para ajustar as barras lado a lado
x = range(len(ordem_classes))  # Criar posições para as categorias

plt.figure(figsize=(8, 6))
plt.bar([pos - largura_barra/2 for pos in x], dados_disney["Frequência Percentual"], width=largura_barra, color='red', alpha=0.6, label="Disney’s Grand Californian")
plt.bar([pos + largura_barra/2 for pos in x], dados_sheraton["Frequência Percentual"], width=largura_barra, color='blue', alpha=0.6, label="Sheraton Anaheim Hotel")

plt.xlabel("Avaliação")
plt.ylabel("Frequência Percentual (%)")
plt.title("Comparação das Avaliações dos Hotéis")
plt.xticks(ticks=x, labels=ordem_classes, rotation=0)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

## Cap 2 - 18

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lista com os valores dos PPJ (média de pontos por jogo)
ppj = [27.0, 28.8, 26.4, 27.1, 22.9, 19.2, 21.0, 20.8, 17.6, 
       21.1, 19.2, 21.2, 15.5, 17.2, 16.7, 18.5, 18.3, 18.2, 
       23.3, 16.4, 18.9, 16.5, 17.7, 16.8, 17.0, 17.4, 14.6, 
       15.7, 17.2, 17.0, 15.3, 17.8, 16.7, 17.4, 16.3, 16.7, 
       17.0, 17.5, 14.0, 16.3, 14.6]

# Criar classes de 10 a 30 com intervalo de 2
bins = np.arange(10, 32, 2)

# Criar a distribuição de frequência
frequencia, bins_edges = np.histogram(ppj, bins=bins)

# Criar DataFrame para visualizar melhor
df_frequencia = pd.DataFrame({
    "Intervalo": [f"{bins_edges[i]} - {bins_edges[i+1]}" for i in range(len(bins_edges)-1)],
    "Frequência": frequencia
})

# Exibir distribuição de frequência
print("\nDistribuição de Frequência:")
print(df_frequencia)

# Calcular distribuição de frequência relativa
df_frequencia["Frequência Relativa"] = df_frequencia["Frequência"] / len(ppj)

# Calcular distribuição de frequência percentual acumulada
df_frequencia["Frequência Percentual Acumulada"] = df_frequencia["Frequência"].cumsum() / len(ppj) * 100

# Exibir tabela completa
print("\nDistribuição de Frequência Completa:")
print(df_frequencia)

# Criar histograma
plt.figure(figsize=(8,6))
plt.hist(ppj, bins=bins, edgecolor="black", alpha=0.7, color="blue")
plt.xlabel("Média de Pontos por Jogo (PPJ)")
plt.ylabel("Frequência")
plt.title("Histograma da Média de Pontos por Jogo (PPJ)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Calcular média e mediana
media = np.mean(ppj)
mediana = np.median(ppj)

# Exibir valores médios
print(f"\nMédia: {media:.2f}, Mediana: {mediana:.2f}")

# Responder sobre a distorção dos dados
print("\nOs dados apresentam uma assimetria positiva (distorção à direita).")

# Calcular a porcentagem de jogadores com PPJ >= 20
porcentagem_20_mais = (sum(np.array(ppj) >= 20) / len(ppj)) * 100
print(f"\nPorcentagem de jogadores com pelo menos 20 pontos por jogo: {porcentagem_20_mais:.2f}%")

## Cap 2 - 37

import matplotlib.pyplot as plt
import numpy as np

# Dados extraídos da imagem
categorias_x = ['A', 'B', 'C', 'D']
valores_y_I = [143, 200, 321, 420]
valores_y_II = [857, 800, 679, 580]

# Posição das barras
x = np.arange(len(categorias_x))
largura = 0.4

# Criando o gráfico de barras
fig, ax = plt.subplots()
ax.bar(x - largura/2, valores_y_I, largura, label='I', color='blue')
ax.bar(x + largura/2, valores_y_II, largura, label='II', color='orange')

# Configurações do gráfico
ax.set_xlabel('x')
ax.set_ylabel('Frequência')
ax.set_title('Distribuição das frequências de y para cada x')
ax.set_xticks(x)
ax.set_xticklabels(categorias_x)
ax.legend(title="y")

# Exibir o gráfico
plt.show()

## Cap 2 - 52

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Criando o DataFrame a partir dos dados fornecidos
data = [
    [1, "Google", "Large", 33],
    [2, "Boston Consulting Group", "Small", 10],
    [3, "SAS Institute", "Midsized", 8],
    [4, "Wegmans Food Markets", "Large", 5],
    [5, "Edward Jones", "Large", 1],
    [6, "NetApp", "Midsized", 30],
    [7, "Camden Property Trust", "Small", -2],
    [8, "Recreational Equipment (REI)", "Large", 12],
    [9, "CHG Healthcare Services", "Small", 17],
    [10, "Quicken Loans", "Midsized", 20],
    [11, "Zappos.com", "Midsized", 70],
    [12, "Mercedes-Benz USA", "Small", 2],
    [13, "DPR Construction", "Small", 18],
    [14, "DreamWorks Animation", "Small", 8],
    [15, "NuStar Energy", "Small", 6],
    [16, "Kimpton Hotels & Restaurants", "Midsized", 4],
    [17, "JM Family Enterprises", "Midsized", -1],
    [18, "Chesapeake Energy", "Large", 23],
    [19, "Intuit", "Midsized", 9],
    [20, "USAA", "Large", 7],
    [21, "Robert W. Baird", "Midsized", 5],
    [22, "The Container Store", "Midsized", 11],
    [23, "Qualcomm", "Large", 6],
    [24, "Alston & Bird", "Small", 3],
    [25, "Ultimate Software", "Small", 15],
    [26, "Burns & McDonnell", "Midsized", 5],
    [27, "Salesforce.com", "Midsized", 39],
    [28, "Devon Energy", "Midsized", -6],
    [29, "PCL Construction", "Small", -5],
    [30, "Bingham McCutchen", "Small", -7],
    [31, "Scottrade", "Midsized", 9],
    [32, "Whole Foods Market", "Large", 6],
    [34, "Nugget Market", "Small", 8],
    [35, "Millennium: The Takeda Oncology Co.", "Small", 3],
    [36, "Southern Ohio Medical Center", "Small", 18],
    [37, "Plante Moran", "Small", 1],
    [38, "W. L. Gore & Associates", "Midsized", 2],
    [39, "St. Jude Children's Research Hospital", "Midsized", 1],
    [40, "SVB Financial Group", "Small", 9],
    [41, "Adobe", "Midsized", 11],
    [42, "Baptist Health South Florida", "Large", 10],
    [44, "Balfour Beatty Construction", "Small", -2],
    [45, "National Instruments", "Midsized", 7],
    [46, "Intel", "Large", 4],
    [47, "American Fidelity Assurance", "Small", 0],
    [48, "PricewaterhouseCoopers", "Large", 9],
    [49, "Children's Healthcare of Atlanta", "Midsized", -1],
    [50, "World Wide Technology", "Small", 23],
    [51, "Allianz Life Insurance", "Small", 2],
    [52, "Autodesk", "Midsized", 5],
    [53, "Methodist Hospital", "Large", 8],
    [54, "Baker Donelson", "Small", 3],
    [55, "Men's Wearhouse", "Large", 2],
    [56, "Scripps Health", "Large", 2],
    [57, "Marriott International", "Large", 3],
    [58, "Perkins Coie", "Small", 7],
    [59, "Ernst & Young", "Large", 6],
    [60, "American Express", "Large", 4],
    [61, "Nordstrom", "Large", 6],
    [62, "Build-A-Bear Workshop", "Midsized", 0],
    [63, "General Mills", "Large", 1],
    [64, "TDIndustries", "Small", 9],
    [65, "Atlantic Health", "Midsized", -2],
    [66, "QuikTrip", "Large", 3],
    [67, "Deloitte", "Large", 7],
    [68, "Genentech", "Large", 1],
    [69, "Umpqua Bank", "Small", 5],
    [70, "Teach For America", "Small", 14],
    [71, "Mayo Clinic", "Large", 3],
    [72, "EOG Resources", "Small", 13],
    [73, "Starbucks", "Large", 3],
    [74, "Rackspace Hosting", "Midsized", 37],
    [75, "FactSet Research Systems", "Small", 22],
    [76, "Microsoft", "Large", -4],
    [77, "Aflac", "Midsized", -4],
    [78, "Publix Super Markets", "Large", 1],
    [79, "Mattel", "Midsized", -4],
    [80, "Stryker", "Large", 24],
    [81, "SRC", "Small", 7],
    [82, "Hasbro", "Midsized", 3],
    [83, "Bright Horizons Family Solutions", "Large", 5],
    [84, "Booz Allen Hamilton", "Large", 7],
    [85, "Four Seasons Hotels & Resorts", "Large", 6],
    [86, "Hitachi Data Systems", "Small", 7],
    [87, "The Everett Clinic", "Small", 4],
    [88, "OhioHealth", "Large", 4],
    [89, "Morningstar", "Small", 8],
    [90, "Cisco", "Large", 7],
    [91, "CarMax", "Large", 16],
    [92, "Accenture", "Large", 9],
    [93, "GoDaddy.com", "Midsized", 25],
    [94, "KPMG", "Large", 5],
    [95, "Navy Federal Credit Union", "Midsized", 8],
    [96, "Meridian Health", "Midsized", 27],
    [97, "Schweitzer Engineering Labs", "Small", 27],
    [98, "Capital One", "Large", 7],
    [99, "Darden Restaurants", "Large", 12],
    [100, "Intercontinental Hotels Group", "Large", -2]
]

# Criando o DataFrame
df = pd.DataFrame(data, columns=['Rank', 'Company', 'Size', 'Job_Growth'])

# a. Construa uma tabulação cruzada com Taxa de crescimento (%) como variável linha e Tamanho como variável coluna.
# Use classes começando em -10 e terminando em 70 em gradações de 10 para a Taxa de crescimento (%).

# Definindo os intervalos para a taxa de crescimento
bins = range(-10, 80, 10)
labels = [f"{i} a {i+9}" for i in range(-10, 70, 10)]

# Criando uma nova coluna com as classes de crescimento
df['Growth_Class'] = pd.cut(df['Job_Growth'], bins=bins, labels=labels, right=True)

# Criando a tabulação cruzada
cross_tab = pd.crosstab(df['Growth_Class'], df['Size'], margins=True, margins_name='Total')
print("a. Tabulação cruzada com Taxa de crescimento (%) como variável linha e Tamanho como variável coluna:")
print(cross_tab)
print("\n")

# b. Mostre a distribuição de frequência para a taxa de crescimento dos funcionários no trabalho (%)
# e a distribuição de frequência para Tamanho.

# Distribuição de frequência para taxa de crescimento
growth_freq = pd.DataFrame(df['Growth_Class'].value_counts()).reset_index()
growth_freq.columns = ['Taxa de Crescimento (%)', 'Frequência']
growth_freq['Frequência Relativa (%)'] = (growth_freq['Frequência'] / len(df) * 100).round(1)
growth_freq['Frequência Acumulada'] = growth_freq['Frequência'].cumsum()
growth_freq['Frequência Relativa Acumulada (%)'] = (growth_freq['Frequência Acumulada'] / len(df) * 100).round(1)
growth_freq = growth_freq.sort_values('Taxa de Crescimento (%)')

print("b.1 Distribuição de frequência para taxa de crescimento:")
print(growth_freq)
print("\n")

# Distribuição de frequência para Tamanho
size_freq = pd.DataFrame(df['Size'].value_counts()).reset_index()
size_freq.columns = ['Tamanho', 'Frequência']
size_freq['Frequência Relativa (%)'] = (size_freq['Frequência'] / len(df) * 100).round(1)
size_freq['Frequência Acumulada'] = size_freq['Frequência'].cumsum()
size_freq['Frequência Relativa Acumulada (%)'] = (size_freq['Frequência Acumulada'] / len(df) * 100).round(1)

print("b.2 Distribuição de frequência para Tamanho:")
print(size_freq)
print("\n")

# c. Usando a tabulação cruzada construída na parte (a), desenvolva uma tabulação cruzada mostrando as porcentagens em coluna.

# Tabulação cruzada com porcentagens em coluna
cross_tab_col_pct = pd.crosstab(df['Growth_Class'], df['Size'], normalize='columns', margins=True, margins_name='Total')
cross_tab_col_pct = cross_tab_col_pct.round(4) * 100

print("c. Tabulação cruzada mostrando as porcentagens em coluna:")
print(cross_tab_col_pct)
print("\n")

# d. Usando a tabulação cruzada construída na parte (a), desenvolva uma tabulação cruzada mostrando as porcentagens em linhas.

# Tabulação cruzada com porcentagens em linha
cross_tab_row_pct = pd.crosstab(df['Growth_Class'], df['Size'], normalize='index', margins=True, margins_name='Total')
cross_tab_row_pct = cross_tab_row_pct.round(4) * 100

print("d. Tabulação cruzada mostrando as porcentagens em linhas:")
print(cross_tab_row_pct)
print("\n")

# e. Comente a relação entre o crescimento percentual no emprego para os funcionários em tempo integral e o tamanho da empresa.

# Estatísticas descritivas por tamanho de empresa
growth_by_size = df.groupby('Size')['Job_Growth'].agg(['mean', 'median', 'min', 'max', 'std']).round(2)

print("e. Estatísticas descritivas para crescimento percentual por tamanho de empresa:")
print(growth_by_size)
print("\n")

# Visualização da relação entre crescimento e tamanho da empresa
plt.figure(figsize=(10, 6))
boxplot = sns.boxplot(x='Size', y='Job_Growth', data=df)
plt.title('Relação entre Crescimento Percentual e Tamanho da Empresa')
plt.xlabel('Tamanho da Empresa')
plt.ylabel('Taxa de Crescimento (%)')
plt.show()

# Gráfico de barras para média de crescimento por tamanho
plt.figure(figsize=(10, 6))
mean_growth = df.groupby('Size')['Job_Growth'].mean().sort_values()
barplot = sns.barplot(x=mean_growth.index, y=mean_growth.values)
plt.title('Média de Crescimento por Tamanho da Empresa')
plt.xlabel('Tamanho da Empresa')
plt.ylabel('Média da Taxa de Crescimento (%)')
plt.show()

## Cap 3 - 12