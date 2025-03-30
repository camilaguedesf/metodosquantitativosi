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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import locale
from datetime import datetime

# Tentar configurar locale para português (opcional, pode não funcionar em todos os sistemas)
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except:
    pass  # Se não funcionar, usaremos método alternativo

# Criando o DataFrame com os dados da tabela
data = {
    'Data_str': [
        '22 de setembro de 2011', '22 de setembro de 2011', '29 de setembro de 2011',
        '6 de outubro de 2011', '13 de outubro de 2011', '20 de outubro de 2011',
        '27 de outubro de 2011', '3 de novembro de 2011', '10 de novembro de 2011',
        '17 de novembro de 2011', '8 de dezembro de 2011', '12 de janeiro de 2012',
        '19 de janeiro de 2012', '26 de janeiro de 2012', '2 de fevereiro de 2012',
        '9 de fevereiro de 2012', '16 de fevereiro de 2012', '23 de fevereiro de 2012',
        '8 de março de 2012', '29 de março de 2012', '5 de abril de 2012'
    ],
    'Telespectadores': [
        14.1, 14.7, 14.6, 13.6, 13.6, 14.9, 14.5, 16.0, 15.9, 15.1, 14.0, 16.1,
        15.8, 16.1, 16.5, 16.2, 15.7, 16.2, 15.0, 14.0, 13.3
    ]
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Função para converter as datas manualmente
def convert_pt_date(date_str):
    day, month_str, year = date_str.split(' de ')
    
    month_dict = {
        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4,
        'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
        'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
    }
    
    month = month_dict[month_str]
    return pd.Timestamp(year=int(year), month=month, day=int(day))

# Aplicando a função de conversão
df['Data'] = df['Data_str'].apply(convert_pt_date)

# Removendo a coluna com strings originais
df = df.drop('Data_str', axis=1)

# Ordenando o DataFrame por data
df = df.sort_values('Data')

# a. Calcule o número mínimo e máximo de telespectadores
min_viewers = df['Telespectadores'].min()
max_viewers = df['Telespectadores'].max()
min_date = df.loc[df['Telespectadores'] == min_viewers, 'Data'].iloc[0]
max_date = df.loc[df['Telespectadores'] == max_viewers, 'Data'].iloc[0]

print("a. Número mínimo e máximo de telespectadores:")
print(f"   Mínimo: {min_viewers} milhões (em {min_date.strftime('%d/%m/%Y')})")
print(f"   Máximo: {max_viewers} milhões (em {max_date.strftime('%d/%m/%Y')})")
print("\n")

# b. Calcule a média, a mediana e a moda
mean_viewers = df['Telespectadores'].mean()
median_viewers = df['Telespectadores'].median()
mode_viewers = df['Telespectadores'].mode()[0]  # Pega a primeira moda (pode haver múltiplas)

print("b. Média, mediana e moda dos telespectadores:")
print(f"   Média: {mean_viewers:.2f} milhões")
print(f"   Mediana: {median_viewers:.2f} milhões")
print(f"   Moda: {mode_viewers:.1f} milhões")
print("\n")

# c. Calcule o primeiro e o terceiro quartis
q1 = df['Telespectadores'].quantile(0.25)
q3 = df['Telespectadores'].quantile(0.75)

print("c. Primeiro e terceiro quartis:")
print(f"   Primeiro quartil (Q1): {q1:.2f} milhões")
print(f"   Terceiro quartil (Q3): {q3:.2f} milhões")
print("\n")

# d. Análise se a audiência aumentou ou diminuiu ao longo da temporada 2011-2012
# Criando uma coluna numérica para representar o tempo (dias desde o primeiro episódio)
df['Dias'] = (df['Data'] - df['Data'].min()).dt.days

# Calculando a correlação entre dias e número de telespectadores
correlation = df['Dias'].corr(df['Telespectadores'])

# Linha de tendência (regressão linear)
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df['Dias'], df['Telespectadores']
)

print("d. Análise da tendência de audiência ao longo da temporada 2011-2012:")
print(f"   Coeficiente de correlação: {correlation:.4f}")
print(f"   Inclinação da linha de tendência: {slope:.4f} milhões por dia")
if slope > 0:
    print("   A audiência teve uma tendência de AUMENTO ao longo da temporada.")
else:
    print("   A audiência teve uma tendência de DIMINUIÇÃO ao longo da temporada.")

if abs(correlation) < 0.3:
    print("   A correlação é fraca, indicando que a tendência não é muito significativa.")
elif abs(correlation) < 0.7:
    print("   A correlação é moderada.")
else:
    print("   A correlação é forte, indicando uma tendência clara.")

# Primeira metade vs. segunda metade da temporada
half_point = len(df) // 2
first_half_avg = df.iloc[:half_point]['Telespectadores'].mean()
second_half_avg = df.iloc[half_point:]['Telespectadores'].mean()

print(f"   Média da primeira metade da temporada: {first_half_avg:.2f} milhões")
print(f"   Média da segunda metade da temporada: {second_half_avg:.2f} milhões")

if second_half_avg > first_half_avg:
    print(f"   Aumento de {second_half_avg - first_half_avg:.2f} milhões na segunda metade.")
else:
    print(f"   Diminuição de {first_half_avg - second_half_avg:.2f} milhões na segunda metade.")

# Visualização da tendência
plt.figure(figsize=(12, 6))
plt.plot(df['Data'], df['Telespectadores'], marker='o', linestyle='-', color='b')
plt.title('Audiência de The Big Bang Theory - Temporada 2011-2012')
plt.xlabel('Data de Transmissão')
plt.ylabel('Telespectadores (milhões)')
plt.grid(True, linestyle='--', alpha=0.7)

# Adicionando linha de tendência
plt.plot(df['Data'], intercept + slope * df['Dias'], 'r--', 
         label=f'Tendência (inclinação={slope:.4f})')

plt.legend()
plt.tight_layout()
plt.show()

# Boxplot para análise estatística
plt.figure(figsize=(8, 6))
plt.boxplot(df['Telespectadores'], vert=False)
plt.title('Distribuição da Audiência')
plt.xlabel('Telespectadores (milhões)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

## Cap 3 - 29

import numpy as np
import matplotlib.pyplot as plt

# Dados de Pomona
pomona_dados = np.array([28, 42, 58, 48, 45, 55, 60, 49, 50])

# a) Calcular a amplitude e a amplitude interquartil
pomona_min = np.min(pomona_dados)
pomona_max = np.max(pomona_dados)
pomona_amplitude = pomona_max - pomona_min

# Amplitude interquartil (IQR)
pomona_q1 = np.percentile(pomona_dados, 25)
pomona_q3 = np.percentile(pomona_dados, 75)
pomona_iqr = pomona_q3 - pomona_q1

# b) Calcular a variância amostral e o desvio padrão da amostra
pomona_variancia = np.var(pomona_dados, ddof=1)  # ddof=1 para variância amostral
pomona_desvio_padrao = np.std(pomona_dados, ddof=1)

# c) Comparações entre Pomona e Anaheim
# Dados de Anaheim
anaheim_media = 48.5
anaheim_variancia = 136
anaheim_desvio_padrao = 11.66

# Calcular a média de Pomona para comparação
pomona_media = np.mean(pomona_dados)

# Função para visualizar as comparações
def visualizar_comparacoes():
    # Criar um gráfico de barras para comparar as estatísticas
    labels = ['Média', 'Variância', 'Desvio Padrão']
    pomona_stats = [pomona_media, pomona_variancia, pomona_desvio_padrao]
    anaheim_stats = [anaheim_media, anaheim_variancia, anaheim_desvio_padrao]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pomona_stats, width, label='Pomona')
    rects2 = ax.bar(x + width/2, anaheim_stats, width, label='Anaheim')
    
    ax.set_ylabel('Valores')
    ax.set_title('Comparação das Estatísticas de Qualidade do Ar')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Adicionar os valores nas barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    # Também podemos realizar um boxplot para comparação da distribuição
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.boxplot([pomona_dados], labels=['Pomona'])
    ax2.set_title('Boxplot da Qualidade do Ar em Pomona')
    ax2.set_ylabel('Índice de Qualidade do Ar')
    
    plt.show()

# Exibir os resultados
print("\nResultados para Pomona:")
print(f"a) Amplitude: {pomona_amplitude}")
print(f"   Amplitude Interquartil (IQR): {pomona_iqr:.2f}")
print(f"b) Variância Amostral: {pomona_variancia:.2f}")
print(f"   Desvio Padrão Amostral: {pomona_desvio_padrao:.2f}")

print("\nComparação entre Pomona e Anaheim:")
print(f"Média: Pomona = {pomona_media:.2f}, Anaheim = {anaheim_media}")
print(f"Variância: Pomona = {pomona_variancia:.2f}, Anaheim = {anaheim_variancia}")
print(f"Desvio Padrão: Pomona = {pomona_desvio_padrao:.2f}, Anaheim = {anaheim_desvio_padrao}")

print("\nInterpretação dos resultados:")
if pomona_media > anaheim_media:
    print(f"- A qualidade do ar em Pomona tem um índice médio mais alto ({pomona_media:.2f}) do que em Anaheim ({anaheim_media}).")
elif pomona_media < anaheim_media:
    print(f"- A qualidade do ar em Pomona tem um índice médio mais baixo ({pomona_media:.2f}) do que em Anaheim ({anaheim_media}).")
else:
    print(f"- A qualidade do ar em Pomona e Anaheim têm índices médios semelhantes ({pomona_media:.2f}).")

if pomona_variancia > anaheim_variancia:
    print(f"- Os dados de Pomona apresentam maior variabilidade (variância = {pomona_variancia:.2f}) do que os de Anaheim (variância = {anaheim_variancia}).")
elif pomona_variancia < anaheim_variancia:
    print(f"- Os dados de Pomona apresentam menor variabilidade (variância = {pomona_variancia:.2f}) do que os de Anaheim (variância = {anaheim_variancia}).")
else:
    print(f"- Os dados de Pomona e Anaheim apresentam variabilidade semelhante.")

if pomona_desvio_padrao > anaheim_desvio_padrao:
    print(f"- Os dados de Pomona apresentam maior dispersão (desvio padrão = {pomona_desvio_padrao:.2f}) do que os de Anaheim (desvio padrão = {anaheim_desvio_padrao}).")
elif pomona_desvio_padrao < anaheim_desvio_padrao:
    print(f"- Os dados de Pomona apresentam menor dispersão (desvio padrão = {pomona_desvio_padrao:.2f}) do que os de Anaheim (desvio padrão = {anaheim_desvio_padrao}).")
else:
    print(f"- Os dados de Pomona e Anaheim apresentam dispersão semelhante.")

## Cap 3 - 34

import numpy as np
import matplotlib.pyplot as plt

# Dados dos tempos dos corredores (em minutos)
tempos_quarto_milha = np.array([0.92, 0.98, 1.04, 0.90, 0.99])
tempos_milha = np.array([4.52, 4.35, 4.60, 4.70, 4.50])

# Calcular estatísticas para as duas amostras
# Para os tempos de um quarto de milha
media_quarto = np.mean(tempos_quarto_milha)
dp_quarto = np.std(tempos_quarto_milha, ddof=1)  # desvio padrão amostral (ddof=1)
cv_quarto = (dp_quarto / media_quarto) * 100  # coeficiente de variação em percentual

# Para os tempos de uma milha
media_milha = np.mean(tempos_milha)
dp_milha = np.std(tempos_milha, ddof=1)  # desvio padrão amostral (ddof=1)
cv_milha = (dp_milha / media_milha) * 100  # coeficiente de variação em percentual

# Função para criar visualizações
def visualizar_resultados():
    # Criar um gráfico para comparar os coeficientes de variação
    categorias = ['Quarto de Milha', 'Milha Completa']
    coeficientes = [cv_quarto, cv_milha]
    
    plt.figure(figsize=(10, 6))
    plt.bar(categorias, coeficientes, color=['blue', 'green'])
    plt.title('Coeficiente de Variação por Tipo de Prova')
    plt.ylabel('Coeficiente de Variação (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(coeficientes):
        plt.text(i, v + 0.3, f'{v:.2f}%', ha='center')
    
    # Gráficos de dispersão para visualizar a variabilidade dos tempos
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(range(1, len(tempos_quarto_milha) + 1), tempos_quarto_milha, color='blue')
    plt.axhline(y=media_quarto, color='red', linestyle='--', label=f'Média: {media_quarto:.2f}')
    plt.fill_between(range(1, len(tempos_quarto_milha) + 1), 
                     media_quarto - dp_quarto, 
                     media_quarto + dp_quarto, 
                     alpha=0.2, color='red', label=f'DP: ±{dp_quarto:.3f}')
    plt.title('Tempos da Prova de Quarto de Milha')
    plt.xlabel('Corredor')
    plt.ylabel('Tempo (minutos)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(1, len(tempos_milha) + 1), tempos_milha, color='green')
    plt.axhline(y=media_milha, color='red', linestyle='--', label=f'Média: {media_milha:.2f}')
    plt.fill_between(range(1, len(tempos_milha) + 1), 
                     media_milha - dp_milha, 
                     media_milha + dp_milha, 
                     alpha=0.2, color='red', label=f'DP: ±{dp_milha:.3f}')
    plt.title('Tempos da Prova de Uma Milha')
    plt.xlabel('Corredor')
    plt.ylabel('Tempo (minutos)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Exibir resultados
print("\nAnálise dos Tempos de Corrida - Equipe Universitária")
print("-" * 50)

print("\nProva de Um Quarto de Milha:")
print(f"Tempos (minutos): {', '.join([str(t) for t in tempos_quarto_milha])}")
print(f"Média: {media_quarto:.3f} minutos")
print(f"Desvio Padrão: {dp_quarto:.3f} minutos")
print(f"Coeficiente de Variação: {cv_quarto:.2f}%")

print("\nProva de Uma Milha:")
print(f"Tempos (minutos): {', '.join([str(t) for t in tempos_milha])}")
print(f"Média: {media_milha:.3f} minutos")
print(f"Desvio Padrão: {dp_milha:.3f} minutos")
print(f"Coeficiente de Variação: {cv_milha:.2f}%")

print("\nAnálise Comparativa:")
if cv_quarto < cv_milha:
    print(f"O coeficiente de variação da prova de um quarto de milha ({cv_quarto:.2f}%) é menor que o da prova de uma milha ({cv_milha:.2f}%).")
    print("Isso indica que os tempos na prova de um quarto de milha são mais consistentes, o que contradiz a declaração do treinador.")
    print("A declaração do técnico não se sustenta com base nessa análise estatística.")
elif cv_quarto > cv_milha:
    print(f"O coeficiente de variação da prova de um quarto de milha ({cv_quarto:.2f}%) é maior que o da prova de uma milha ({cv_milha:.2f}%).")
    print("Isso indica que os tempos na prova de uma milha são mais consistentes, o que confirma a declaração do treinador.")
    print("A declaração do técnico é suportada pela análise estatística.")
else:
    print(f"Os coeficientes de variação são iguais ({cv_quarto:.2f}%).")
    print("Não há diferença de consistência entre as duas provas com base nessa análise.")

print("\nInterpretação do Coeficiente de Variação:")
print("O coeficiente de variação (CV) expressa a variabilidade dos dados em relação à média.")
print("Quanto menor o CV, maior a consistência dos dados (menor variabilidade relativa).")
print("Um CV menor indica tempos mais homogêneos entre os corredores.")

## Cap 3 - 36

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Parâmetros da amostra
media = 500
desvio_padrao = 100

# Valores para calcular os escores-z
valores = np.array([520, 650, 500, 450, 280])

# Calcular os escores-z para cada valor
escores_z = (valores - media) / desvio_padrao

# Criar uma função para visualizar os resultados
def visualizar_distribuicao():
    # Criar um gráfico da distribuição normal com os pontos marcados
    x = np.linspace(media - 4*desvio_padrao, media + 4*desvio_padrao, 1000)
    y = stats.norm.pdf(x, media, desvio_padrao)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Distribuição Normal')
    
    # Marcar a média
    plt.axvline(x=media, color='r', linestyle='--', label=f'Média = {media}')
    
    # Marcar cada valor e seu escore-z
    colors = ['green', 'purple', 'orange', 'brown', 'magenta']
    for i, (valor, z, color) in enumerate(zip(valores, escores_z, colors)):
        plt.axvline(x=valor, color=color, linestyle='-', 
                   label=f'Valor = {valor}, z = {z:.2f}')
        
    plt.title('Distribuição Normal com Escores-z')
    plt.xlabel('Valores')
    plt.ylabel('Densidade de Probabilidade')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Gráfico de barras dos escores-z
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(valores)), escores_z, color=colors)
    
    # Adicionar rótulos às barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height < 0:
            va = 'top'
            offset = -0.3
        else:
            va = 'bottom'
            offset = 0.3
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'z = {escores_z[i]:.2f}\n({valores[i]})',
                ha='center', va=va)
    
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='z = +1')
    plt.axhline(y=-1, color='green', linestyle='--', alpha=0.7, label='z = -1')
    plt.axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='z = +2')
    plt.axhline(y=-2, color='orange', linestyle='--', alpha=0.7, label='z = -2')
    
    plt.title('Escores-z para os Valores Fornecidos')
    plt.xlabel('Valores da Amostra')
    plt.ylabel('Escore-z')
    plt.xticks(range(len(valores)), [str(v) for v in valores])
    plt.grid(True, axis='y')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Exibir os resultados
print("\nCálculo de Escores-z com Média = 500 e Desvio Padrão = 100")
print("-" * 60)
print(f"Fórmula do Escore-z: z = (x - μ) / σ, onde:")
print(f"  - x é o valor a ser padronizado")
print(f"  - μ é a média da distribuição (500)")
print(f"  - σ é o desvio padrão da distribuição (100)")
print("-" * 60)

# Criar uma tabela com os resultados
print("\nResultados:")
print(f"{'Valor':<10} | {'Cálculo':<20} | {'Escore-z':<10} | {'Interpretação':<40}")
print("-" * 85)

for valor, z in zip(valores, escores_z):
    calculo = f"({valor} - 500) / 100"
    
    # Interpretação do escore-z
    if abs(z) < 1:
        interpretacao = "Próximo à média (menos de 1 DP de distância)"
    elif abs(z) < 2:
        interpretacao = "Moderadamente distante da média (entre 1 e 2 DP)"
    elif abs(z) < 3:
        interpretacao = "Consideravelmente distante da média (entre 2 e 3 DP)"
    else:
        interpretacao = "Extremamente distante da média (mais de 3 DP)"
        
    if z > 0:
        interpretacao += ", acima da média"
    elif z < 0:
        interpretacao += ", abaixo da média"
    
    print(f"{valor:<10} | {calculo:<20} | {z:.2f}{' ':<6} | {interpretacao:<40}")

# Informações adicionais sobre escores-z
print("\nObservações:")
print("- O escore-z representa quantos desvios padrão um valor está acima ou abaixo da média")
print("- Valores positivos estão acima da média, valores negativos estão abaixo")
print("- Na distribuição normal:")
print("  * Aproximadamente 68% dos valores têm escores-z entre -1 e +1")
print("  * Aproximadamente 95% dos valores têm escores-z entre -2 e +2")
print("  * Aproximadamente 99.7% dos valores têm escores-z entre -3 e +3")

## Cap 3 - 44

import numpy as np
import scipy.stats as stats

# Dados fornecidos na tabela
pontos_vencedor = [90, 85, 75, 78, 71, 65, 72, 76, 77, 82]
pontos_perdedor = [66, 66, 70, 57, 63, 62, 66, 70, 67, 56]
margem_pontos = [24, 19, 5, 21, 8, 3, 6, 6, 10, 26]

# a. Calcule a média e o desvio padrão para os pontos marcados pelo time vencedor
media_pontos_vencedor = np.mean(pontos_vencedor)
desvio_padrao_pontos_vencedor = np.std(pontos_vencedor, ddof=1)  # ddof=1 para desvio padrão amostral

print(f"a. Pontos do time vencedor:")
print(f"   Média: {media_pontos_vencedor:.2f}")
print(f"   Desvio padrão: {desvio_padrao_pontos_vencedor:.2f}")

# b. Suponha distribuição normal, estimando a porcentagem de jogos onde o vencedor marca mais de 90 pontos
# Usando a distribuição normal com a média e desvio padrão calculados
z_score = (90 - media_pontos_vencedor) / desvio_padrao_pontos_vencedor
probabilidade = 1 - stats.norm.cdf(z_score)
porcentagem = probabilidade * 100

print(f"\nb. Porcentagem de jogos em que o time vencedor marca mais de 90 pontos:")
print(f"   {porcentagem:.2f}%")

# c. Calcule a média e o desvio padrão para a margem de pontos do time vitorioso
media_margem = np.mean(margem_pontos)
desvio_padrao_margem = np.std(margem_pontos, ddof=1)

print(f"\nc. Margem de pontos do time vitorioso:")
print(f"   Média: {media_margem:.2f}")
print(f"   Desvio padrão: {desvio_padrao_margem:.2f}")

## Cap 3 - 48

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dados fornecidos
dados = [5, 15, 18, 10, 8, 12, 16, 10, 6]

# Ordenando os dados
dados_ordenados = sorted(dados)
print("Dados ordenados:", dados_ordenados)

# Cálculo do resumo de cinco números
minimo = min(dados)
maximo = max(dados)
mediana = np.median(dados)
q1 = np.percentile(dados, 25)
q3 = np.percentile(dados, 75)
iqr = q3 - q1

# Exibindo o resumo de cinco números
print("\nResumo de cinco números (Five-number summary):")
print(f"Mínimo: {minimo}")
print(f"Q1 (Primeiro quartil): {q1}")
print(f"Mediana: {mediana}")
print(f"Q3 (Terceiro quartil): {q3}")
print(f"Máximo: {maximo}")
print(f"IQR (Intervalo interquartil): {iqr}")

# Verificando outliers
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr
print(f"\nLimite inferior para outliers: {limite_inferior}")
print(f"Limite superior para outliers: {limite_superior}")

outliers = [x for x in dados if x < limite_inferior or x > limite_superior]
print(f"Outliers (se houver): {outliers}")

# Criando o boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=dados)
plt.title('Boxplot dos dados')
plt.xlabel('Valores')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

## Cap 3 - 56

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Dados fornecidos
x = [6, 11, 15, 21, 27]
y = [6, 9, 6, 17, 12]

# a. Construir um diagrama de dispersão
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', s=100)
plt.title('Diagrama de Dispersão')
plt.xlabel('Variável x')
plt.ylabel('Variável y')
plt.grid(True, linestyle='--', alpha=0.7)

# Adicionar labels aos pontos
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'({xi}, {yi})', (xi, yi), xytext=(5, 5), textcoords='offset points')

# Adicionar linha de tendência
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", alpha=0.8, label=f'Linha de tendência: y = {z[0]:.4f}x + {z[1]:.4f}')
plt.legend()
plt.tight_layout()
plt.show()

# c. Calcular a covariância amostral
covariancia = np.cov(x, y, ddof=1)[0, 1]  # ddof=1 para covariância amostral
print(f"c. Covariância amostral: {covariancia:.4f}")

# d. Calcular e interpretar o coeficiente de correlação amostral
correlacao, p_valor = stats.pearsonr(x, y)
print(f"d. Coeficiente de correlação amostral (r): {correlacao:.4f}")
print(f"   p-valor: {p_valor:.4f}")

# Interpretação do coeficiente de correlação
if abs(correlacao) < 0.3:
    interpretacao = "fraca"
elif abs(correlacao) < 0.7:
    interpretacao = "moderada"
else:
    interpretacao = "forte"

if correlacao > 0:
    direcao = "positiva"
else:
    direcao = "negativa"

print(f"   Interpretação: Existe uma correlação {interpretacao} e {direcao} entre as variáveis x e y.")

# Estatísticas descritivas adicionais
print("\nEstatísticas descritivas:")
print(f"Média de x: {np.mean(x):.4f}")
print(f"Média de y: {np.mean(y):.4f}")
print(f"Desvio padrão de x: {np.std(x, ddof=1):.4f}")
print(f"Desvio padrão de y: {np.std(y, ddof=1):.4f}")

# b. Análise do diagrama de dispersão
print("\nb. O diagrama de dispersão indica:")
if correlacao > 0.5:
    print("- Uma tendência de crescimento de y conforme x aumenta")
elif correlacao < -0.5:
    print("- Uma tendência de decrescimento de y conforme x aumenta")
else:
    print("- Uma relação não muito clara entre as variáveis")

if abs(correlacao) > 0.7:
    print("- Uma forte associação linear entre as variáveis")
elif abs(correlacao) > 0.3:
    print("- Uma associação linear moderada entre as variáveis")
else:
    print("- Uma associação linear fraca entre as variáveis")

## Cap 3 - 59

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Dados fornecidos
percentagem_detectores = [0.50, 0.67, 0.74, 0.76, 0.77, 0.82, 0.81, 0.85, 0.86, 0.88, 0.90, 0.92, 0.95, 0.96, 0.97, 0.96, 0.96]
mortes_por_milhao = [22.9, 20.8, 17.3, 20.5, 19.4, 18.9, 17.6, 16.2, 13.6, 14.4, 14.3, 13.0, 10.9, 10.9, 10.2, 8.4, 8.1]

# Convertendo para percentuais (0-100) para melhor visualização
percentagem_detectores_viz = [p * 100 for p in percentagem_detectores]

# Criar DataFrame
df = pd.DataFrame({
    'Percentual_detectores': percentagem_detectores,
    'Percentual_detectores_viz': percentagem_detectores_viz,
    'Mortes_por_milhao': mortes_por_milhao
})

# a. Relação entre uso de detectores e mortes
correlacao, p_valor = stats.pearsonr(percentagem_detectores, mortes_por_milhao)
print(f"a. Relação entre uso de detectores de fumaça e mortes:")
print(f"   Coeficiente de correlação: {correlacao:.4f}")
print(f"   p-valor: {p_valor:.8f}")

if correlacao < 0:
    print("   Existe uma relação NEGATIVA entre o uso de detectores de fumaça e mortes por incêndio.")
    print("   Isso significa que quanto maior a porcentagem de residências com detectores, menor é a taxa de mortalidade.")
else:
    print("   Existe uma relação POSITIVA entre o uso de detectores de fumaça e mortes por incêndio.")
    print("   Isso significa que quanto maior a porcentagem de residências com detectores, maior é a taxa de mortalidade.")

# b. Cálculo do coeficiente de correlação
print(f"\nb. Coeficiente de correlação: {correlacao:.4f}")
if abs(correlacao) < 0.3:
    forca = "fraca"
elif abs(correlacao) < 0.7:
    forca = "moderada"
else:
    forca = "forte"
    
print(f"   Existe uma correlação {forca} e negativa entre o uso de detectores de fumaça e mortes por incêndios.")
print(f"   Isso sugere que o aumento na adoção de detectores de fumaça está associado a uma diminuição nas mortes por incêndio.")
print(f"   O valor p de {p_valor:.8f} indica que essa correlação é estatisticamente significativa.")

# c. Gráfico de dispersão
plt.figure(figsize=(10, 6))
sns.regplot(x='Percentual_detectores_viz', y='Mortes_por_milhao', data=df, 
            scatter_kws={'s': 80, 'alpha': 0.7}, line_kws={'color': 'red'})

plt.title('Relação entre Percentual de Residências com Detectores de Fumaça e Mortes por Incêndio')
plt.xlabel('Percentual de Residências com Detectores de Fumaça (%)')
plt.ylabel('Mortes por Incêndio por Milhão de Habitantes')
plt.grid(True, linestyle='--', alpha=0.7)

# Adicionar equação da linha de tendência
slope, intercept, r_value, p_value, std_err = stats.linregress(percentagem_detectores_viz, mortes_por_milhao)
equation = f'y = {slope:.4f}x + {intercept:.4f}'
plt.annotate(f'Equação: {equation}\nR² = {r_value**2:.4f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()