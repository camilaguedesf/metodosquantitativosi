#3ª lista de exercicios
#Exercícios dos capítulos 7 e 8 do livro Estatística Aplicada a Administração e Economia.
#Cap. 7:  15, 18, 19, 22, 23, 28, 31
#Cap. 8: 2, 7, 11, 13, 19, 22, 23

import numpy as np
#Cap.7:15
# Dados da amostra
classificacoes = [57, 61, 86, 74, 72, 73, 20, 57, 80, 79, 83, 74]

# a. Estimativa pontual da média (média amostral)
media_amostral = np.mean(classificacoes)

# b. Estimativa pontual do desvio padrão (desvio padrão amostral)
# Usando n-1 no denominador (estimativa não enviesada para a população)
desvio_padrao_amostral = np.std(classificacoes, ddof=1)

print(f"a. Estimativa pontual da classificação média: {media_amostral:.2f}")
print(f"b. Estimativa pontual do desvio padrão: {desvio_padrao_amostral:.2f}")
#Cap:7:18
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Parâmetros da população
media_populacional = 200
desvio_padrao_populacional = 50
tamanho_amostra = 100

# a. Cálculo do valor esperado de x̄
valor_esperado_x_barra = media_populacional
print(f"a. Valor esperado de x̄ = {valor_esperado_x_barra}")

# b. Cálculo do desvio padrão de x̄ (erro padrão)
desvio_padrao_x_barra = desvio_padrao_populacional / np.sqrt(tamanho_amostra)
print(f"b. Desvio padrão de x̄ = {desvio_padrao_x_barra}")

# c. Simulação da distribuição amostral de x̄
num_simulacoes = 10000
medias_amostrais = []

for _ in range(num_simulacoes):
    # Gerando uma amostra de tamanho 100 da população
    amostra = np.random.normal(media_populacional, desvio_padrao_populacional, tamanho_amostra)
    # Calculando a média da amostra
    media_amostral = np.mean(amostra)
    medias_amostrais.append(media_amostral)

# Criação do histograma das médias amostrais
plt.figure(figsize=(10, 6))
plt.hist(medias_amostrais, bins=30, alpha=0.7, color='blue', edgecolor='black')

# Adicionando a curva normal teórica
x = np.linspace(min(medias_amostrais), max(medias_amostrais), 100)
y = stats.norm.pdf(x, media_populacional, desvio_padrao_x_barra) * num_simulacoes * (max(medias_amostrais) - min(medias_amostrais)) / 30
plt.plot(x, y, 'r-', linewidth=2)

plt.axvline(media_populacional, color='g', linestyle='--', linewidth=2, label=f'Média Populacional (μ = {media_populacional})')
plt.axvline(np.mean(medias_amostrais), color='m', linestyle=':', linewidth=2, label=f'Média das Médias Amostrais = {np.mean(medias_amostrais):.2f}')

plt.title('Distribuição Amostral das Médias (n = 100)')
plt.xlabel('Média Amostral (x̄)')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# d. Análise da distribuição amostral
print("\nc. Distribuição amostral de x̄:")
print("   A distribuição amostral de x̄ segue uma distribuição normal com:")
print(f"   - Média = {media_populacional}")
print(f"   - Desvio padrão = {desvio_padrao_x_barra}")

print("\nd. O que mostra a distribuição amostral de x̄:")
print("   A distribuição amostral de x̄ mostra como se comportam as médias de amostras")
print("   de tamanho 100 retiradas da população. Pelo Teorema do Limite Central,")
print("   essa distribuição é aproximadamente normal, independentemente da distribuição")
print("   da população original, principalmente porque o tamanho da amostra é grande (n = 100).")
print("   Esta distribuição é centrada na média populacional (μ = 200) e tem desvio")
print(f"   padrão reduzido (σ/√n = {desvio_padrao_x_barra}).")
print("   A distribuição amostral permite calcular probabilidades relacionadas à média amostral.")
plt.show()

#Cap. 7:19
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Parâmetros do problema
media_populacional = 200
desvio_padrao_populacional = 50
tamanho_amostra = 100

# Cálculo do erro padrão da média (desvio padrão da distribuição amostral)
erro_padrao = desvio_padrao_populacional / np.sqrt(tamanho_amostra)
print(f"Erro padrão da média: {erro_padrao}")

# a. Probabilidade de que a média amostral esteja dentro de ±5 da média populacional
# Padronizando os valores para a distribuição normal padrão
z_5_superior = (media_populacional + 5 - media_populacional) / erro_padrao
z_5_inferior = (media_populacional - 5 - media_populacional) / erro_padrao

# Calculando a probabilidade usando a função de distribuição acumulada (CDF)
prob_dentro_5 = stats.norm.cdf(z_5_superior) - stats.norm.cdf(z_5_inferior)
print(f"\na. Probabilidade de que a média amostral esteja dentro de ±5 da média populacional:")
print(f"   P({media_populacional-5} < x̄ < {media_populacional+5}) = {prob_dentro_5:.4f} = {prob_dentro_5*100:.2f}%")

# b. Probabilidade de que a média amostral esteja dentro de ±10 da média populacional
z_10_superior = (media_populacional + 10 - media_populacional) / erro_padrao
z_10_inferior = (media_populacional - 10 - media_populacional) / erro_padrao

prob_dentro_10 = stats.norm.cdf(z_10_superior) - stats.norm.cdf(z_10_inferior)
print(f"\nb. Probabilidade de que a média amostral esteja dentro de ±10 da média populacional:")
print(f"   P({media_populacional-10} < x̄ < {media_populacional+10}) = {prob_dentro_10:.4f} = {prob_dentro_10*100:.2f}%")

# Criando um gráfico para visualizar melhor
x = np.linspace(media_populacional - 4*erro_padrao, media_populacional + 4*erro_padrao, 1000)
y = stats.norm.pdf(x, media_populacional, erro_padrao)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.axvline(media_populacional, color='k', linestyle='--', alpha=0.5)

# Sombreando a região dentro de ±5
x_fill_5 = np.linspace(media_populacional-5, media_populacional+5, 100)
y_fill_5 = stats.norm.pdf(x_fill_5, media_populacional, erro_padrao)
plt.fill_between(x_fill_5, y_fill_5, color='red', alpha=0.3, label=f'±5 ({prob_dentro_5*100:.2f}%)')

# Sombreando a região dentro de ±10
x_fill_10 = np.linspace(media_populacional-10, media_populacional+10, 100)
y_fill_10 = stats.norm.pdf(x_fill_10, media_populacional, erro_padrao)
plt.fill_between(x_fill_10, y_fill_10, color='blue', alpha=0.2, label=f'±10 ({prob_dentro_10*100:.2f}%)')

plt.title('Distribuição Amostral da Média (n=100)')
plt.xlabel('Média Amostral (x̄)')
plt.ylabel('Densidade de Probabilidade')
plt.axvline(media_populacional-5, color='r', linestyle='--', alpha=0.7)
plt.axvline(media_populacional+5, color='r', linestyle='--', alpha=0.7)
plt.axvline(media_populacional-10, color='b', linestyle='--', alpha=0.7)
plt.axvline(media_populacional+10, color='b', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.legend()

# Método alternativo de cálculo usando a distribuição normal padrão diretamente
# Estes cálculos servem para verificar os resultados acima
print("\nVerificação usando valores de z:")
print(f"Para ±5: z = ±{5/erro_padrao:.4f}")
print(f"P(-{5/erro_padrao:.4f} < Z < {5/erro_padrao:.4f}) = {prob_dentro_5:.4f}")

print(f"Para ±10: z = ±{10/erro_padrao:.4f}")
print(f"P(-{10/erro_padrao:.4f} < Z < {10/erro_padrao:.4f}) = {prob_dentro_10:.4f}")
plt.show()

#Cap.7:22

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Vamos assumir que a população de gestores tem:
# - média populacional μ = 100 (por exemplo, pontuação em alguma avaliação)
# - desvio padrão populacional σ = 15

media_populacional = 100
desvio_padrao_populacional = 15

# Tamanhos de amostra a serem analisados
tamanho_amostra_1 = 60  # Parte a do problema
tamanho_amostra_2 = 120  # Parte b do problema
tamanhos_adicionais = [30, 240, 480]  # Para a parte c, demonstrar o efeito do aumento do tamanho amostral

# Número de simulações para cada tamanho de amostra
num_simulacoes = 10000

# a. Distribuição amostral de x̄ com n = 60
medias_amostrais_60 = []
for _ in range(num_simulacoes):
    amostra = np.random.normal(media_populacional, desvio_padrao_populacional, tamanho_amostra_1)
    media_amostral = np.mean(amostra)
    medias_amostrais_60.append(media_amostral)

# b. Distribuição amostral de x̄ com n = 120
medias_amostrais_120 = []
for _ in range(num_simulacoes):
    amostra = np.random.normal(media_populacional, desvio_padrao_populacional, tamanho_amostra_2)
    media_amostral = np.mean(amostra)
    medias_amostrais_120.append(media_amostral)
plt.show()
# c. Distribuições amostrais adicionais para demonstrar o efeito do aumento do tamanho amostral
medias_adicionais = {}
for n in tamanhos_adicionais:
    medias_adicionais[n] = []
    for _ in range(num_simulacoes):
        amostra = np.random.normal(media_populacional, desvio_padrao_populacional, n)
        media_amostral = np.mean(amostra)
        medias_adicionais[n].append(media_amostral)

# Calcular erro padrão teórico para cada tamanho de amostra
erro_padrao_60 = desvio_padrao_populacional / np.sqrt(tamanho_amostra_1)
erro_padrao_120 = desvio_padrao_populacional / np.sqrt(tamanho_amostra_2)

# Criar visualizações
plt.figure(figsize=(15, 10))

# Gráfico 1: Comparação das distribuições n=60 e n=120
plt.subplot(2, 1, 1)
plt.hist(medias_amostrais_60, bins=30, alpha=0.5, color='blue', label=f'n=60, σx̄={erro_padrao_60:.2f}')
plt.hist(medias_amostrais_120, bins=30, alpha=0.5, color='red', label=f'n=120, σx̄={erro_padrao_120:.2f}')
plt.axvline(media_populacional, color='black', linestyle='--', linewidth=1)
plt.title('Distribuições Amostrais das Médias para n=60 e n=120')
plt.xlabel('Média Amostral (x̄)')
plt.ylabel('Frequência')
plt.legend()
plt.grid(alpha=0.3)

# Gráfico 2: Demonstração do efeito do aumento do tamanho amostral
plt.subplot(2, 1, 2)

# Lista completa de tamanhos amostrais
todos_tamanhos = [tamanho_amostra_1, tamanho_amostra_2] + tamanhos_adicionais
todos_tamanhos.sort()  # Ordenar para melhor visualização

# Cores para cada tamanho amostral
cores = plt.cm.viridis(np.linspace(0, 1, len(todos_tamanhos)))

# Plotar densidade de probabilidade para cada tamanho amostral
for i, n in enumerate(todos_tamanhos):
    if n in tamanhos_adicionais:
        dados = medias_adicionais[n]
    elif n == tamanho_amostra_1:
        dados = medias_amostrais_60
    else:
        dados = medias_amostrais_120
    
    erro_padrao = desvio_padrao_populacional / np.sqrt(n)
    
    # Usar KDE (Kernel Density Estimation) para visualizar a densidade
    x = np.linspace(min(dados), max(dados), 1000)
    kernel = stats.gaussian_kde(dados)
    plt.plot(x, kernel(x), color=cores[i], label=f'n={n}, σx̄={erro_padrao:.2f}')

plt.axvline(media_populacional, color='black', linestyle='--', linewidth=1, label='Média Populacional (μ)')
plt.title('Efeito do Aumento do Tamanho Amostral na Distribuição Amostral')
plt.xlabel('Média Amostral (x̄)')
plt.ylabel('Densidade de Probabilidade')
plt.legend(loc='upper right', fontsize=8)
plt.grid(alpha=0.3)

plt.tight_layout()

# Tabela com estatísticas comparativas
estatisticas = {'Tamanho da Amostra (n)': [], 
                'Erro Padrão Teórico (σx̄)': [], 
                'Média das Médias Amostrais': [], 
                'Desvio Padrão Empírico das Médias': []}

for n in todos_tamanhos:
    if n in tamanhos_adicionais:
        dados = medias_adicionais[n]
    elif n == tamanho_amostra_1:
        dados = medias_amostrais_60
    else:
        dados = medias_amostrais_120
    
    erro_padrao_teorico = desvio_padrao_populacional / np.sqrt(n)
    media_empirica = np.mean(dados)
    desvio_padrao_empirico = np.std(dados)
    
    estatisticas['Tamanho da Amostra (n)'].append(n)
    estatisticas['Erro Padrão Teórico (σx̄)'].append(erro_padrao_teorico)
    estatisticas['Média das Médias Amostrais'].append(media_empirica)
    estatisticas['Desvio Padrão Empírico das Médias'].append(desvio_padrao_empirico)

# Criar dataframe para exibir as estatísticas
df_estatisticas = pd.DataFrame(estatisticas)

# Relatório escrito das observações
print("RELATÓRIO DE ANÁLISE DA DISTRIBUIÇÃO AMOSTRAL")
print("=============================================\n")
print("a. Distribuição amostral de x̄ com n = 60:")
print(f"   - Média teórica: {media_populacional}")
print(f"   - Erro padrão teórico: {erro_padrao_60:.4f}")
print(f"   - Média empírica obtida: {np.mean(medias_amostrais_60):.4f}")
print(f"   - Desvio padrão empírico: {np.std(medias_amostrais_60):.4f}")
print("\nb. Distribuição amostral de x̄ com n = 120:")
print(f"   - Média teórica: {media_populacional}")
print(f"   - Erro padrão teórico: {erro_padrao_120:.4f}")
print(f"   - Média empírica obtida: {np.mean(medias_amostrais_120):.4f}")
print(f"   - Desvio padrão empírico: {np.std(medias_amostrais_120):.4f}")
print("\nc. Declaração geral sobre o que acontece com a distribuição amostral quando o tamanho amostral aumenta:")
print("   À medida que o tamanho da amostra aumenta, observamos que:")
print("   1. A distribuição amostral da média se aproxima cada vez mais de uma distribuição normal,")
print("      independentemente da forma da distribuição da população original (Teorema do Limite Central).")
print("   2. A variabilidade da distribuição amostral diminui, ou seja, o erro padrão (σx̄) diminui")
print("      proporcionalmente à raiz quadrada do tamanho da amostra (σx̄ = σ/√n).")
print("   3. A média da distribuição amostral permanece igual à média populacional (μ),")
print("      independentemente do tamanho da amostra.")
print("\n   Esta generalização é lógica porque:")
print("   - Quanto maior a amostra, mais informação temos sobre a população, o que reduz a incerteza")
print("     na estimativa da média populacional.")
print("   - A redução do erro padrão é proporcional à raiz quadrada do tamanho da amostra (não linear),")
print("     o que significa que o ganho em precisão diminui à medida que o tamanho da amostra aumenta.")
print("   - Isso explica porque, em pesquisas estatísticas, há um ponto onde aumentar o tamanho da amostra")
print("     não traz ganhos significativos em precisão que justifiquem o custo adicional da coleta de dados.")

print("\nTabela comparativa de estatísticas para diferentes tamanhos amostrais:")
print(df_estatisticas.to_string(index=False))

plt.show()

#Cap.7: 23

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados do problema:
# - Sabemos que para n=30, a probabilidade de se obter uma média amostral dentro
#   de ±US$ 500 da média populacional é 0,5034
# - Precisamos calcular essa probabilidade para n=60 e n=120

# Vamos determinar o desvio padrão populacional utilizando a informação fornecida
# Para n=30 e probabilidade 0,5034 no intervalo ±500

# Primeiro, vamos encontrar o valor z correspondente à probabilidade 0,5034/2 = 0,2517
# (dividimos por 2 porque queremos a área de um lado, já que o intervalo é ±500)
z = stats.norm.ppf(0.5034/2 + 0.5)  # Somamos 0.5 porque queremos a área à direita do centro

# Com o valor de z, podemos calcular o desvio padrão populacional
# Para n=30, temos: z = 500 / (sigma/sqrt(30))
# Portanto: sigma = 500 * sqrt(30) / z
sigma = 500 * np.sqrt(30) / z

print(f"Valor de z para probabilidade 0,5034: {z:.4f}")
print(f"Desvio padrão populacional estimado: {sigma:.4f}")

# a. Probabilidade para n=60
erro_padrao_60 = sigma / np.sqrt(60)
z_60 = 500 / erro_padrao_60
prob_60 = 2 * (stats.norm.cdf(z_60) - 0.5)  # Multiplicamos por 2 para obter a área total

print(f"\na. Para n=60:")
print(f"   Erro padrão: {erro_padrao_60:.4f}")
print(f"   Valor de z: {z_60:.4f}")
print(f"   Probabilidade da média amostral estar dentro de ±US$ 500 da média populacional: {prob_60:.4f}")

# b. Probabilidade para n=120
erro_padrao_120 = sigma / np.sqrt(120)
z_120 = 500 / erro_padrao_120
prob_120 = 2 * (stats.norm.cdf(z_120) - 0.5)

print(f"\nb. Para n=120:")
print(f"   Erro padrão: {erro_padrao_120:.4f}")
print(f"   Valor de z: {z_120:.4f}")
print(f"   Probabilidade da média amostral estar dentro de ±US$ 500 da média populacional: {prob_120:.4f}")

# Vamos criar uma visualização para ilustrar o efeito do tamanho da amostra na distribuição amostral
plt.figure(figsize=(12, 6))


# Definindo o intervalo para plotagem
media_populacional = 0  # Podemos centralizar em 0 para simplificar
x = np.linspace(media_populacional - 4*erro_padrao_30, media_populacional + 4*erro_padrao_30, 1000)

# Calculando o erro padrão para n=30 (já sabemos que a probabilidade é 0,5034)
erro_padrao_30 = sigma / np.sqrt(30)

# Plotando as distribuições amostrais para os três tamanhos de amostra
plt.plot(x, stats.norm.pdf(x, media_populacional, erro_padrao_30), 
         'b-', label=f'n=30, σx̄={erro_padrao_30:.2f}, P(±500)={0.5034:.4f}')
plt.plot(x, stats.norm.pdf(x, media_populacional, erro_padrao_60), 
         'r-', label=f'n=60, σx̄={erro_padrao_60:.2f}, P(±500)={prob_60:.4f}')
plt.plot(x, stats.norm.pdf(x, media_populacional, erro_padrao_120), 
         'g-', label=f'n=120, σx̄={erro_padrao_120:.2f}, P(±500)={prob_120:.4f}')

# Marcando os limites de ±US$ 500
plt.axvline(media_populacional - 500, color='k', linestyle='--', alpha=0.7)
plt.axvline(media_populacional + 500, color='k', linestyle='--', alpha=0.7)

# Preenchendo as áreas dentro do intervalo ±US$ 500 para cada distribuição
x_fill = np.linspace(media_populacional - 500, media_populacional + 500, 100)

y_fill_30 = stats.norm.pdf(x_fill, media_populacional, erro_padrao_30)
plt.fill_between(x_fill, y_fill_30, color='blue', alpha=0.2)

y_fill_60 = stats.norm.pdf(x_fill, media_populacional, erro_padrao_60)
plt.fill_between(x_fill, y_fill_60, color='red', alpha=0.2)

y_fill_120 = stats.norm.pdf(x_fill, media_populacional, erro_padrao_120)
plt.fill_between(x_fill, y_fill_120, color='green', alpha=0.2)

plt.title('Distribuições Amostrais da Média para Diferentes Tamanhos de Amostra')
plt.xlabel('Desvio da Média Amostral em relação à Média Populacional (US$)')
plt.ylabel('Densidade de Probabilidade')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(media_populacional, color='k', linestyle='-', alpha=0.5)

# Adicionando anotações para explicar o gráfico
plt.annotate(f'±US$ 500', xy=(500, 0), xytext=(520, 0.0003),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10)

# Exibindo a solução visual
plt.tight_layout()
plt.show()

#Cap.7:28

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados do problema
# Califórnia:
media_calif = 22  # polegadas (média populacional)
n_calif = 30  # tamanho da amostra

# Nova York:
media_ny = 42  # polegadas (média populacional)
n_ny = 45  # tamanho da amostra

# Desvio padrão populacional para ambos os estados
desvio_padrao = 4  # polegadas

# a. Distribuição da precipitação média anual - Califórnia
erro_padrao_calif = desvio_padrao / np.sqrt(n_calif)

# b. Probabilidade da média amostral estar dentro de 1 polegada da média populacional - Califórnia
z_calif = 1 / erro_padrao_calif
prob_calif = 2 * (stats.norm.cdf(z_calif) - 0.5)  # Multiplica por 2 para obter a área total (duas caudas)

# c. Probabilidade da média amostral estar dentro de 1 polegada da média populacional - Nova York
erro_padrao_ny = desvio_padrao / np.sqrt(n_ny)
z_ny = 1 / erro_padrao_ny
prob_ny = 2 * (stats.norm.cdf(z_ny) - 0.5)

# d. Comparação das probabilidades
# Já calculamos as probabilidades nas partes b e c

# Visualização para ajudar na interpretação
plt.figure(figsize=(12, 8))
plt.show()
# Definindo intervalo para o eixo x
x_calif = np.linspace(media_calif - 4*erro_padrao_calif, media_calif + 4*erro_padrao_calif, 1000)
x_ny = np.linspace(media_ny - 4*erro_padrao_ny, media_ny + 4*erro_padrao_ny, 1000)

# Gráfico 1: Distribuição amostral para Califórnia
plt.subplot(2, 1, 1)
plt.plot(x_calif, stats.norm.pdf(x_calif, media_calif, erro_padrao_calif), 'b-', 
         label=f'Califórnia (n={n_calif}, σx̄={erro_padrao_calif:.4f})')

# Área dentro de ±1 polegada da média
x_fill_calif = np.linspace(media_calif - 1, media_calif + 1, 100)
y_fill_calif = stats.norm.pdf(x_fill_calif, media_calif, erro_padrao_calif)
plt.fill_between(x_fill_calif, y_fill_calif, color='blue', alpha=0.3, 
                 label=f'P(±1 polegada) = {prob_calif:.4f}')

plt.axvline(media_calif, color='k', linestyle='--')
plt.axvline(media_calif - 1, color='r', linestyle=':')
plt.axvline(media_calif + 1, color='r', linestyle=':')

plt.title('Distribuição Amostral da Precipitação Média Anual - Califórnia')
plt.xlabel('Precipitação Média Anual (polegadas)')
plt.ylabel('Densidade de Probabilidade')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: Distribuição amostral para Nova York
plt.subplot(2, 1, 2)
plt.plot(x_ny, stats.norm.pdf(x_ny, media_ny, erro_padrao_ny), 'g-', 
         label=f'Nova York (n={n_ny}, σx̄={erro_padrao_ny:.4f})')

# Área dentro de ±1 polegada da média
x_fill_ny = np.linspace(media_ny - 1, media_ny + 1, 100)
y_fill_ny = stats.norm.pdf(x_fill_ny, media_ny, erro_padrao_ny)
plt.fill_between(x_fill_ny, y_fill_ny, color='green', alpha=0.3, 
                 label=f'P(±1 polegada) = {prob_ny:.4f}')

plt.axvline(media_ny, color='k', linestyle='--')
plt.axvline(media_ny - 1, color='r', linestyle=':')
plt.axvline(media_ny + 1, color='r', linestyle=':')

plt.title('Distribuição Amostral da Precipitação Média Anual - Nova York')
plt.xlabel('Precipitação Média Anual (polegadas)')
plt.ylabel('Densidade de Probabilidade')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Gráfico 3: Comparação direta das duas distribuições amostrais
plt.figure(figsize=(12, 6))
plt.show()
# Normalizamos as distribuições para centrá-las em 0 para facilitar a comparação
x_calif_norm = np.linspace(-4*erro_padrao_calif, 4*erro_padrao_calif, 1000)
x_ny_norm = np.linspace(-4*erro_padrao_ny, 4*erro_padrao_ny, 1000)

plt.plot(x_calif_norm, stats.norm.pdf(x_calif_norm, 0, erro_padrao_calif), 'b-', 
         label=f'Califórnia (n={n_calif}, σx̄={erro_padrao_calif:.4f}, P(±1)={prob_calif:.4f})')
plt.plot(x_ny_norm, stats.norm.pdf(x_ny_norm, 0, erro_padrao_ny), 'g-', 
         label=f'Nova York (n={n_ny}, σx̄={erro_padrao_ny:.4f}, P(±1)={prob_ny:.4f})')

# Área dentro de ±1 polegada
x_fill_norm = np.linspace(-1, 1, 100)
y_fill_calif_norm = stats.norm.pdf(x_fill_norm, 0, erro_padrao_calif)
y_fill_ny_norm = stats.norm.pdf(x_fill_norm, 0, erro_padrao_ny)

plt.fill_between(x_fill_norm, y_fill_calif_norm, color='blue', alpha=0.3)
plt.fill_between(x_fill_norm, y_fill_ny_norm, color='green', alpha=0.3)

plt.axvline(0, color='k', linestyle='-')
plt.axvline(-1, color='r', linestyle='--')
plt.axvline(1, color='r', linestyle='--')

plt.title('Comparação das Distribuições Amostrais (Centralizadas)')
plt.xlabel('Desvio da Média Amostral em relação à Média Populacional (polegadas)')
plt.ylabel('Densidade de Probabilidade')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Resultados finais
print("RESULTADOS DA ANÁLISE DE DISTRIBUIÇÕES AMOSTRAIS")
print("=================================================\n")

print("a. Distribuição da precipitação média anual para a Califórnia:")
print(f"   - Média da distribuição amostral: {media_calif} polegadas")
print(f"   - Erro padrão (desvio padrão da distribuição amostral): {erro_padrao_calif:.4f} polegadas")
print(f"   - A distribuição segue uma Normal com N({media_calif}, {erro_padrao_calif:.4f}²)")

print("\nb. Probabilidade da média amostral estar dentro de 1 polegada da média populacional para a Califórnia:")
print(f"   P({media_calif-1} < x̄ < {media_calif+1}) = {prob_calif:.4f} = {prob_calif*100:.2f}%")

print("\nc. Probabilidade da média amostral estar dentro de 1 polegada da média populacional para Nova York:")
print(f"   P({media_ny-1} < x̄ < {media_ny+1}) = {prob_ny:.4f} = {prob_ny*100:.2f}%")

print("\nd. Comparação das probabilidades:")
if prob_calif > prob_ny:
    maior_prob = "Califórnia"
    explicacao = "menor tamanho de amostra (30 vs 45)"
else:
    maior_prob = "Nova York"
    explicacao = "maior tamanho de amostra (45 vs 30)"

print(f"   A probabilidade é maior para {maior_prob} apesar do {explicacao}.")
print("\n   Análise detalhada:")
print(f"   - Califórnia: n={n_calif}, σx̄={erro_padrao_calif:.4f}, z={z_calif:.4f}, P(±1)={prob_calif:.4f}")
print(f"   - Nova York: n={n_ny}, σx̄={erro_padrao_ny:.4f}, z={z_ny:.4f}, P(±1)={prob_ny:.4f}")
print("\n   Explicação:")
print("   Como o desvio padrão populacional é o mesmo (4 polegadas) para ambos os estados,")
print("   o erro padrão depende apenas do tamanho da amostra (σx̄ = σ/√n).")
print("   Nova York tem uma amostra maior (n=45 vs n=30), o que resulta em um menor erro padrão")
print("   e, consequentemente, uma distribuição amostral mais concentrada em torno da média populacional.")
print("   Isso aumenta a probabilidade de a média amostral estar dentro de ±1 polegada da média populacional.")
plt.show()

#Cap. 7:31

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Dados do problema
p = 0.40  # proporção populacional
n = 100   # tamanho da amostra

# a. Valor esperado de p̂ (p chapéu)
valor_esperado_p_chapeu = p
print(f"a. Valor esperado de p̂: {valor_esperado_p_chapeu}")

# b. Erro padrão de p̂
erro_padrao_p_chapeu = np.sqrt((p * (1 - p)) / n)
print(f"b. Erro padrão de p̂: {erro_padrao_p_chapeu:.4f}")

# c. Distribuição amostral de p̂
# Verificamos se satisfaz as condições para aproximação normal:
condicao_np = n * p
condicao_nq = n * (1 - p)
print(f"\nVerificação das condições para aproximação normal:")
print(f"n·p = {condicao_np} ≥ 5? {'Sim' if condicao_np >= 5 else 'Não'}")
print(f"n·(1-p) = {condicao_nq} ≥ 5? {'Sim' if condicao_nq >= 5 else 'Não'}")

# Como as condições são satisfeitas, podemos usar a aproximação normal
# Realizamos uma simulação para demonstrar a distribuição amostral de p̂

num_simulacoes = 10000
proporcoes_amostrais = []

# Gerando amostras aleatórias e calculando a proporção em cada uma
for _ in range(num_simulacoes):
    # Gerando amostra de uma distribuição Bernoulli com p=0.40
    amostra = np.random.binomial(1, p, n)
    # Calculando a proporção na amostra
    proporcao_amostral = np.mean(amostra)
    proporcoes_amostrais.append(proporcao_amostral)

# Calculando média e desvio padrão empíricos
media_empirica = np.mean(proporcoes_amostrais)
desvio_padrao_empirico = np.std(proporcoes_amostrais)

# Criando a visualização da distribuição amostral de p̂
plt.figure(figsize=(12, 8))

# Histograma das proporções amostrais simuladas
plt.subplot(2, 1, 1)
sns.histplot(proporcoes_amostrais, bins=30, kde=True, color='blue', alpha=0.6)

# Adicionando a curva normal teórica
x = np.linspace(p - 4*erro_padrao_p_chapeu, p + 4*erro_padrao_p_chapeu, 1000)
y = stats.norm.pdf(x, p, erro_padrao_p_chapeu)
plt.plot(x, y, 'r-', linewidth=2, label='Distribuição Normal Teórica')

plt.axvline(p, color='green', linestyle='--', linewidth=2, 
           label=f'Proporção Populacional (p = {p})')
plt.axvline(media_empirica, color='purple', linestyle=':', linewidth=2, 
           label=f'Média Empírica (p̂ = {media_empirica:.4f})')

plt.title('Distribuição Amostral da Proporção (p̂) - n = 100, p = 0.40')
plt.xlabel('Proporção Amostral (p̂)')
plt.ylabel('Frequência')
plt.legend()
plt.grid(alpha=0.3)

# Gráfico QQ para verificar normalidade
plt.subplot(2, 1, 2)
stats.probplot(proporcoes_amostrais, dist="norm", plot=plt)
plt.title('Gráfico Q-Q: Verificação da Normalidade da Distribuição Amostral')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# d. O que a distribuição amostral de p̂ mostra
print("\nd. O que a distribuição amostral de p̂ mostra:")
print("   1. A distribuição amostral de p̂ segue aproximadamente uma distribuição normal")
print(f"      com média igual à proporção populacional (p = {p}) e erro padrão")
print(f"      σp̂ = √(p(1-p)/n) = {erro_padrao_p_chapeu:.4f}.")
print("   2. Esta distribuição mostra como a proporção amostral (p̂) varia quando")
print("      diferentes amostras de tamanho 100 são retiradas da população.")
print("   3. A distribuição nos permite calcular probabilidades associadas a")
print("      possíveis valores de p̂ e construir intervalos de confiança para p.")
print("   4. A maioria das proporções amostrais estará em torno da proporção")
print("      populacional, com aproximadamente 95% dos valores dentro de ±2 erros")
print(f"      padrão: {p-2*erro_padrao_p_chapeu:.4f} a {p+2*erro_padrao_p_chapeu:.4f}.")
print("   5. À medida que o tamanho da amostra aumenta, a distribuição se torna")
print("      mais concentrada em torno da proporção populacional, reduzindo a")
print("      variabilidade das estimativas.")
print("\nConfirmação empírica dos parâmetros teóricos da distribuição:")
print(f"Média teórica de p̂: {p}")
print(f"Média empírica de p̂ (baseada na simulação): {media_empirica:.4f}")
print(f"Erro padrão teórico: {erro_padrao_p_chapeu:.4f}")
print(f"Desvio padrão empírico: {desvio_padrao_empirico:.4f}")

# Calculando algumas probabilidades de interesse
z_1 = (0.45 - p) / erro_padrao_p_chapeu
prob_maior_045 = 1 - stats.norm.cdf(z_1)
print(f"\nProbabilidade de p̂ > 0.45: {prob_maior_045:.4f} = {prob_maior_045*100:.2f}%")

z_2 = (0.35 - p) / erro_padrao_p_chapeu
prob_menor_035 = stats.norm.cdf(z_2)
print(f"Probabilidade de p̂ < 0.35: {prob_menor_035:.4f} = {prob_menor_035*100:.2f}%")

intervalo_1_sigma = 2 * stats.norm.cdf(1) - 1
intervalo_2_sigma = 2 * stats.norm.cdf(2) - 1
print(f"\nProbabilidade de p̂ estar dentro de ±1 erro padrão de p: {intervalo_1_sigma:.4f} = {intervalo_1_sigma*100:.2f}%")
print(f"Probabilidade de p̂ estar dentro de ±2 erros padrão de p: {intervalo_2_sigma:.4f} = {intervalo_2_sigma*100:.2f}%")

#Cap.8:2

import numpy as np
import scipy.stats as stats

# Dados do problema
media_amostral = 32
n = 50
sigma = 6  # Desvio padrão populacional é dado (σ = 6)

# Para calcular o intervalo de confiança quando conhecemos o desvio padrão populacional,
# usamos a distribuição normal (z)
def calcular_ic_normal(nivel_confianca):
    # Obtendo o valor crítico z
    z = stats.norm.ppf((1 + nivel_confianca) / 2)
    
    # Erro padrão da média
    erro_padrao = sigma / np.sqrt(n)
    
    # Margem de erro
    margem_erro = z * erro_padrao
    
    # Intervalo de confiança
    limite_inferior = media_amostral - margem_erro
    limite_superior = media_amostral + margem_erro
    
    return limite_inferior, limite_superior

# a. Intervalo de confiança de 90% para a média populacional
ic_90 = calcular_ic_normal(0.90)
print(f"a. IC 90%: ({ic_90[0]:.4f}, {ic_90[1]:.4f})")

# b. Intervalo de confiança de 95% para a média populacional
ic_95 = calcular_ic_normal(0.95)
print(f"b. IC 95%: ({ic_95[0]:.4f}, {ic_95[1]:.4f})")

# c. Intervalo de confiança de 99% para a média populacional
ic_99 = calcular_ic_normal(0.99)
print(f"c. IC 99%: ({ic_99[0]:.4f}, {ic_99[1]:.4f})")

#Cap 8:7

import numpy as np
import scipy.stats as stats
import pandas as pd

# Dados do problema
custo_medio_anual = 1843  # Custo médio anual em US$
desvio_padrao = 255      # Desvio padrão populacional (σ = US$ 255)
tamanho_amostra = 50     # Tamanho da amostra

# a. Margem de erro para o intervalo de confiança de 95%
def calcular_margem_erro(nivel_confianca):
    # Obtendo o valor crítico z para o nível de confiança especificado
    z = stats.norm.ppf((1 + nivel_confianca) / 2)
    
    # Calculando o erro padrão da média
    erro_padrao = desvio_padrao / np.sqrt(tamanho_amostra)
    
    # Calculando a margem de erro
    margem_erro = z * erro_padrao
    
    return margem_erro

# Calculando a margem de erro para 95% de confiança
margem_erro_95 = calcular_margem_erro(0.95)
print(f"a. Margem de erro para IC de 95%: US$ {margem_erro_95:.2f}")

# b. Simulando os dados dos 50 proprietários para cálculo do IC de 95%
# Vamos gerar dados aleatórios com base na média e desvio padrão informados
np.random.seed(42)  # Para reprodutibilidade
dados_setters = np.random.normal(custo_medio_anual, desvio_padrao, tamanho_amostra)

# Calculando a média amostral
media_amostral = np.mean(dados_setters)
print(f"\nb. Média amostral calculada: US$ {media_amostral:.2f}")

# Calculando o IC de 95% usando a distribuição t de Student (para amostra)
# Como estamos usando uma amostra e não sabemos o desvio padrão populacional,
# usamos o desvio padrão amostral e a distribuição t
desvio_padrao_amostral = np.std(dados_setters, ddof=1)  # ddof=1 para desvio padrão amostral
erro_padrao_amostral = desvio_padrao_amostral / np.sqrt(tamanho_amostra)
t_critico = stats.t.ppf(0.975, tamanho_amostra - 1)  # 0.975 para IC de 95% (bilateral)

limite_inferior = media_amostral - t_critico * erro_padrao_amostral
limite_superior = media_amostral + t_critico * erro_padrao_amostral

print(f"Intervalo de Confiança de 95% para o custo médio: (US$ {limite_inferior:.2f}, US$ {limite_superior:.2f})")
print(f"Margem de erro (usando a distribuição t): US$ {t_critico * erro_padrao_amostral:.2f}")

# Alternativa: calculando o IC usando a função stats.t.interval diretamente
ic_95 = stats.t.interval(0.95, tamanho_amostra - 1, loc=media_amostral, scale=erro_padrao_amostral)
print(f"IC 95% (usando stats.t.interval): (US$ {ic_95[0]:.2f}, US$ {ic_95[1]:.2f})")

#Cap.8:11

import scipy.stats as stats
import numpy as np

# Definindo os graus de liberdade
gl = 16

# a. À direita de 2,120
t_a = 2.120
prob_a = 1 - stats.t.cdf(t_a, gl)
print(f"a. Probabilidade à direita de 2,120: {prob_a:.6f}")

# b. À esquerda de 1,337
t_b = 1.337
prob_b = stats.t.cdf(t_b, gl)
print(f"b. Probabilidade à esquerda de 1,337: {prob_b:.6f}")

# c. À esquerda de -1,746
t_c = -1.746
prob_c = stats.t.cdf(t_c, gl)
print(f"c. Probabilidade à esquerda de -1,746: {prob_c:.6f}")

# d. À direita de 2,583
t_d = 2.583
prob_d = 1 - stats.t.cdf(t_d, gl)
print(f"d. Probabilidade à direita de 2,583: {prob_d:.6f}")

# e. Entre -2,120 e 2,120
t_e_min = -2.120
t_e_max = 2.120
prob_e = stats.t.cdf(t_e_max, gl) - stats.t.cdf(t_e_min, gl)
print(f"e. Probabilidade entre -2,120 e 2,120: {prob_e:.6f}")

# f. Entre -1,746 e 1,746
t_f_min = -1.746
t_f_max = 1.746
prob_f = stats.t.cdf(t_f_max, gl) - stats.t.cdf(t_f_min, gl)
print(f"f. Probabilidade entre -1,746 e 1,746: {prob_f:.6f}")
plt.show()

#Cap.8:13

import numpy as np
import scipy.stats as stats

# Dados amostrais
dados = np.array([10, 8, 12, 15, 13, 11, 6, 5])

# a. Estimativa pontual da média populacional
media_amostral = np.mean(dados)

# b. Estimativa pontual do desvio padrão populacional
# Usando n-1 no denominador (desvio padrão amostral)
desvio_padrao_amostral = np.std(dados, ddof=1)

# c. Calculando a margem de erro com 95% de confiança
n = len(dados)
graus_liberdade = n - 1
t_valor = stats.t.ppf(0.975, graus_liberdade)  # t crítico para 95% de confiança
margem_erro = t_valor * (desvio_padrao_amostral / np.sqrt(n))

# d. Intervalo de confiança de 95% para a média populacional
ic_inferior = media_amostral - margem_erro
ic_superior = media_amostral + margem_erro

# Exibindo os resultados
print(f"a. Estimativa pontual da média populacional: {media_amostral:.4f}")
print(f"b. Estimativa pontual do desvio padrão populacional: {desvio_padrao_amostral:.4f}")
print(f"c. Margem de erro (95% de confiança): {margem_erro:.4f}")
print(f"d. Intervalo de confiança de 95% para a média populacional: [{ic_inferior:.4f}, {ic_superior:.4f}]")

# Informações adicionais para verificação
print("\nInformações adicionais:")
print(f"Tamanho da amostra (n): {n}")
print(f"Graus de liberdade: {graus_liberdade}")
print(f"Valor t crítico (95% de confiança): {t_valor:.4f}")
print(f"Erro padrão da média: {desvio_padrao_amostral / np.sqrt(n):.4f}")

#Cap.8:19

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Dados do problema
preco_toquio = 40  # preço em US$ para duas pessoas em Tóquio
n_hong_kong = 42   # tamanho da amostra de Hong Kong

# Vamos simular os dados de Hong Kong com base nas informações
# Nota: Em um cenário real, usaríamos os 42 valores reais do arquivo HongKongMeals
# Para fins de exemplo, vamos gerar dados hipotéticos com média próxima de 45
np.random.seed(123)  # Para reprodutibilidade
dados_hong_kong = np.random.normal(45, 8, n_hong_kong)

# a. Calculando a margem de erro com 95% de confiança
media_hong_kong = np.mean(dados_hong_kong)
desvio_padrao_hong_kong = np.std(dados_hong_kong, ddof=1)
erro_padrao = desvio_padrao_hong_kong / np.sqrt(n_hong_kong)
t_valor = stats.t.ppf(0.975, n_hong_kong - 1)  # valor crítico t para 95% de confiança
margem_erro = t_valor * erro_padrao

# b. Calculando o intervalo de confiança de 95% para a média populacional
ic_inferior = media_hong_kong - margem_erro
ic_superior = media_hong_kong + margem_erro

# c. Comparação dos preços entre Hong Kong e Tóquio
diferenca_percentual = ((media_hong_kong - preco_toquio) / preco_toquio) * 100

# Exibindo resultados
print(f"Preço médio em Tóquio: US$ {preco_toquio:.2f} para duas pessoas")
print(f"Preço médio estimado em Hong Kong: US$ {media_hong_kong:.2f} para duas pessoas")
print(f"\na. Margem de erro (95% de confiança): {margem_erro:.2f}")
print(f"\nb. Intervalo de confiança de 95% para a média em Hong Kong: [US$ {ic_inferior:.2f}, US$ {ic_superior:.2f}]")

print(f"\nc. Comparação entre Hong Kong e Tóquio:")
if preco_toquio < ic_inferior:
    print(f"   As refeições em Hong Kong são significativamente mais caras que em Tóquio (cerca de {diferenca_percentual:.1f}% mais caras)")
elif preco_toquio > ic_superior:
    print(f"   As refeições em Hong Kong são significativamente mais baratas que em Tóquio (cerca de {-diferenca_percentual:.1f}% mais baratas)")
else:
    print(f"   Não há diferença estatisticamente significativa entre os preços de Hong Kong e Tóquio")

# Visualização (opcional)
plt.figure(figsize=(10, 6))
plt.hist(dados_hong_kong, bins=10, alpha=0.7, color='skyblue', label='Preços em Hong Kong')
plt.axvline(media_hong_kong, color='blue', linestyle='dashed', linewidth=2, label=f'Média Hong Kong (US$ {media_hong_kong:.2f})')
plt.axvline(preco_toquio, color='red', linestyle='dashed', linewidth=2, label=f'Preço Tóquio (US$ {preco_toquio})')
plt.axvline(ic_inferior, color='green', linestyle='dotted', linewidth=2, label=f'IC 95% Inferior (US$ {ic_inferior:.2f})')
plt.axvline(ic_superior, color='green', linestyle='dotted', linewidth=2, label=f'IC 95% Superior (US$ {ic_superior:.2f})')
plt.xlabel('Preço (US$)')
plt.ylabel('Frequência')
plt.title('Comparação de Preços de Refeições para Duas Pessoas')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#Cap.8:22

import numpy as np
import scipy.stats as stats

# Dados do problema
receita_total = 94.3e6  # US$ 94,3 milhões
n_cinemas = 30  # tamanho da amostra
preco_ingresso = 8.11  # US$ por ingresso
total_salas = 4080  # número total de salas exibindo o filme

# Vamos supor que os dados da amostra estão em um arquivo guardians.txt
# Como não temos acesso aos dados reais, vamos simular dados compatíveis com o problema
np.random.seed(123)  # Para reprodutibilidade
# Simulando receitas por sala que resultariam em aproximadamente US$ 94,3 milhões no total
receita_media_por_sala = receita_total / total_salas
# Adicionando alguma variação para simular os dados da amostra
receitas_amostra = np.random.normal(receita_media_por_sala, receita_media_por_sala * 0.2, n_cinemas)

# a. Calculando o intervalo de confiança de 95% para a receita média por sala
media_amostra = np.mean(receitas_amostra)
desvio_padrao_amostra = np.std(receitas_amostra, ddof=1)
erro_padrao = desvio_padrao_amostra / np.sqrt(n_cinemas)
t_valor = stats.t.ppf(0.975, n_cinemas - 1)  # valor crítico t para 95% de confiança
margem_erro = t_valor * erro_padrao

ic_inferior = media_amostra - margem_erro
ic_superior = media_amostra + margem_erro

# b. Calculando o número médio de clientes por sala
clientes_por_sala = media_amostra / preco_ingresso
ic_clientes_inferior = ic_inferior / preco_ingresso
ic_clientes_superior = ic_superior / preco_ingresso

# c. Estimando o número total de clientes no fim de semana
total_clientes_estimado = clientes_por_sala * total_salas

# Exibindo resultados
print("Análise da Bilheteria de 'Guardiões da Galáxia'")
print("==============================================")
print(f"Receita total reportada: US$ {receita_total/1e6:.1f} milhões")
print(f"Amostra de {n_cinemas} cinemas")
print(f"Preço do ingresso: US$ {preco_ingresso}")
print(f"Total de salas exibindo o filme: {total_salas}\n")

print(f"a. Intervalo de confiança de 95% para a receita média por sala:")
print(f"   [US$ {ic_inferior:.2f}, US$ {ic_superior:.2f}]")
print(f"   Receita média estimada por sala: US$ {media_amostra:.2f}")
print(f"   Margem de erro: US$ {margem_erro:.2f}\n")

print(f"b. Número médio de clientes por sala de cinema:")
print(f"   {clientes_por_sala:.0f} clientes por sala")
print(f"   Intervalo de confiança de 95%: [{ic_clientes_inferior:.0f}, {ic_clientes_superior:.0f}] clientes\n")

print(f"c. Número total estimado de clientes que assistiram ao filme:")
print(f"   {total_clientes_estimado:.0f} clientes no total")
print(f"   (Total de vendas de ingressos em {total_salas} salas no fim de semana prolongado)")


#Cap.8:23

import numpy as np
import scipy.stats as stats
import math

# Dados do problema
margem_erro_desejada = 10
desvio_padrao_populacional = 40
nivel_confianca = 0.95

# Para um IC de 95%, o z-crítico é aproximadamente 1.96
z_critico = stats.norm.ppf(1 - (1 - nivel_confianca) / 2)

# Fórmula para calcular o tamanho amostral necessário:
# n = (z^2 * σ^2) / E^2
# Onde:
# n = tamanho da amostra
# z = valor crítico z
# σ = desvio padrão populacional
# E = margem de erro desejada

tamanho_amostral = ((z_critico**2) * (desvio_padrao_populacional**2)) / (margem_erro_desejada**2)

# Como o tamanho da amostra deve ser um número inteiro, arredondamos para cima
tamanho_amostral_arredondado = math.ceil(tamanho_amostral)

# Exibindo os resultados
print(f"Margem de erro desejada: {margem_erro_desejada}")
print(f"Desvio padrão populacional: {desvio_padrao_populacional}")
print(f"Nível de confiança: {nivel_confianca * 100}%")
print(f"Valor z crítico para IC {nivel_confianca * 100}%: {z_critico:.4f}")
print(f"\nTamanho amostral necessário (fórmula): {tamanho_amostral:.4f}")
print(f"Tamanho amostral necessário (arredondado para cima): {tamanho_amostral_arredondado}")

# Verificação (opcional)
# Calculando a margem de erro com o tamanho de amostra arredondado para confirmar
margem_erro_verificacao = z_critico * (desvio_padrao_populacional / math.sqrt(tamanho_amostral_arredondado))
print(f"\nVerificação - Margem de erro com n = {tamanho_amostral_arredondado}: {margem_erro_verificacao:.4f}")