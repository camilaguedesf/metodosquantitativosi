#Capítulo 9 - Testes de hipoteses: 4, 7, 9, 14, 17, 24, 32

#Cap. 9:4

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Definindo o problema
custo_atual = 220  # Custo médio do método atual (US$ por hora)

# Função para executar o teste de hipóteses
def teste_hipoteses(dados_novo_metodo, nivel_significancia=0.05):
    # Tamanho da amostra
    n = len(dados_novo_metodo)
    
    # Média amostral do novo método
    media_novo = np.mean(dados_novo_metodo)
    
    # Desvio padrão amostral do novo método
    desvio_padrao = np.std(dados_novo_metodo, ddof=1)
    
    # Hipóteses:
    # H0: μ >= 220 (o novo método não reduz os custos)
    # H1: μ < 220 (o novo método reduz os custos)
    
    # Estatística do teste t
    t_estatistica = (media_novo - custo_atual) / (desvio_padrao / np.sqrt(n))
    
    # Valor-p para teste unilateral à esquerda
    p_valor = stats.t.cdf(t_estatistica, df=n-1)
    
    # Valor crítico para rejeição de H0
    t_critico = stats.t.ppf(nivel_significancia, df=n-1)
    
    # Resultados
    print(f"Dados do teste de hipóteses:")
    print(f"Custo médio do método atual: US$ {custo_atual} por hora")
    print(f"Custo médio do novo método: US$ {media_novo:.2f} por hora")
    print(f"Tamanho da amostra: {n}")
    print(f"Desvio padrão amostral: {desvio_padrao:.2f}")
    print(f"Estatística t: {t_estatistica:.4f}")
    print(f"Valor-p: {p_valor:.4f}")
    print(f"Valor crítico t para α = {nivel_significancia}: {t_critico:.4f}")
    
    # Conclusão sobre H0
    print("\nConclusão:")
    if p_valor < nivel_significancia:
        print(f"Rejeitamos H0 com nível de significância {nivel_significancia}.")
        print("Há evidência estatística de que o novo método reduz os custos.")
    else:
        print(f"Não rejeitamos H0 com nível de significância {nivel_significancia}.")
        print("Não há evidência estatística suficiente de que o novo método reduz os custos.")
    
    # Informações adicionais sobre quando rejeitar H0
    print("\nInformações sobre rejeição de H0:")
    print("a) H0 é rejeitada quando o valor-p < nível de significância α")
    print(f"   No nosso caso, H0 é rejeitada quando o valor-p < {nivel_significancia}")
    print("b) H0 é rejeitada quando t_calculado < t_crítico (para teste unilateral à esquerda)")
    print(f"   No nosso caso, H0 é rejeitada quando t_calculado < {t_critico:.4f}")
    
    # Visualização
    visualizar_teste(dados_novo_metodo, t_estatistica, t_critico)
    
    return p_valor, t_estatistica, t_critico

# Função para visualizar o teste
def visualizar_teste(dados, t_estatistica, t_critico):
    plt.figure(figsize=(10, 6))
    
    # Definindo o intervalo para plotar a distribuição t
    x = np.linspace(-4, 4, 1000)
    df = len(dados) - 1
    y = stats.t.pdf(x, df)
    
    # Plotando a distribuição t
    plt.plot(x, y, 'b-', lw=2, label='Distribuição t')
    
    # Área de rejeição
    x_rej = np.linspace(-4, t_critico, 100)
    y_rej = stats.t.pdf(x_rej, df)
    plt.fill_between(x_rej, y_rej, color='red', alpha=0.3, label='Região de rejeição')
    
    # Marcando o valor t calculado
    plt.axvline(t_estatistica, color='green', linestyle='--', lw=2, label='t calculado')
    plt.axvline(t_critico, color='red', linestyle='--', lw=2, label='t crítico')
    
    plt.title('Teste de Hipóteses para Redução de Custos')
    plt.xlabel('Valor t')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True)
    plt.show()

# Exemplo de uso:
# Vamos simular alguns dados para teste
np.random.seed(42)  # Para reprodutibilidade
# Simulando um novo método com média 210 (menor que o atual) e desvio padrão 15
dados_simulados = np.random.normal(210, 15, 30)

# Executando o teste
teste_hipoteses(dados_simulados)

# Para responder às partes específicas da questão:
print("\nRespostas específicas para a questão:")
print("a) Hipóteses alternativas:")
print("   H0: μ >= 220 (o novo método não reduz os custos)")
print("   H1: μ < 220 (o novo método reduz os custos)")
print("\nb) Conclusão de quando H0 não pode ser rejeitada:")
print("   H0 não pode ser rejeitada quando o valor-p ≥ α ou quando t_calculado ≥ t_crítico")
print("\nc) Conclusão de quando H0 pode ser rejeitada:")
print("   H0 pode ser rejeitada quando o valor-p < α ou quando t_calculado < t_crítico")

#Cap.9:7

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Configurações iniciais
media_atual = 8000  # Média atual de vendas: US$ 8 mil por semana por vendedor

def analise_programa_incentivo(dados_experimentais, alfa=0.05):
    """
    Realiza análise estatística para determinar se o programa de incentivo aumentou as vendas.
    
    Args:
        dados_experimentais: Dados de vendas após implementação do programa de incentivo
        alfa: Nível de significância para o teste
    """
    # Estatísticas descritivas
    n = len(dados_experimentais)
    media_nova = np.mean(dados_experimentais)
    desvio_padrao = np.std(dados_experimentais, ddof=1)
    
    # Hipóteses:
    # H0: μ ≤ 8000 (o programa não aumentou a média de vendas)
    # H1: μ > 8000 (o programa aumentou a média de vendas)
    
    # Cálculo do teste t
    t_estatistica = (media_nova - media_atual) / (desvio_padrao / np.sqrt(n))
    
    # Valor-p para teste unilateral à direita
    p_valor = 1 - stats.t.cdf(t_estatistica, df=n-1)
    
    # Valor crítico
    t_critico = stats.t.ppf(1 - alfa, df=n-1)
    
    # Resultados
    print("=== Análise do Programa de Incentivo de Vendas da Carpetland ===")
    print(f"Média atual de vendas: US$ {media_atual/1000:.1f} mil por semana")
    print(f"Média após programa de incentivo: US$ {media_nova/1000:.2f} mil por semana")
    print(f"Tamanho da amostra experimental: {n}")
    print(f"Desvio padrão amostral: US$ {desvio_padrao/1000:.2f} mil")
    print(f"Estatística t calculada: {t_estatistica:.4f}")
    print(f"Valor-p: {p_valor:.4f}")
    print(f"Valor crítico t (α = {alfa}): {t_critico:.4f}")
    
    # Conclusão
    print("\nConclusão do teste:")
    if p_valor < alfa:
        print(f"Rejeitamos H0 com nível de significância {alfa}.")
        print("Há evidência estatística de que o programa de incentivo aumentou a média de vendas.")
    else:
        print(f"Não rejeitamos H0 com nível de significância {alfa}.")
        print("Não há evidência estatística suficiente de que o programa de incentivo aumentou a média de vendas.")
    
    # Análise dos erros
    print("\n=== Análise dos Erros Tipo I e Tipo II ===")
    print("a) Hipóteses formuladas:")
    print("   H0: μ ≤ $8,000 (o programa de incentivo não aumentou a média de vendas)")
    print("   H1: μ > $8,000 (o programa de incentivo aumentou a média de vendas)")
    
    print("\nb) Erro Tipo I nesta situação:")
    print("   Definição: Rejeitar H0 quando H0 é verdadeira.")
    print("   Contexto: Concluir que o programa de incentivo aumentou as vendas quando na realidade não aumentou.")
    print("   Consequências: A empresa implementaria o programa de incentivo acreditando em sua eficácia,")
    print("   mas estaria pagando mais aos vendedores sem obter o aumento esperado nas vendas.")
    print("   Isso resultaria em custos adicionais sem o retorno financeiro esperado.")
    
    print("\nc) Erro Tipo II nesta situação:")
    print("   Definição: Não rejeitar H0 quando H0 é falsa.")
    print("   Contexto: Concluir que o programa de incentivo não aumentou as vendas quando na realidade aumentou.")
    print("   Consequências: A empresa deixaria de implementar um programa que realmente aumentaria as vendas.")
    print("   Isso resultaria em perda de oportunidade de aumento nas receitas e potencial vantagem competitiva.")
    
    # Visualização do teste
    visualizar_teste(t_estatistica, t_critico, n-1)
    
    return p_valor, t_estatistica, t_critico

def visualizar_teste(t_estat, t_crit, df):
    """
    Visualiza a distribuição t com região de rejeição e estatística calculada.
    """
    plt.figure(figsize=(10, 6))
    
    # Intervalo para plotar
    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df)
    
    # Plotando a distribuição t
    plt.plot(x, y, 'b-', lw=2, label='Distribuição t')
    
    # Área de rejeição (cauda direita)
    x_rej = np.linspace(t_crit, 4, 100)
    y_rej = stats.t.pdf(x_rej, df)
    plt.fill_between(x_rej, y_rej, color='red', alpha=0.3, label='Região de rejeição')
    
    # Marcando o valor t calculado e crítico
    plt.axvline(t_estat, color='green', linestyle='--', lw=2, label='t calculado')
    plt.axvline(t_crit, color='red', linestyle='--', lw=2, label='t crítico')
    
    plt.title('Teste de Hipóteses: Programa de Incentivo de Vendas')
    plt.xlabel('Valor t')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True)
    plt.show()

# Exemplo de uso com dados simulados
# Simulando dados de vendas após implementação do programa
# Média de $8,500 (aumento de $500) com desvio padrão de $1,000
np.random.seed(123)
dados_vendas_pos_programa = np.random.normal(8500, 1000, 30)

# Executando a análise
analise_programa_incentivo(dados_vendas_pos_programa)

#Cap. 9:9

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Dados do problema
media_amostral = 19.4
desvio_padrao_populacional = 2
tamanho_amostra = 50
media_hipotese = 20
nivel_significancia = 0.05

# a) Cálculo da estatística de teste
# Para médias com desvio padrão populacional conhecido, usamos a distribuição Z
z_estatistica = (media_amostral - media_hipotese) / (desvio_padrao_populacional / np.sqrt(tamanho_amostra))
print(f"a) Estatística de teste Z = {z_estatistica:.4f}")

# b) Cálculo do valor-p
# Como é um teste unilateral à esquerda (H1: μ < 20), calculamos P(Z ≤ z_estatistica)
p_valor = stats.norm.cdf(z_estatistica)
print(f"b) Valor-p = {p_valor:.4f}")

# c) Conclusão usando α = 0.05
print("\nc) Usando α = 0.05:")
if p_valor < nivel_significancia:
    conclusao = "Rejeitamos H0"
    print(f"Como o valor-p ({p_valor:.4f}) < α ({nivel_significancia}), {conclusao}.")
    print("Há evidência estatística de que a média populacional é menor que 20.")
else:
    conclusao = "Não rejeitamos H0"
    print(f"Como o valor-p ({p_valor:.4f}) ≥ α ({nivel_significancia}), {conclusao}.")
    print("Não há evidência estatística suficiente para concluir que a média populacional é menor que 20.")

# d) Valor crítico e regra de rejeição
z_critico = stats.norm.ppf(nivel_significancia)
print(f"\nd) Valor crítico para α = 0.05: z_crítico = {z_critico:.4f}")
print("Regra de rejeição: Rejeitar H0 se z_estatistica < z_crítico")

if z_estatistica < z_critico:
    conclusao_regra = "Rejeitamos H0"
    print(f"Como z_estatistica ({z_estatistica:.4f}) < z_crítico ({z_critico:.4f}), {conclusao_regra}.")
    print("Há evidência estatística de que a média populacional é menor que 20.")
else:
    conclusao_regra = "Não rejeitamos H0"
    print(f"Como z_estatistica ({z_estatistica:.4f}) ≥ z_crítico ({z_critico:.4f}), {conclusao_regra}.")
    print("Não há evidência estatística suficiente para concluir que a média populacional é menor que 20.")

# Visualização do teste
plt.figure(figsize=(10, 6))

# Definindo o intervalo para plotar a distribuição normal
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)

# Plotando a distribuição normal padrão
plt.plot(x, y, 'b-', lw=2, label='Distribuição Normal Padrão')

# Área de rejeição
x_rej = np.linspace(-4, z_critico, 100)
y_rej = stats.norm.pdf(x_rej)
plt.fill_between(x_rej, y_rej, color='red', alpha=0.3, label='Região de rejeição')

# Marcando o valor z calculado e crítico
plt.axvline(z_estatistica, color='green', linestyle='--', lw=2, label='Z calculado')
plt.axvline(z_critico, color='red', linestyle='--', lw=2, label='Z crítico')

plt.title('Teste de Hipóteses: H0: μ ≥ 20 vs H1: μ < 20')
plt.xlabel('Valor Z')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.show()

# Resumo das respostas
print("\nRespostas:")
print(f"a) Estatística de teste: Z = {z_estatistica:.4f}")
print(f"b) Valor-p = {p_valor:.4f}")
print(f"c) Conclusão com α = 0.05: {conclusao}. {'Rejeitamos a hipótese nula.' if p_valor < nivel_significancia else 'Não rejeitamos a hipótese nula.'}")
print(f"d) Regra de rejeição: Rejeitar H0 se Z < {z_critico:.4f}")
print(f"   Conclusão baseada na regra: {conclusao_regra}")

#Cap. 9:14

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Dados do problema
media_hipotese = 22
desvio_padrao_populacional = 10
tamanho_amostra = 75
nivel_significancia = 0.01

# Médias amostrais a serem testadas
medias_amostrais = [23, 25.1, 20]

def teste_hipotese_bilateral(media_amostral):
    """
    Realiza teste de hipótese bilateral:
    H0: μ = 22
    H1: μ ≠ 22
    """
    # Cálculo da estatística de teste Z
    z_estatistica = (media_amostral - media_hipotese) / (desvio_padrao_populacional / np.sqrt(tamanho_amostra))
    
    # Cálculo do valor-p para teste bilateral
    # Para teste bilateral: valor-p = 2 * min(P(Z ≤ z), P(Z ≥ z))
    if z_estatistica < 0:
        p_valor = 2 * stats.norm.cdf(z_estatistica)
    else:
        p_valor = 2 * (1 - stats.norm.cdf(z_estatistica))
    
    # Decisão usando o valor-p
    if p_valor < nivel_significancia:
        decisao = "Rejeitamos H0"
        conclusao = f"Há evidência estatística de que a média populacional é diferente de {media_hipotese}."
    else:
        decisao = "Não rejeitamos H0"
        conclusao = f"Não há evidência estatística suficiente para concluir que a média populacional é diferente de {media_hipotese}."
    
    return z_estatistica, p_valor, decisao, conclusao

# Calcular valor crítico para α = 0.01 (teste bilateral)
z_critico = stats.norm.ppf(1 - nivel_significancia/2)  # Para teste bilateral dividimos alfa por 2

# Aplicar o teste para cada média amostral
resultados = []
for i, media in enumerate(medias_amostrais, start=1):
    letra = chr(96 + i)  # 'a', 'b', 'c'
    z, p, decisao, conclusao = teste_hipotese_bilateral(media)
    resultados.append((letra, media, z, p, decisao, conclusao))
    
    print(f"\nCaso {letra}. x̄ = {media}")
    print(f"Estatística Z = {z:.4f}")
    print(f"Valor-p = {p:.4f}")
    print(f"Para α = {nivel_significancia}, {decisao}.")
    print(conclusao)

# Visualização gráfica dos resultados
plt.figure(figsize=(12, 8))

# Definindo o intervalo para plotar a distribuição normal
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)

# Plotando a distribuição normal padrão
plt.plot(x, y, 'b-', lw=2, label='Distribuição Normal Padrão')

# Áreas de rejeição
x_rej_esq = np.linspace(-4, -z_critico, 100)
y_rej_esq = stats.norm.pdf(x_rej_esq)
plt.fill_between(x_rej_esq, y_rej_esq, color='red', alpha=0.3)

x_rej_dir = np.linspace(z_critico, 4, 100)
y_rej_dir = stats.norm.pdf(x_rej_dir)
plt.fill_between(x_rej_dir, y_rej_dir, color='red', alpha=0.3, label='Regiões de rejeição')

# Marcando os valores críticos
plt.axvline(-z_critico, color='red', linestyle='--', lw=2, label=f'Z crítico = ±{z_critico:.4f}')
plt.axvline(z_critico, color='red', linestyle='--', lw=2)

# Marcando os valores Z calculados
cores = ['green', 'purple', 'orange']
for i, (letra, media, z, p, _, _) in enumerate(resultados):
    plt.axvline(z, color=cores[i], linestyle='--', lw=2, label=f'Z para caso {letra} = {z:.4f}')

plt.title('Teste de Hipóteses Bilateral: H0: μ = 22 vs H1: μ ≠ 22')
plt.xlabel('Valor Z')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)
plt.show()

# Tabela resumo dos resultados
print("\nResumo dos resultados:")
print("-" * 80)
print(f"{'Caso':<5}{'Média':<10}{'Estatística Z':<15}{'Valor-p':<15}{'Decisão':<20}{'Conclusão'}")
print("-" * 80)
for letra, media, z, p, decisao, conclusao in resultados:
    print(f"{letra}. {media:<10}{z:<15.4f}{p:<15.4f}{decisao:<20}{conclusao}")
print("-" * 80)
print(f"Valor crítico para α = {nivel_significancia}: Z = ±{z_critico:.4f}")
print(f"Regra de rejeição: Rejeitar H0 se |Z| > {z_critico:.4f}")

#Cap 9:17

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Dados do problema
salario_medio_reportado = 24.57  # Salário médio reportado para indústrias produtoras de bens (US$/hora)
media_amostral = 23.89  # Média amostral da indústria manufatureira (US$/hora)
desvio_padrao_populacional = 2.40  # Desvio padrão populacional (US$/hora)
tamanho_amostra = 30  # Tamanho da amostra
nivel_significancia = 0.05  # Nível de significância para o teste

def teste_hipoteses_salario():
    """
    Realiza teste de hipóteses para verificar se o salário médio na indústria
    manufatureira difere do salário médio reportado para indústrias produtoras de bens.
    """
    # a) Declaração das hipóteses
    print("a) Hipóteses para o teste:")
    print("   H0: μ = $24.57 (salário médio na indústria manufatureira é igual ao reportado)")
    print("   H1: μ ≠ $24.57 (salário médio na indústria manufatureira é diferente do reportado)")
    print("\n   Justificativa: Como queremos verificar se o salário difere (podendo ser maior ou menor),")
    print("   usamos um teste bilateral.\n")
    
    # b) Cálculo da estatística de teste Z e valor-p
    erro_padrao = desvio_padrao_populacional / np.sqrt(tamanho_amostra)
    z_estatistica = (media_amostral - salario_medio_reportado) / erro_padrao
    
    # Cálculo do valor-p para teste bilateral
    if z_estatistica < 0:
        p_valor = 2 * stats.norm.cdf(z_estatistica)
    else:
        p_valor = 2 * (1 - stats.norm.cdf(z_estatistica))
    
    print(f"b) Estatística de teste: Z = {z_estatistica:.4f}")
    print(f"   Valor-p = {p_valor:.4f}")
    
    # c) Conclusão usando α = 0.05
    print("\nc) Usando α = 0.05:")
    if p_valor < nivel_significancia:
        conclusao = "Rejeitamos H0"
        explicacao = "Há evidência estatística suficiente para concluir que o salário médio na indústria manufatureira é diferente de $24.57 por hora."
    else:
        conclusao = "Não rejeitamos H0"
        explicacao = "Não há evidência estatística suficiente para concluir que o salário médio na indústria manufatureira é diferente de $24.57 por hora."
    
    print(f"   Como valor-p ({p_valor:.4f}) {'<' if p_valor < nivel_significancia else '≥'} α ({nivel_significancia}), {conclusao}.")
    print(f"   {explicacao}")
    
    # d) Teste usando o critério do valor crítico
    z_critico = stats.norm.ppf(1 - nivel_significancia/2)  # Para teste bilateral
    
    print(f"\nd) Usando o critério do valor crítico:")
    print(f"   Valor crítico para α = 0.05 (teste bilateral): Z = ±{z_critico:.4f}")
    print(f"   Regra de rejeição: Rejeitar H0 se |Z| > {z_critico:.4f}")
    
    if abs(z_estatistica) > z_critico:
        conclusao_critico = "Rejeitamos H0"
        explicacao_critico = "Há evidência estatística suficiente para concluir que o salário médio na indústria manufatureira é diferente de $24.57 por hora."
    else:
        conclusao_critico = "Não rejeitamos H0"
        explicacao_critico = "Não há evidência estatística suficiente para concluir que o salário médio na indústria manufatureira é diferente de $24.57 por hora."
    
    print(f"   Como |Z| ({abs(z_estatistica):.4f}) {'>' if abs(z_estatistica) > z_critico else '≤'} {z_critico:.4f}, {conclusao_critico}.")
    print(f"   {explicacao_critico}")
    
    # Retornando os resultados para visualização
    return z_estatistica, p_valor, z_critico, conclusao

def visualizar_teste(z_estatistica, z_critico):
    """
    Cria uma visualização gráfica do teste de hipóteses.
    """
    plt.figure(figsize=(12, 6))
    
    # Definindo o intervalo para plotar a distribuição normal
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)
    
    # Plotando a distribuição normal padrão
    plt.plot(x, y, 'b-', lw=2, label='Distribuição Normal Padrão')
    
    # Áreas de rejeição (bilateral)
    x_rej_esq = np.linspace(-4, -z_critico, 100)
    y_rej_esq = stats.norm.pdf(x_rej_esq)
    plt.fill_between(x_rej_esq, y_rej_esq, color='red', alpha=0.3)
    
    x_rej_dir = np.linspace(z_critico, 4, 100)
    y_rej_dir = stats.norm.pdf(x_rej_dir)
    plt.fill_between(x_rej_dir, y_rej_dir, color='red', alpha=0.3, label='Regiões de rejeição (α=0.05)')
    
    # Marcando os valores críticos
    plt.axvline(-z_critico, color='red', linestyle='--', lw=2, label=f'Valores críticos (±{z_critico:.4f})')
    plt.axvline(z_critico, color='red', linestyle='--', lw=2)
    
    # Marcando o valor Z calculado
    plt.axvline(z_estatistica, color='green', linestyle='--', lw=2, label=f'Z calculado ({z_estatistica:.4f})')
    
    plt.title('Teste de Hipóteses: Salário Médio na Indústria Manufatureira')
    plt.xlabel('Valor Z')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True)
    
    # Adicionando informações adicionais no gráfico
    info_text = (
        f"Salário reportado: ${salario_medio_reportado}/h\n"
        f"Média amostral: ${media_amostral}/h\n"
        f"Tamanho da amostra: {tamanho_amostra}\n"
        f"Desvio padrão: ${desvio_padrao_populacional}/h\n"
        f"α = {nivel_significancia}"
    )
    plt.text(2.5, 0.3, info_text, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Executar o teste de hipóteses
z_estatistica, p_valor, z_critico, conclusao = teste_hipoteses_salario()

# Criar visualização gráfica
visualizar_teste(z_estatistica, z_critico)

#Cap.9:24

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Dados do problema
media_amostral = 17
desvio_padrao_amostral = 4.5
tamanho_amostra = 48
media_hipotese = 18
nivel_significancia = 0.05

# a) Cálculo da estatística de teste t
t_estatistica = (media_amostral - media_hipotese) / (desvio_padrao_amostral / np.sqrt(tamanho_amostra))
print(f"a) Estatística de teste t = {t_estatistica:.4f}")

# b) Calculando o valor-p usando a tabela de distribuição t
# Graus de liberdade = n - 1
graus_liberdade = tamanho_amostra - 1

# Para teste bilateral, o valor-p é:
if t_estatistica < 0:
    p_valor = 2 * stats.t.cdf(t_estatistica, df=graus_liberdade)
else:
    p_valor = 2 * (1 - stats.t.cdf(t_estatistica, df=graus_liberdade))

print(f"b) Valor-p = {p_valor:.4f}")
print(f"   Graus de liberdade = {graus_liberdade}")

# c) Conclusão usando α = 0.05
print("\nc) Usando α = 0.05:")
if p_valor < nivel_significancia:
    conclusao = "Rejeitamos H0"
    explicacao = "Há evidência estatística suficiente para concluir que a média populacional é diferente de 18."
else:
    conclusao = "Não rejeitamos H0"
    explicacao = "Não há evidência estatística suficiente para concluir que a média populacional é diferente de 18."

print(f"   Como valor-p ({p_valor:.4f}) {'<' if p_valor < nivel_significancia else '≥'} α ({nivel_significancia}), {conclusao}.")
print(f"   {explicacao}")

# d) Valor crítico e regra de rejeição
t_critico = stats.t.ppf(1 - nivel_significancia/2, df=graus_liberdade)
print(f"\nd) Valor crítico para α = 0.05 e {graus_liberdade} graus de liberdade: t = ±{t_critico:.4f}")
print(f"   Regra de rejeição: Rejeitar H0 se |t| > {t_critico:.4f}")

if abs(t_estatistica) > t_critico:
    conclusao_criterio = "Rejeitamos H0"
    explicacao_criterio = "Há evidência estatística suficiente para concluir que a média populacional é diferente de 18."
else:
    conclusao_criterio = "Não rejeitamos H0"
    explicacao_criterio = "Não há evidência estatística suficiente para concluir que a média populacional é diferente de 18."

print(f"   Como |t| ({abs(t_estatistica):.4f}) {'>' if abs(t_estatistica) > t_critico else '≤'} {t_critico:.4f}, {conclusao_criterio}.")
print(f"   {explicacao_criterio}")

# Visualização gráfica do teste
plt.figure(figsize=(12, 6))

# Definindo o intervalo para plotar a distribuição t
x = np.linspace(-4, 4, 1000)
y = stats.t.pdf(x, df=graus_liberdade)

# Plotando a distribuição t
plt.plot(x, y, 'b-', lw=2, label=f'Distribuição t ({graus_liberdade} graus de liberdade)')

# Áreas de rejeição
x_rej_esq = np.linspace(-4, -t_critico, 100)
y_rej_esq = stats.t.pdf(x_rej_esq, df=graus_liberdade)
plt.fill_between(x_rej_esq, y_rej_esq, color='red', alpha=0.3)

x_rej_dir = np.linspace(t_critico, 4, 100)
y_rej_dir = stats.t.pdf(x_rej_dir, df=graus_liberdade)
plt.fill_between(x_rej_dir, y_rej_dir, color='red', alpha=0.3, label='Regiões de rejeição (α=0.05)')

# Marcando os valores críticos
plt.axvline(-t_critico, color='red', linestyle='--', lw=2, label=f'Valores críticos (±{t_critico:.4f})')
plt.axvline(t_critico, color='red', linestyle='--', lw=2)

# Marcando o valor t calculado
plt.axvline(t_estatistica, color='green', linestyle='--', lw=2, label=f't calculado ({t_estatistica:.4f})')

plt.title('Teste de Hipóteses: H0: μ = 18 vs H1: μ ≠ 18')
plt.xlabel('Valor t')
plt.ylabel('Densidade')
plt.legend()
plt.grid(True)

# Adicionando informações adicionais no gráfico
info_text = (
    f"Média amostral: {media_amostral}\n"
    f"Desvio padrão amostral: {desvio_padrao_amostral}\n"
    f"Tamanho da amostra: {tamanho_amostra}\n"
    f"Graus de liberdade: {graus_liberdade}\n"
    f"α = {nivel_significancia}"
)
plt.text(2.5, 0.3, info_text, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Resumo final com todos os resultados
print("\nResumo do teste de hipóteses:")
print("-" * 70)
print(f"Hipótese nula (H0): μ = {media_hipotese}")
print(f"Hipótese alternativa (H1): μ ≠ {media_hipotese}")
print(f"Média amostral: {media_amostral}")
print(f"Desvio padrão amostral: {desvio_padrao_amostral}")
print(f"Tamanho da amostra: {tamanho_amostra}")
print(f"Graus de liberdade: {graus_liberdade}")
print(f"Estatística t: {t_estatistica:.4f}")
print(f"Valor-p: {p_valor:.4f}")
print(f"Valor crítico (α = {nivel_significancia}): ±{t_critico:.4f}")
print(f"Conclusão: {conclusao}")
print("-" * 70)

#Cap. 9:32

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Dados do problema
preco_medio_nacional = 10192  # Preço médio nacional de carros usados (US$)
tamanho_amostra = 50  # Número de carros na amostra da revendedora

# Simulando o arquivo UsedCars com dados fictícios (apenas para demonstração)
# Em um caso real, estes dados seriam lidos de um arquivo real
np.random.seed(42)  # Para reprodutibilidade dos resultados
# Gerando preços com média um pouco diferente da nacional para ilustração
precos_amostra = np.random.normal(loc=9800, scale=1200, size=tamanho_amostra)
precos_amostra = np.round(precos_amostra, 2)  # Arredondando para 2 casas decimais

# Criando um DataFrame como se fosse o arquivo UsedCars
df = pd.DataFrame({'Preco': precos_amostra})

# Calculando estatísticas da amostra
media_amostral = df['Preco'].mean()
desvio_padrao_amostral = df['Preco'].std(ddof=1)  # ddof=1 para desvio padrão amostral

def teste_hipoteses_carros_usados():
    """
    Realiza teste de hipóteses para verificar se o preço médio dos carros usados
    na revendedora de Kansas City difere da média nacional.
    """
    print("Análise de Preços de Carros Usados - Kansas City vs. Média Nacional")
    print("=" * 70)
    
    # a) Formulação das hipóteses
    print("a) Hipóteses para o teste:")
    print("   H0: μ = $10,192 (o preço médio na revendedora é igual à média nacional)")
    print("   H1: μ ≠ $10,192 (o preço médio na revendedora é diferente da média nacional)")
    print("\n   Justificativa: Como queremos determinar se existe uma diferença no preço")
    print("   médio (podendo ser maior ou menor), usamos um teste bilateral.\n")
    
    # b) Cálculo da estatística de teste t e valor-p
    erro_padrao = desvio_padrao_amostral / np.sqrt(tamanho_amostra)
    t_estatistica = (media_amostral - preco_medio_nacional) / erro_padrao
    
    # Graus de liberdade
    gl = tamanho_amostra - 1
    
    # Cálculo do valor-p para teste bilateral
    if t_estatistica < 0:
        p_valor = 2 * stats.t.cdf(t_estatistica, df=gl)
    else:
        p_valor = 2 * (1 - stats.t.cdf(t_estatistica, df=gl))
    
    print(f"b) Estatística de teste: t = {t_estatistica:.4f}")
    print(f"   Graus de liberdade: {gl}")
    print(f"   Valor-p = {p_valor:.4f}")
    
    # c) Conclusão usando α = 0.05
    alpha = 0.05
    print(f"\nc) Com α = {alpha}:")
    if p_valor < alpha:
        conclusao = "Rejeitamos H0"
        explicacao = "Há evidência estatística suficiente para concluir que o preço médio dos carros usados na revendedora de Kansas City é diferente da média nacional de $10,192."
    else:
        conclusao = "Não rejeitamos H0"
        explicacao = "Não há evidência estatística suficiente para concluir que o preço médio dos carros usados na revendedora de Kansas City é diferente da média nacional de $10,192."
    
    print(f"   Como valor-p ({p_valor:.4f}) {'<' if p_valor < alpha else '≥'} α ({alpha}), {conclusao}.")
    print(f"   {explicacao}")
    
    # Valor crítico e região de rejeição
    t_critico = stats.t.ppf(1 - alpha/2, df=gl)
    print(f"\n   Valor crítico para α = {alpha} (teste bilateral): t = ±{t_critico:.4f}")
    print(f"   Regra de rejeição: Rejeitar H0 se |t| > {t_critico:.4f}")
    
    if abs(t_estatistica) > t_critico:
        conclusao_criterio = "rejeitamos H0"
    else:
        conclusao_criterio = "não rejeitamos H0"
    
    print(f"   Como |t| ({abs(t_estatistica):.4f}) {'>' if abs(t_estatistica) > t_critico else '≤'} {t_critico:.4f}, {conclusao_criterio}.")
    
    return t_estatistica, p_valor, gl, conclusao

def visualizar_resultados(t_estatistica, gl):
    """
    Cria visualizações para melhor compreensão dos resultados.
    """
    plt.figure(figsize=(14, 8))
    
    # Gráfico 1: Distribuição dos preços da amostra
    plt.subplot(2, 1, 1)
    plt.hist(df['Preco'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(media_amostral, color='red', linestyle='--', linewidth=2, label=f'Média Amostral: ${media_amostral:.2f}')
    plt.axvline(preco_medio_nacional, color='green', linestyle='-', linewidth=2, label=f'Média Nacional: ${preco_medio_nacional}')
    plt.title('Distribuição dos Preços dos Carros Usados na Amostra')
    plt.xlabel('Preço (US$)')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Visualização do teste t
    plt.subplot(2, 1, 2)
    
    # Definindo o intervalo para plotar a distribuição t
    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df=gl)
    
    # Plotando a distribuição t
    plt.plot(x, y, 'b-', lw=2, label=f'Distribuição t ({gl} graus de liberdade)')
    
    # Valor crítico para α = 0.05 bilateral
    t_critico = stats.t.ppf(0.975, df=gl)
    
    # Áreas de rejeição
    x_rej_esq = np.linspace(-4, -t_critico, 100)
    y_rej_esq = stats.t.pdf(x_rej_esq, df=gl)
    plt.fill_between(x_rej_esq, y_rej_esq, color='red', alpha=0.3)
    
    x_rej_dir = np.linspace(t_critico, 4, 100)
    y_rej_dir = stats.t.pdf(x_rej_dir, df=gl)
    plt.fill_between(x_rej_dir, y_rej_dir, color='red', alpha=0.3, label='Regiões de rejeição (α=0.05)')
    
    # Marcando os valores críticos
    plt.axvline(-t_critico, color='red', linestyle='--', lw=2, label=f'Valores críticos (±{t_critico:.4f})')
    plt.axvline(t_critico, color='red', linestyle='--', lw=2)
    
    # Marcando o valor t calculado
    plt.axvline(t_estatistica, color='green', linestyle='--', lw=2, label=f't calculado ({t_estatistica:.4f})')
    
    plt.title('Teste de Hipóteses: Preço Médio dos Carros Usados')
    plt.xlabel('Valor t')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Executar o teste de hipóteses
t_estatistica, p_valor, gl, conclusao = teste_hipoteses_carros_usados()

# Visualizar os resultados
visualizar_resultados(t_estatistica, gl)

# Exibir resumo final
print("\nResumo dos dados:")
print(f"Média nacional: ${preco_medio_nacional}")
print(f"Média amostral na revendedora: ${media_amostral:.2f}")
print(f"Desvio padrão amostral: ${desvio_padrao_amostral:.2f}")
print(f"Tamanho da amostra: {tamanho_amostra}")