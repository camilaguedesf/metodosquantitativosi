
#Cap10:27

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns

# Definindo os dados da tabela
cargos = ['Diretores de arte', 'Astrônomos', 'Audiologistas', 'Higienistas dentais', 
          'Economistas', 'Engenheiros', 'Professores de Direito', 'Optometristas', 
          'Cientistas políticos', 'Planejadores urbanos e regionais']

salarios = np.array([81, 96, 70, 70, 92, 92, 100, 98, 102, 65])
tolerancia_estresse = np.array([69.0, 62.0, 67.5, 71.3, 63.3, 69.5, 62.8, 65.5, 60.1, 69.0])

# Criando um DataFrame para melhor visualização
df = pd.DataFrame({
    'Cargo': cargos,
    'Salário Médio Anual (US$ 1.000)': salarios,
    'Tolerância ao Estresse': tolerancia_estresse
})

print("Dados originais:")
print(df)
print("\n")

# a. Desenvolva um diagrama de dispersão
plt.figure(figsize=(10, 6))
sns.scatterplot(x=salarios, y=tolerancia_estresse)

# Adicionando rótulos aos pontos
for i, cargo in enumerate(cargos):
    plt.annotate(cargo, (salarios[i], tolerancia_estresse[i]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.title('Diagrama de Dispersão: Salário Médio Anual vs. Tolerância ao Estresse')
plt.xlabel('Salário Médio Anual (US$ 1.000)')
plt.ylabel('Tolerância ao Estresse (0-100)')
plt.grid(True, linestyle='--', alpha=0.7)

# b. Equação de regressão estimada
slope, intercept, r_value, p_value, std_err = stats.linregress(salarios, tolerancia_estresse)

# Plotando a linha de regressão
x_line = np.linspace(min(salarios) - 5, max(salarios) + 5, 100)
y_line = intercept + slope * x_line
plt.plot(x_line, y_line, color='red', label=f'y = {intercept:.2f} + {slope:.4f}x')
plt.legend()

print("b. Equação de regressão estimada:")
print(f"Tolerância ao Estresse = {intercept:.2f} + {slope:.4f} × Salário Médio Anual")
print(f"Coeficiente de determinação (R²): {r_value**2:.4f}")
print("\n")

# c. Análise da significância estatística
print("c. Análise da significância estatística:")
print(f"Valor-p: {p_value:.6f}")
nivel_significancia = 0.05
if p_value < nivel_significancia:
    print(f"Como o valor-p ({p_value:.6f}) é menor que o nível de significância (0.05), ")
    print("rejeitamos a hipótese nula. Há uma diferença estatística significativa entre as duas variáveis.")
else:
    print(f"Como o valor-p ({p_value:.6f}) é maior ou igual ao nível de significância (0.05), ")
    print("não rejeitamos a hipótese nula. Não há evidência de diferença estatística significativa entre as duas variáveis.")
print("\n")

# d. Previsão para uma ocupação hipotética
print("d. Previsão para uma ocupação diferente:")
salario_hipotetico = 85  # Exemplo de salário para previsão
previsao = intercept + slope * salario_hipotetico
print(f"Para um salário médio anual de US$ {salario_hipotetico}.000:")
print(f"A tolerância ao estresse prevista seria de {previsao:.2f} (numa escala de 0-100)")
print("\n")

# e. Análise da relação entre as variáveis
print("e. Análise da relação entre o salário médio anual e a tolerância ao estresse:")
print(f"Correlação de Pearson (r): {r_value:.4f}")

if abs(r_value) < 0.3:
    intensidade = "fraca"
elif abs(r_value) < 0.7:
    intensidade = "moderada"
else:
    intensidade = "forte"

if r_value > 0:
    direcao = "positiva"
else:
    direcao = "negativa"

print(f"Existe uma correlação {intensidade} e {direcao} entre as variáveis.")
print(f"Isso significa que {direcao == 'positiva' and 'quanto maior' or 'quanto menor'} o salário médio anual, ", end="")
print(f"{direcao == 'positiva' and 'maior' or 'menor'} tende a ser a tolerância ao estresse.")

# Para uma análise mais completa, vamos adicionar algumas estatísticas descritivas
print("\nEstatísticas descritivas:")
print("Salário médio anual:")
print(f"  Média: {np.mean(salarios):.2f}")
print(f"  Desvio padrão: {np.std(salarios, ddof=1):.2f}")
print(f"  Mínimo: {np.min(salarios)}")
print(f"  Máximo: {np.max(salarios)}")

print("\nTolerância ao estresse:")
print(f"  Média: {np.mean(tolerancia_estresse):.2f}")
print(f"  Desvio padrão: {np.std(tolerancia_estresse, ddof=1):.2f}")
print(f"  Mínimo: {np.min(tolerancia_estresse):.1f}")
print(f"  Máximo: {np.max(tolerancia_estresse):.1f}")

plt.tight_layout()
plt.show()

Cap10:35

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

# Dados da tabela
gpa = np.array([2.6, 3.4, 3.6, 3.2, 3.5, 2.9])
salario = np.array([3600, 3900, 4300, 3800, 4200, 3900])

# Criando DataFrame para visualização
df = pd.DataFrame({
    'GPA': gpa,
    'Salário Mensal (US$)': salario
})

print("Dados originais:")
print(df)
print("\n")

# Parâmetros da equação de regressão já fornecidos:
# y = 2090.5 + 581.1x onde QMRes = 21284
intercepto = 2090.5
coef_angular = 581.1
qm_res = 21284  # QMRes - Quadrado Médio dos Resíduos

# Número de observações
n = len(gpa)

# a. Estimativa pontual do salário inicial para um estudante com uma GPA de 3,0
gpa_novo = 3.0
salario_estimado = intercepto + coef_angular * gpa_novo

print("a. Estimativa pontual do salário inicial para GPA = 3,0:")
print(f"   Salário estimado = {salario_estimado:.2f} US$")
print("\n")

# b. Intervalo de confiança de 95% para o salário inicial médio
# Cálculo da média de GPA
gpa_medio = np.mean(gpa)

# Soma dos quadrados da diferença entre cada GPA e a média de GPA
sxx = np.sum((gpa - gpa_medio) ** 2)

# Erro padrão da média estimada
se_medio = np.sqrt(qm_res * (1/n + (gpa_novo - gpa_medio)**2 / sxx))

# Graus de liberdade
gl = n - 2

# Valor crítico t para 95% de confiança
t_critico = stats.t.ppf(0.975, gl)  # 0.975 para um intervalo bicaudal de 95%

# Intervalo de confiança
ic_inferior = salario_estimado - t_critico * se_medio
ic_superior = salario_estimado + t_critico * se_medio

print("b. Intervalo de confiança de 95% para o salário inicial médio com GPA = 3,0:")
print(f"   Erro padrão da média: {se_medio:.2f}")
print(f"   Valor crítico t (95%, {gl} gl): {t_critico:.4f}")
print(f"   Intervalo de confiança: [{ic_inferior:.2f}, {ic_superior:.2f}] US$")
print("\n")

# c. Intervalo de previsão de 95% para um estudante específico (Ryan Dailey) com GPA = 3,0
# Erro padrão para uma nova observação
se_previsao = np.sqrt(qm_res * (1 + 1/n + (gpa_novo - gpa_medio)**2 / sxx))

# Intervalo de previsão
ip_inferior = salario_estimado - t_critico * se_previsao
ip_superior = salario_estimado + t_critico * se_previsao

print("c. Intervalo de previsão de 95% para Ryan Dailey com GPA = 3,0:")
print(f"   Erro padrão da previsão: {se_previsao:.2f}")
print(f"   Intervalo de previsão: [{ip_inferior:.2f}, {ip_superior:.2f}] US$")
print("\n")

# d. Análise das diferenças entre os intervalos (b) e (c)
diferenca_amplitude = (ip_superior - ip_inferior) - (ic_superior - ic_inferior)

print("d. Análise das diferenças entre os intervalos (b) e (c):")
print(f"   Amplitude do intervalo de confiança: {ic_superior - ic_inferior:.2f} US$")
print(f"   Amplitude do intervalo de previsão: {ip_superior - ip_inferior:.2f} US$")
print(f"   Diferença de amplitude: {diferenca_amplitude:.2f} US$")
print("   O intervalo de previsão é mais amplo que o intervalo de confiança.")
print("   Isso ocorre porque o intervalo de previsão considera a variabilidade")
print("   adicional de uma observação individual, enquanto o intervalo de confiança")
print("   estima apenas a variabilidade da média.")
print("\n")

# Visualização gráfica da regressão e intervalos
plt.figure(figsize=(10, 6))

# Scatter plot dos dados originais
sns.scatterplot(x=gpa, y=salario, color='blue', label='Dados originais')

# Linha de regressão
x_line = np.linspace(min(gpa) - 0.2, max(gpa) + 0.2, 100)
y_line = intercepto + coef_angular * x_line
plt.plot(x_line, y_line, color='red', label=f'Regressão: y = {intercepto:.1f} + {coef_angular:.1f}x')

# Ponto de estimativa para GPA = 3.0
plt.scatter([gpa_novo], [salario_estimado], color='green', s=100, marker='*', 
            label=f'Estimativa para GPA = {gpa_novo} (US$ {salario_estimado:.2f})')

# Intervalos
plt.errorbar(gpa_novo, salario_estimado, 
             yerr=[[salario_estimado - ic_inferior], [ic_superior - salario_estimado]],
             fmt='o', color='purple', capsize=10, label='IC 95% para média')

plt.errorbar(gpa_novo + 0.05, salario_estimado, 
             yerr=[[salario_estimado - ip_inferior], [ip_superior - salario_estimado]],
             fmt='o', color='orange', capsize=10, label='IP 95% para Ryan')

plt.title('Regressão Linear: GPA vs Salário Mensal')
plt.xlabel('GPA')
plt.ylabel('Salário Mensal (US$)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Mostrar o gráfico
plt.show()

#Cap14:42

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns

# Dados da tabela ANOVA
df_regression = 1
df_error = 28
df_total = 29

ss_regression = 6828.6
ss_error = 2298.8
ss_total = 9127.4

ms_regression = 6828.6
ms_error = 82.1

# Coeficientes da regressão
constante = 80.0
coef_x = 50.0
se_constante = 11.333
se_coef_x = 5.482
t_constante = 7.06
t_coef_x = 9.12

# a. Equação de regressão estimada
print("a. Equação de regressão estimada:")
print(f"   Y = {constante} + {coef_x}X")
print("   onde Y é vendas anuais (em milhares de dólares) e X é o número de vendedores")
print("\n")

# b. Número de escritórios de filiais envolvidos no estudo
print("b. Número de escritórios de filiais envolvidos no estudo:")
print(f"   Graus de liberdade total = {df_total}")
print(f"   Número de observações = {df_total + 1}")
print(f"   Portanto, {df_total + 1} escritórios de filiais estavam envolvidos no estudo.")
print("\n")

# c. Cálculo da estatística F e teste de significância
f_stat = ms_regression / ms_error
p_value = 1 - stats.f.cdf(f_stat, df_regression, df_error)

print("c. Cálculo da estatística F e teste de significância:")
print(f"   Estatística F = MS(Regressão) / MS(Erro) = {ms_regression} / {ms_error} = {f_stat:.4f}")
print(f"   Valor-p = {p_value:.10f}")
print(f"   Valor crítico F(1, 28) para α=0.05 = {stats.f.ppf(0.95, df_regression, df_error):.4f}")

if p_value < 0.05:
    print("   Como o valor-p é menor que 0.05, rejeitamos a hipótese nula.")
    print("   Existe uma relação estatisticamente significativa entre o número de vendedores e as vendas anuais.")
else:
    print("   Como o valor-p é maior ou igual a 0.05, não rejeitamos a hipótese nula.")
    print("   Não há evidência de uma relação estatisticamente significativa.")
print("\n")

# d. Previsão das vendas anuais para a filial em Memphis com 12 vendedores
x_memphis = 12
y_previsto = constante + coef_x * x_memphis

print("d. Previsão das vendas anuais para a filial em Memphis (12 vendedores):")
print(f"   Y previsto = {constante} + {coef_x} × {x_memphis}")
print(f"   Y previsto = {constante} + {coef_x * x_memphis}")
print(f"   Vendas anuais previstas = {y_previsto} milhares de dólares (US$ {y_previsto*1000:.2f})")
print("\n")

# Cálculo do coeficiente de determinação (R²)
r_quadrado = ss_regression / ss_total
r_quadrado_ajustado = 1 - (ss_error / df_error) / (ss_total / df_total)

print("Informações adicionais:")
print(f"Coeficiente de determinação (R²) = {r_quadrado:.4f}")
print(f"R² ajustado = {r_quadrado_ajustado:.4f}")
print(f"Isto significa que {r_quadrado*100:.2f}% da variação nas vendas anuais é explicada pelo número de vendedores.")
print("\n")

# Visualização da regressão
plt.figure(figsize=(10, 6))

# Como não temos os dados originais, vamos simular alguns pontos para ilustrar
# Criando um intervalo de possíveis valores X (número de vendedores)
x_range = np.linspace(0, 20, 30)

# Criando valores Y correspondentes com base na equação de regressão
y_predicted = constante + coef_x * x_range

# Adicionando um ruído aleatório para simular a dispersão dos dados reais
np.random.seed(42)  # Para reprodutibilidade
y_simulated = y_predicted + np.random.normal(0, np.sqrt(ms_error), len(x_range))

# Plotando os pontos simulados
plt.scatter(x_range, y_simulated, alpha=0.7, label='Dados simulados')

# Linha de regressão
plt.plot(x_range, y_predicted, color='red', label=f'Y = {constante} + {coef_x}X')

# Destacando Memphis
plt.scatter([x_memphis], [y_previsto], color='green', s=100, marker='*', 
            label=f'Memphis (12 vendedores, {y_previsto} mil US$)')

plt.title('Regressão Linear: Número de Vendedores vs Vendas Anuais')
plt.xlabel('Número de Vendedores (X)')
plt.ylabel('Vendas Anuais (Y) em Milhares de US$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Criando uma tabela ANOVA para visualização}

anova_table = pd.DataFrame(anova_data)
print("Tabela ANOVA:")
print(anova_table.to_string(index=False))

plt.show()


#Questão 66

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

# Dados dos retornos trimestrais (diferença entre o retorno percentual e o retorno livre de riscos)
sp500 = np.array([1.2, -2.5, -3.0, 2.0, 5.0, 1.2, 3.0, -1.0, 0.5, 2.5])
horizon = np.array([-7, -2.0, -5.5, 4.7, 1.8, 4.1, 2.6, 2.0, -1.3, 5.5])

# Criando DataFrame para visualização
df = pd.DataFrame({
    'S&P 500': sp500,
    'Horizon Technology': horizon
})

print("Dados originais:")
print(df)
print("\n")

# a. Equação de regressão estimada para prever o beta de mercado da Horizon Technology
# O beta de mercado é o coeficiente angular (b1) da regressão linear entre os retornos

# Realizando a regressão linear
slope, intercept, r_value, p_value, std_err = stats.linregress(sp500, horizon)

# O beta de mercado da Horizon Technology é o valor do coeficiente angular (slope)
beta_horizon = slope

print("a. Equação de regressão estimada e beta de mercado:")
print(f"   Equação: Horizon = {intercept:.4f} + {beta_horizon:.4f} × S&P 500")
print(f"   Beta de mercado da Horizon Technology = {beta_horizon:.4f}")
print("\n")

# b. Teste quanto a uma relação significativa (α = 0.05)
print("b. Teste de significância da relação:")
print(f"   Valor do coeficiente de correlação (r): {r_value:.4f}")
print(f"   Coeficiente de determinação (R²): {r_value**2:.4f}")
print(f"   Valor-p: {p_value:.6f}")

# Teste de hipótese
alpha = 0.05
if p_value < alpha:
    print(f"   Como o valor-p ({p_value:.6f}) é menor que o nível de significância ({alpha}),")
    print("   rejeitamos a hipótese nula. Há evidência estatística de uma relação significativa.")
else:
    print(f"   Como o valor-p ({p_value:.6f}) é maior ou igual ao nível de significância ({alpha}),")
    print("   não rejeitamos a hipótese nula. Não há evidência estatística de uma relação significativa.")
print("\n")

# c. Análise do ajuste da regressão
print("c. Análise do ajuste da equação de regressão:")
print(f"   Coeficiente de determinação (R²): {r_value**2:.4f}")

# Avaliação do ajuste
if r_value**2 >= 0.8:
    ajuste = "excelente"
elif r_value**2 >= 0.6:
    ajuste = "bom"
elif r_value**2 >= 0.4:
    ajuste = "moderado"
else:
    ajuste = "fraco"

print(f"   O modelo tem um ajuste {ajuste}, explicando {r_value**2*100:.2f}% da variação nos retornos da Horizon Technology.")

# Cálculo do erro padrão da estimativa
y_pred = intercept + slope * sp500
residuos = horizon - y_pred
erro_padrao_estimativa = np.sqrt(np.sum(residuos**2) / (len(sp500) - 2))
print(f"   Erro padrão da estimativa: {erro_padrao_estimativa:.4f}")
print("\n")

# d. Comparação do risco entre Xerox e Horizon Technology
# Como não temos os dados da Xerox, vamos comparar apenas o beta da Horizon com o benchmark (1.0)
print("d. Comparação do risco:")
print(f"   Beta de mercado da Horizon Technology: {beta_horizon:.4f}")
if beta_horizon > 1:
    print("   Como o beta da Horizon Technology é maior que 1, isso indica que a ação é mais volátil")
    print("   do que a média do mercado (S&P 500). Para cada movimento de 1% no S&P 500,")
    print(f"   espera-se que a Horizon Technology se mova aproximadamente {beta_horizon:.2f}%.")
elif beta_horizon < 1:
    print("   Como o beta da Horizon Technology é menor que 1, isso indica que a ação é menos volátil")
    print("   do que a média do mercado (S&P 500). Para cada movimento de 1% no S&P 500,")
    print(f"   espera-se que a Horizon Technology se mova aproximadamente {beta_horizon:.2f}%.")
else:
    print("   O beta da Horizon Technology é igual a 1, indicando que a volatilidade da ação")
    print("   é similar à média do mercado (S&P 500).")
print("\n")

# Visualização da regressão
plt.figure(figsize=(10, 6))

# Scatter plot dos dados originais
sns.scatterplot(x=sp500, y=horizon, color='blue', label='Dados Observados')

# Linha de regressão
x_line = np.linspace(min(sp500) - 1, max(sp500) + 1, 100)
y_line = intercept + slope * x_line
plt.plot(x_line, y_line, color='red', label=f'Beta = {beta_horizon:.4f}\nY = {intercept:.4f} + {beta_horizon:.4f}X')

# Adicionar a referência do beta = 1 (mesma volatilidade que o mercado)
y_ref = 1 * x_line
plt.plot(x_line, y_ref, color='green', linestyle='--', label='Beta = 1')

plt.title('Regressão Linear: S&P 500 vs Horizon Technology')
plt.xlabel('Retorno do S&P 500 (X)')
plt.ylabel('Retorno da Horizon Technology (Y)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.legend()
plt.tight_layout()

# Mostrar o gráfico
plt.show()