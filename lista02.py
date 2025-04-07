# lista 2

'''
Descrição:
Exercícios dos capítulos 4, 5 e 6 do livro Estatística Aplicada a Administração e Economia.
Cap. 4: 10, 20, 28, 33, 38, 42
Cap. 5: 3, 8, 18, 27, 34
Cap. 6: 2, 10, 16, 22, 24, 40

'''
#Cap4:10

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np

# Criando o DataFrame com os dados da tabela
data = {
    'Companhia': ['Virgin America', 'JetBlue', 'AirTran Airways', 'Delta Air Lines', 
                  'Alaska Airlines', 'Frontier Airlines', 'Southwest Airlines', 
                  'US Airways', 'American Airlines', 'United Airlines'],
    'ChegadasHorario': [83.5, 79.1, 87.1, 86.5, 87.5, 77.9, 83.1, 85.9, 76.9, 77.4],
    'BagagensExtraviadas': [0.87, 1.88, 1.58, 2.10, 2.93, 2.22, 3.08, 2.14, 2.92, 3.87],
    'ReclamacoesClientes': [1.50, 0.79, 0.91, 0.73, 0.51, 1.05, 0.25, 1.74, 1.30, 4.24]
}

df = pd.DataFrame(data)

# a. Probabilidade de um voo da Delta Air Lines chegar no horário
def questao_a():
    prob_delta = df[df['Companhia'] == 'Delta Air Lines']['ChegadasHorario'].values[0] / 100
    return prob_delta

# b. Probabilidade de escolher uma companhia com menos de dois relatórios de bagagem extraviada
def questao_b():
    companhias_menos_2_bagagens = df[df['BagagensExtraviadas'] < 2]
    prob = len(companhias_menos_2_bagagens) / len(df)
    return prob

# c. Probabilidade de escolher uma companhia com mais de uma reclamação de cliente
def questao_c():
    companhias_mais_1_reclamacao = df[df['ReclamacoesClientes'] > 1]
    prob = len(companhias_mais_1_reclamacao) / len(df)
    return prob

# d. Probabilidade de um voo da AirTran Airways não chegar no horário
def questao_d():
    prob_airtran_no_horario = df[df['Companhia'] == 'AirTran Airways']['ChegadasHorario'].values[0] / 100
    prob_airtran_nao_horario = 1 - prob_airtran_no_horario
    return prob_airtran_nao_horario

# Executando e mostrando os resultados
print("a. Probabilidade de um voo da Delta Air Lines chegar no horário:", questao_a())
print("b. Probabilidade de escolher uma companhia com menos de dois relatórios de bagagem extraviada:", questao_b())
print("c. Probabilidade de escolher uma companhia com mais de uma reclamação de cliente:", questao_c())
print("d. Probabilidade de um voo da AirTran Airways não chegar no horário:", questao_d())

#Cap4:20

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Criando o DataFrame com os dados da tabela
data = {
    'Idade': ['16 a 20', '21 a 24', '25 a 27', '28 ou mais'],
    'Numero_Respostas': [191, 467, 244, 42]
}

df = pd.DataFrame(data)

# Calculando o total de respostas
total_respostas = df['Numero_Respostas'].sum()
print(f"Total de respostas: {total_respostas}")

# a. Probabilidade de ser financeiramente independente para cada categoria de idade
def questao_a():
    df['Probabilidade'] = df['Numero_Respostas'] / total_respostas
    print("\na. Probabilidade para cada categoria de idade:")
    for i, row in df.iterrows():
        print(f"   {row['Idade']}: {row['Probabilidade']:.4f} ou {row['Probabilidade']*100:.2f}%")
    return df['Probabilidade'].tolist()

# b. Probabilidade de ser financeiramente independente antes dos 25 anos
def questao_b():
    antes_25 = df[df['Idade'].isin(['16 a 20', '21 a 24'])]['Numero_Respostas'].sum()
    prob_antes_25 = antes_25 / total_respostas
    print(f"\nb. Probabilidade de ser financeiramente independente antes dos 25 anos: {prob_antes_25:.4f} ou {prob_antes_25*100:.2f}%")
    return prob_antes_25

# c. Probabilidade de ser financeiramente independente após os 24 anos
def questao_c():
    apos_24 = df[df['Idade'].isin(['25 a 27', '28 ou mais'])]['Numero_Respostas'].sum()
    prob_apos_24 = apos_24 / total_respostas
    print(f"\nc. Probabilidade de ser financeiramente independente após os 24 anos: {prob_apos_24:.4f} ou {prob_apos_24*100:.2f}%")
    return prob_apos_24

# d. Análise sobre o realismo das expectativas
def questao_d():
    print("\nd. Análise sobre o realismo das expectativas:")
    
    # Calculando a idade média estimada (aproximadamente)
    idades_medias = {
        '16 a 20': 18,  # média aproximada da faixa
        '21 a 24': 22.5,
        '25 a 27': 26,
        '28 ou mais': 30  # valor arbitrário para representar essa faixa
    }
    
    df['Idade_Media'] = df['Idade'].map(idades_medias)
    idade_media_ponderada = sum(df['Idade_Media'] * df['Numero_Respostas']) / total_respostas
    
    print(f"   A idade média aproximada em que os adolescentes acreditam que serão independentes: {idade_media_ponderada:.2f} anos")
    print("   Análise: A maioria dos adolescentes (aproximadamente {:.2f}%) espera ser financeiramente independente entre 21 e 24 anos.".format(df.loc[1, 'Probabilidade']*100))
    print("   Isso pode ser considerado otimista, dado que muitos jovens nessa faixa etária")
    print("   ainda estão na faculdade ou iniciando suas carreiras, com salários iniciais mais baixos.")
    
    # Criando um gráfico para visualizar as respostas
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Idade'], df['Numero_Respostas'], color='skyblue')
    plt.title('Expectativa de Idade para Independência Financeira')
    plt.xlabel('Faixa Etária')
    plt.ylabel('Número de Respostas')
    
    # Adicionar valores em cima das barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}', ha='center', va='bottom')
    
    # Exibir o gráfico (comentado para não interromper a execução)
    # plt.show()
    
    return idade_media_ponderada

# Executando todas as funções
questao_a()
questao_b()
questao_c()
questao_d()

#cap4:28

# Importando bibliotecas necessárias
import numpy as np

# Dados do problema
comercial = 0.458  # 45,8% alugaram por razões comerciais
pessoal = 0.54  # 54% alugaram por razões pessoais
ambos = 0.30  # 30% alugaram por ambas as razões

# a. Probabilidade de um assinante ter alugado um carro por razões comerciais OU pessoais
def questao_a():
    # Probabilidade da união (A ou B) = P(A) + P(B) - P(A e B)
    # onde A = alugou por razões comerciais e B = alugou por razões pessoais
    probabilidade = comercial + pessoal - ambos
    return probabilidade

# b. Probabilidade de um assinante NÃO ter alugado um carro por razões comerciais ou pessoais
def questao_b():
    # Primeiro calculamos a probabilidade de ter alugado (já feito na questão a)
    prob_alugou = questao_a()
    
    # Probabilidade de NÃO ter alugado = 1 - probabilidade de ter alugado
    probabilidade = 1 - prob_alugou
    return probabilidade

# Calculando e exibindo os resultados
print("Dados do problema:")
print(f"- Assinantes que alugaram carro por razões comerciais: {comercial*100:.1f}%")
print(f"- Assinantes que alugaram carro por razões pessoais: {pessoal*100:.1f}%")
print(f"- Assinantes que alugaram carro por ambas as razões: {ambos*100:.1f}%")
print()

# Resultados das questões
prob_a = questao_a()
prob_b = questao_b()

print(f"a. Probabilidade de um assinante ter alugado um carro por razões comerciais OU pessoais: {prob_a:.4f} ou {prob_a*100:.2f}%")
print(f"b. Probabilidade de um assinante NÃO ter alugado um carro: {prob_b:.4f} ou {prob_b*100:.2f}%")

# Verificação adicional
print("\nVerificação:")
print(f"Soma das probabilidades (alugou + não alugou): {prob_a + prob_b}")

#Cap.4:33

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np

# Criando o DataFrame com os dados da tabela
data = {
    'Status Matrícula': ['Período Integral', 'Meio Período', 'Totais'],
    'Administração': [352, 150, 502],
    'Engenharia': [197, 161, 358],
    'Outros': [251, 194, 445],
    'Totais': [800, 505, 1305]
}

df = pd.DataFrame(data)

# Exibindo a tabela
print("Tabela original:")
print(df)
print("\n")

# a. Desenvolver tabela de probabilidades conjuntas
def questao_a():
    print("a. Tabela de probabilidades conjuntas:")
    prob_df = pd.DataFrame(index=df['Status Matrícula'], columns=df.columns[1:])
    
    total_geral = df.loc[2, 'Totais']  # Total de estudantes (1305)
    
    for i in range(3):  # Linhas
        for j in range(1, 5):  # Colunas
            prob_df.iloc[i, j-1] = df.iloc[i, j] / total_geral
    
    print(prob_df)
    return prob_df

# b. Probabilidades marginais por setor de graduação
def questao_b():
    print("\nb. Probabilidades marginais por setor de graduação:")
    
    total_geral = df.loc[2, 'Totais']  # Total de estudantes (1305)
    
    prob_adm = df.loc[2, 'Administração'] / total_geral
    prob_eng = df.loc[2, 'Engenharia'] / total_geral
    prob_outros = df.loc[2, 'Outros'] / total_geral
    
    print(f"Probabilidade de ser graduado em Administração: {prob_adm:.4f} ou {prob_adm*100:.2f}%")
    print(f"Probabilidade de ser graduado em Engenharia: {prob_eng:.4f} ou {prob_eng*100:.2f}%")
    print(f"Probabilidade de ser graduado em Outros cursos: {prob_outros:.4f} ou {prob_outros*100:.2f}%")
    
    # Verificando qual setor tem maior potencial
    if prob_adm > prob_eng and prob_adm > prob_outros:
        print("O setor de Administração tem o maior potencial de estudantes de MBA.")
    elif prob_eng > prob_adm and prob_eng > prob_outros:
        print("O setor de Engenharia tem o maior potencial de estudantes de MBA.")
    else:
        print("O setor de Outros cursos tem o maior potencial de estudantes de MBA.")
    
    return prob_adm, prob_eng, prob_outros

# c. Probabilidade de ser graduado em Engenharia e frequentar MBA em tempo integral
def questao_c():
    total_geral = df.loc[2, 'Totais']  # Total de estudantes (1305)
    eng_tempo_integral = df.loc[0, 'Engenharia']  # Engenharia em tempo integral (197)
    
    prob = eng_tempo_integral / total_geral
    
    print(f"\nc. Probabilidade de ser graduado em Engenharia e frequentar MBA em tempo integral: {prob:.4f} ou {prob*100:.2f}%")
    return prob

# d. Probabilidade de frequentar meio período sendo da Administração
def questao_d():
    total_adm = df.loc[2, 'Administração']  # Total de estudantes de Administração (502)
    adm_meio_periodo = df.loc[1, 'Administração']  # Administração em meio período (150)
    
    prob = adm_meio_periodo / total_adm
    
    print(f"\nd. Probabilidade de frequentar meio período sendo da Administração: {prob:.4f} ou {prob*100:.2f}%")
    return prob

# e. Verificar independência dos eventos
def questao_e():
    print("\ne. Verificação de independência dos eventos:")
    
    total_geral = df.loc[2, 'Totais']  # Total de estudantes (1305)
    
    # Probabilidade de frequentar em tempo integral
    prob_integral = df.loc[0, 'Totais'] / total_geral
    
    # Probabilidade de ser graduado em Administração
    prob_adm = df.loc[2, 'Administração'] / total_geral
    
    # Probabilidade de frequentar em tempo integral E ser graduado em Administração
    prob_integral_adm = df.loc[0, 'Administração'] / total_geral
    
    # Se os eventos são independentes, P(A e B) = P(A) * P(B)
    prob_se_independente = prob_integral * prob_adm
    
    print(f"P(Tempo Integral) = {prob_integral:.4f}")
    print(f"P(Administração) = {prob_adm:.4f}")
    print(f"P(Tempo Integral e Administração) = {prob_integral_adm:.4f}")
    print(f"P(Tempo Integral) * P(Administração) = {prob_se_independente:.4f}")
    
    if abs(prob_integral_adm - prob_se_independente) < 0.001:
        print("Os eventos são independentes, pois P(A e B) ≈ P(A) * P(B)")
    else:
        print("Os eventos NÃO são independentes, pois P(A e B) ≠ P(A) * P(B)")
        
        # Verificar se são positivamente ou negativamente dependentes
        if prob_integral_adm > prob_se_independente:
            print("Há uma dependência positiva: ser da Administração aumenta a chance de frequentar em tempo integral")
        else:
            print("Há uma dependência negativa: ser da Administração diminui a chance de frequentar em tempo integral")
    
    return prob_integral_adm, prob_se_independente

# Executando todas as funções
tabela_prob = questao_a()
questao_b()
questao_c()
questao_d()
questao_e()

#Cap.4:38

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Criando a tabela de probabilidades conjuntas
data = {
    'Status': ['Satisfatório', 'Inadimplente', 'Total'],
    'Diploma_Sim': [0.26, 0.16, 0.42],
    'Diploma_Nao': [0.24, 0.34, 0.58], 
    'Total': [0.50, 0.50, 1.00]
}

df = pd.DataFrame(data)

# Exibir a tabela de probabilidades
print("Tabela de Probabilidades Conjuntas:")
print(df)
print("\n")

# a. Probabilidade de um aluno ter recebido um diploma universitário
def questao_a():
    prob_diploma = df.loc[2, 'Diploma_Sim']  # Total da coluna Diploma_Sim
    
    print(f"a. Probabilidade de um aluno ter recebido um diploma universitário: {prob_diploma:.4f} ou {prob_diploma*100:.2f}%")
    return prob_diploma

# b. Probabilidade de um aluno não ter recebido um diploma universitário
def questao_b():
    prob_sem_diploma = df.loc[2, 'Diploma_Nao']  # Total da coluna Diploma_Nao
    
    print(f"b. Probabilidade de um aluno não ter recebido um diploma universitário: {prob_sem_diploma:.4f} ou {prob_sem_diploma*100:.2f}%")
    return prob_sem_diploma

# c. Probabilidade de ter empréstimo inadimplente dado que recebeu diploma
def questao_c():
    prob_inadimplente_dado_diploma = df.loc[1, 'Diploma_Sim'] / df.loc[2, 'Diploma_Sim']
    
    print(f"c. Probabilidade de ter empréstimo inadimplente dado que recebeu diploma: {prob_inadimplente_dado_diploma:.4f} ou {prob_inadimplente_dado_diploma*100:.2f}%")
    return prob_inadimplente_dado_diploma

# d. Probabilidade de ter empréstimo inadimplente dado que não recebeu diploma
def questao_d():
    prob_inadimplente_dado_sem_diploma = df.loc[1, 'Diploma_Nao'] / df.loc[2, 'Diploma_Nao']
    
    print(f"d. Probabilidade de ter empréstimo inadimplente dado que não recebeu diploma: {prob_inadimplente_dado_sem_diploma:.4f} ou {prob_inadimplente_dado_sem_diploma*100:.2f}%")
    return prob_inadimplente_dado_sem_diploma

# e. Impacto de abandonar a faculdade sem diploma para alunos com empréstimo
def questao_e():
    # Probabilidade condicional de inadimplência dado diploma
    prob_inadimplente_dado_diploma = df.loc[1, 'Diploma_Sim'] / df.loc[2, 'Diploma_Sim']
    
    # Probabilidade condicional de inadimplência dado sem diploma
    prob_inadimplente_dado_sem_diploma = df.loc[1, 'Diploma_Nao'] / df.loc[2, 'Diploma_Nao']
    
    # Diferença entre as probabilidades
    diferenca = prob_inadimplente_dado_sem_diploma - prob_inadimplente_dado_diploma
    
    # Aumento percentual na chance de inadimplência
    aumento_percentual = (diferenca / prob_inadimplente_dado_diploma) * 100
    
    print("\ne. Impacto de abandonar a faculdade sem diploma para alunos com empréstimo:")
    print(f"   Probabilidade de inadimplência com diploma: {prob_inadimplente_dado_diploma:.4f} ou {prob_inadimplente_dado_diploma*100:.2f}%")
    print(f"   Probabilidade de inadimplência sem diploma: {prob_inadimplente_dado_sem_diploma:.4f} ou {prob_inadimplente_dado_sem_diploma*100:.2f}%")
    print(f"   Diferença absoluta: {diferenca:.4f} ou {diferenca*100:.2f} pontos percentuais")
    print(f"   Aumento relativo: {aumento_percentual:.2f}%")
    
    print("\n   Conclusão: Não obter um diploma universitário aumenta a probabilidade")
    print(f"   de inadimplência em {diferenca*100:.2f} pontos percentuais, o que representa")
    print(f"   um aumento de {aumento_percentual:.2f}% na chance de inadimplência.")
    
    # Visualização das probabilidades condicionais
    labels = ['Com Diploma', 'Sem Diploma']
    satisfatorio = [df.loc[0, 'Diploma_Sim'] / df.loc[2, 'Diploma_Sim'], 
                    df.loc[0, 'Diploma_Nao'] / df.loc[2, 'Diploma_Nao']]
    inadimplente = [prob_inadimplente_dado_diploma, prob_inadimplente_dado_sem_diploma]
    
    # Criando figura para visualização
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Comentando a plotagem para não interromper a execução
    # rects1 = plt.bar(x - width/2, satisfatorio, width, label='Pagamento Satisfatório')
    # rects2 = plt.bar(x + width/2, inadimplente, width, label='Inadimplente')
    
    # plt.ylabel('Probabilidade')
    # plt.title('Status do Empréstimo por Situação do Diploma')
    # plt.xticks(x, labels)
    # plt.ylim(0, 1)
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()
    
    return diferenca, aumento_percentual

# Executando todas as funções
questao_a()
questao_b()
questao_c()
questao_d()
questao_e()

#Cap. 4:42

# Importando bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt

# Dados do problema
p_priori_inadimplencia = 0.05  # Probabilidade a priori de inadimplência (5%)
p_deixar_pagamento_dado_nao_inadimplente = 0.20  # P(deixar pagamento | não inadimplente)
p_deixar_pagamento_dado_inadimplente = 1.0  # P(deixar pagamento | inadimplente)

# a. Calcular a probabilidade a posteriori de inadimplência dado que o cliente deixou de efetuar pagamentos
def calcular_posteriori_inadimplencia():
    # Aplicando o Teorema de Bayes:
    # P(inadimplente | deixou pagamento) = P(deixou pagamento | inadimplente) * P(inadimplente) / P(deixou pagamento)
    
    # Primeiro calculamos P(deixou pagamento)
    # P(deixou pagamento) = P(deixou pagamento | inadimplente) * P(inadimplente) + 
    #                       P(deixou pagamento | não inadimplente) * P(não inadimplente)
    p_nao_inadimplente = 1 - p_priori_inadimplencia
    p_deixar_pagamento = (p_deixar_pagamento_dado_inadimplente * p_priori_inadimplencia + 
                         p_deixar_pagamento_dado_nao_inadimplente * p_nao_inadimplente)
    
    # Agora aplicamos o Teorema de Bayes
    p_inadimplente_dado_deixar_pagamento = (p_deixar_pagamento_dado_inadimplente * 
                                           p_priori_inadimplencia / p_deixar_pagamento)
    
    return p_inadimplente_dado_deixar_pagamento

# b. Analisar se o banco deveria cancelar o cartão se o cliente deixar de fazer um pagamento
def analisar_decisao_cancelamento():
    p_posteriori = calcular_posteriori_inadimplencia()
    
    # Verificar se a probabilidade a posteriori é maior que 0.20
    deveria_cancelar = p_posteriori > 0.20
    
    return p_posteriori, deveria_cancelar

# Executando as funções
p_posteriori = calcular_posteriori_inadimplencia()
decisao_resultado = analisar_decisao_cancelamento()

# Exibindo os resultados
print(f"Probabilidade a priori de inadimplência: {p_priori_inadimplencia:.2f} ou {p_priori_inadimplencia*100:.1f}%")
print(f"Probabilidade de clientes não inadimplentes deixarem de efetuar pagamento: {p_deixar_pagamento_dado_nao_inadimplente:.2f} ou {p_deixar_pagamento_dado_nao_inadimplente*100:.1f}%")
print(f"Probabilidade de clientes inadimplentes deixarem de efetuar pagamento: {p_deixar_pagamento_dado_inadimplente:.2f} ou {p_deixar_pagamento_dado_inadimplente*100:.1f}%")

print("\na. Probabilidade a posteriori de inadimplência dado que deixou de efetuar pagamento:")
print(f"   P(inadimplente | deixou pagamento) = {p_posteriori:.4f} ou {p_posteriori*100:.2f}%")

print("\nb. Análise sobre cancelamento do cartão:")
print(f"   Probabilidade a posteriori de inadimplência: {p_posteriori:.4f} ou {p_posteriori*100:.2f}%")
print(f"   Limite estabelecido pelo banco para cancelamento: 0.20 ou 20%")

if decisao_resultado[1]:
    print("   Decisão: SIM, o banco deveria cancelar o cartão.")
    print("   Justificativa: A probabilidade a posteriori de inadimplência ({:.2f}%) é maior que o limite de 20%.".format(p_posteriori*100))
else:
    print("   Decisão: NÃO, o banco não deveria cancelar o cartão.")
    print("   Justificativa: A probabilidade a posteriori de inadimplência ({:.2f}%) é menor que o limite de 20%.".format(p_posteriori*100))

# Visualização da alteração na probabilidade de inadimplência (antes e depois)
# Comentado para não interromper a execução
'''
labels = ['Probabilidade a priori', 'Probabilidade a posteriori']
valores = [p_priori_inadimplencia, p_posteriori]

plt.figure(figsize=(10, 6))
plt.bar(labels, valores, color=['blue', 'red'])
plt.title('Comparação entre probabilidade a priori e a posteriori de inadimplência')
plt.ylabel('Probabilidade')
plt.axhline(y=0.20, color='green', linestyle='--', label='Limite para cancelamento (20%)')
plt.ylim(0, max(valores) * 1.2)
plt.legend()
plt.show()
'''

#Cap.5:3

import random

def simular_entrevistas():
    """
    Simula o experimento de três estudantes em entrevistas, onde cada um pode 
    receber ou não uma oferta de emprego.
    
    Retorna: O número total de ofertas recebidas pelos três estudantes.
    """
    # Simula o resultado de cada entrevista (True = oferta recebida, False = sem oferta)
    resultados = [random.choice([True, False]) for _ in range(3)]
    
    # Conta o número total de ofertas
    num_ofertas = sum(resultados)
    
    return num_ofertas

# a) Enumerar os resultados experimentais
print("a) Resultados experimentais possíveis:")
print("0 ofertas: nenhum estudante recebe oferta")
print("1 oferta: apenas um estudante recebe oferta")
print("2 ofertas: dois estudantes recebem ofertas")
print("3 ofertas: todos os três estudantes recebem ofertas")

# b) Definir uma variável aleatória
print("\nb) Definição da variável aleatória:")
print("X = número de ofertas recebidas pelos três estudantes")
print("X pode assumir os valores: 0, 1, 2 ou 3")
print("A variável aleatória é discreta, pois assume um número finito de valores")

# c) Mostrar o valor da variável aleatória para cada resultado experimental
print("\nc) Valores da variável aleatória para cada resultado experimental:")
print("Se nenhum estudante recebe oferta: X = 0")
print("Se apenas um estudante recebe oferta: X = 1")
print("Se dois estudantes recebem ofertas: X = 2")
print("Se todos os três estudantes recebem ofertas: X = 3")

# Demonstração da simulação
print("\nDemonstração da simulação:")
for i in range(5):
    ofertas = simular_entrevistas()
    print(f"Simulação {i+1}: {ofertas} oferta(s) recebida(s)")

#Cap. 5:8

import numpy as np
import matplotlib.pyplot as plt

# Dados do problema
num_dias = 20
salas_por_dia = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]

# a) Distribuição discreta de probabilidade empírica
def calcular_distribuicao(dados):
    # Contagem de ocorrências
    valores_unicos = sorted(set(dados))
    contagem = {valor: dados.count(valor) for valor in valores_unicos}
    
    # Cálculo da frequência relativa (probabilidade empírica)
    total = len(dados)
    probabilidade = {valor: contagem[valor] / total for valor in valores_unicos}
    
    return probabilidade

# Calcular a distribuição de probabilidade
distribuicao = calcular_distribuicao(salas_por_dia)

print("a) Distribuição de probabilidade empírica:")
for salas, prob in distribuicao.items():
    print(f"P(X = {salas}) = {prob:.2f} ou {prob*100:.0f}%")

# b) Desenhar o gráfico da distribuição
def plotar_distribuicao(distribuicao):
    valores = list(distribuicao.keys())
    probabilidades = list(distribuicao.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(valores, probabilidades, color='skyblue', edgecolor='black')
    plt.xlabel('Número de salas de cirurgia em uso')
    plt.ylabel('Probabilidade')
    plt.title('Distribuição de Probabilidade - Salas de Cirurgia em Uso')
    plt.xticks(valores)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar os valores de probabilidade acima das barras
    for i, v in enumerate(probabilidades):
        plt.text(valores[i], v + 0.01, f'{v:.2f}', ha='center')
    
    # Para visualização em notebook ou janela separada
    plt.tight_layout()
    plt.show()

# Imprimir as informações para quem não pode visualizar o gráfico
print("\nb) Gráfico da distribuição (representação textual):")
for salas, prob in distribuicao.items():
    barra = '#' * int(prob * 50)
    print(f"{salas} sala(s): {barra} {prob:.2f}")

plotar_distribuicao(distribuicao)

# c) Verificar se satisfaz as condições de uma distribuição discreta válida
def verificar_distribuicao_valida(distribuicao):
    # Condição 1: Todos os valores de probabilidade devem ser não-negativos
    condicao1 = all(p >= 0 for p in distribuicao.values())
    
    # Condição 2: A soma de todas as probabilidades deve ser igual a 1
    condicao2 = abs(sum(distribuicao.values()) - 1.0) < 1e-10
    
    return condicao1, condicao2

condicao1, condicao2 = verificar_distribuicao_valida(distribuicao)

print("\nc) Verificação das condições para uma distribuição discreta válida:")
print(f"1. Todas as probabilidades são não-negativas: {condicao1}")
print(f"2. A soma de todas as probabilidades é 1: {condicao2}")
print(f"3. Soma das probabilidades = {sum(distribuicao.values()):.10f}")

if condicao1 and condicao2:
    print("A distribuição satisfaz as condições necessárias para ser uma distribuição discreta de probabilidade válida.")
else:
    print("A distribuição NÃO satisfaz todas as condições necessárias.")

# Estatísticas adicionais
valores = np.array(list(distribuicao.keys()))
probabilidades = np.array(list(distribuicao.values()))
media = sum(valores * probabilidades)
variancia = sum((valores - media)**2 * probabilidades)

print("\nEstatísticas adicionais:")
print(f"Valor esperado (média): {media:.2f} salas")
print(f"Variância: {variancia:.2f}")
print(f"Desvio padrão: {np.sqrt(variancia):.2f}")


#Cap. 5: 18

import numpy as np
import matplotlib.pyplot as plt

# Dados da tabela (em milhares de casas)
dados = {
    'vezes': [0, 1, 2, 3, 4],  # 4 representa "4 vezes ou mais"
    'proprietarios': [439, 1100, 249, 98, 120],
    'locatarios': [394, 760, 221, 92, 111]
}

# Convertendo para arrays numpy
vezes = np.array(dados['vezes'])
proprietarios = np.array(dados['proprietarios'])
locatarios = np.array(dados['locatarios'])

# Calculando o total de casas para cada categoria
total_proprietarios = sum(proprietarios)
total_locatarios = sum(locatarios)

# a) Variável aleatória x e distribuição de probabilidade para proprietários
def calcular_distribuicao(valores, total):
    """Calcula a distribuição de probabilidade."""
    return valores / total

# Distribuição de probabilidade para proprietários
prob_proprietarios = calcular_distribuicao(proprietarios, total_proprietarios)

print("a) Variável aleatória x = número de vezes que casas ocupadas por proprietários")
print("   tiveram interrupção no fornecimento de água com duração de 6 horas ou mais")
print("   nos últimos 3 meses.")
print("\nDistribuição de probabilidade para x:")
for i, prob in enumerate(prob_proprietarios):
    valor_x = "4 ou mais" if i == 4 else str(i)
    print(f"P(x = {valor_x}) = {prob:.4f} ou {prob*100:.2f}%")

# b) Valor esperado e variância para x (proprietários)
def calcular_valor_esperado(valores, probabilidades):
    """Calcula o valor esperado (média) de uma distribuição de probabilidade."""
    return sum(valores * probabilidades)

def calcular_variancia(valores, probabilidades, valor_esperado):
    """Calcula a variância de uma distribuição de probabilidade."""
    return sum((valores - valor_esperado)**2 * probabilidades)

# Para o cálculo correto do valor esperado e variância quando temos "4 ou mais",
# vamos considerar como exatamente 4 para simplificar
valor_esperado_x = calcular_valor_esperado(vezes, prob_proprietarios)
variancia_x = calcular_variancia(vezes, prob_proprietarios, valor_esperado_x)

print("\nb) Para casas ocupadas por proprietários:")
print(f"   Valor esperado de x = {valor_esperado_x:.4f}")
print(f"   Variância de x = {variancia_x:.4f}")

# c) Variável aleatória y e distribuição de probabilidade para locatários
prob_locatarios = calcular_distribuicao(locatarios, total_locatarios)

print("\nc) Variável aleatória y = número de vezes que casas ocupadas por locatários")
print("   tiveram interrupção no fornecimento de água com duração de 6 horas ou mais")
print("   nos últimos 3 meses.")
print("\nDistribuição de probabilidade para y:")
for i, prob in enumerate(prob_locatarios):
    valor_y = "4 ou mais" if i == 4 else str(i)
    print(f"P(y = {valor_y}) = {prob:.4f} ou {prob*100:.2f}%")

# d) Valor esperado e variância para y (locatários)
valor_esperado_y = calcular_valor_esperado(vezes, prob_locatarios)
variancia_y = calcular_variancia(vezes, prob_locatarios, valor_esperado_y)

print("\nd) Para casas ocupadas por locatários:")
print(f"   Valor esperado de y = {valor_esperado_y:.4f}")
print(f"   Variância de y = {variancia_y:.4f}")

# Visualização das distribuições (opcional)
def plotar_comparacao():
    """Plota um gráfico comparando as duas distribuições."""
    plt.figure(figsize=(10, 6))
    
    bar_width = 0.35
    posicoes = np.arange(len(vezes))
    
    plt.bar(posicoes - bar_width/2, prob_proprietarios, bar_width, 
            label='Proprietários', color='skyblue', edgecolor='black')
    plt.bar(posicoes + bar_width/2, prob_locatarios, bar_width,
            label='Locatários', color='lightgreen', edgecolor='black')
    
    # Adicionando rótulos e título
    plt.xlabel('Número de interrupções')
    plt.ylabel('Probabilidade')
    plt.title('Distribuição de Probabilidade de Interrupções no Fornecimento de Água')
    plt.xticks(posicoes, ['0', '1', '2', '3', '4 ou mais'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Análise comparativa
print("\nAnálise Comparativa:")
print(f"Média de interrupções em casas de proprietários: {valor_esperado_x:.4f}")
print(f"Média de interrupções em casas de locatários: {valor_esperado_y:.4f}")

if valor_esperado_x > valor_esperado_y:
    print("Em média, casas ocupadas por proprietários sofrem mais interrupções.")
elif valor_esperado_x < valor_esperado_y:
    print("Em média, casas ocupadas por locatários sofrem mais interrupções.")
else:
    print("Em média, ambos os tipos de ocupação sofrem o mesmo número de interrupções.")

print(f"\nVariabilidade para proprietários: {variancia_x:.4f}")
print(f"Variabilidade para locatários: {variancia_y:.4f}")

# Apenas para representação textual do gráfico (quando não é possível visualizar)
print("\nRepresentação textual das distribuições:")
print("\nProprietários:")
for i, prob in enumerate(prob_proprietarios):
    barra = "#" * int(prob * 50)
    valor = "4+" if i == 4 else str(i)
    print(f"{valor}: {barra} {prob:.4f}")

print("\nLocatários:")
for i, prob in enumerate(prob_locatarios):
    barra = "#" * int(prob * 50)
    valor = "4+" if i == 4 else str(i)
    print(f"{valor}: {barra} {prob:.4f}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dados da tabela
vezes_interrupcao = [0, 1, 2, 3, 4]  # 4 representa "4 vezes ou mais"
casas_proprietarios = [439, 1100, 249, 98, 120]  # em milhares
casas_locatarios = [394, 760, 221, 92, 111]  # em milhares

# a. Variável aleatória x = número de vezes para casas ocupadas por proprietários
# Calcular as probabilidades para x
total_proprietarios = sum(casas_proprietarios)
prob_x = [valor / total_proprietarios for valor in casas_proprietarios]

# Calcular valor esperado E[x] e variância para x
E_x = sum(x * p for x, p in zip(vezes_interrupcao, prob_x))
E_x2 = sum(x**2 * p for x, p in zip(vezes_interrupcao, prob_x))
Var_x = E_x2 - E_x**2

# c. Variável aleatória y = número de vezes para casas ocupadas por locatários
# Calcular as probabilidades para y
total_locatarios = sum(casas_locatarios)
prob_y = [valor / total_locatarios for valor in casas_locatarios]

# Calcular valor esperado E[y] e variância para y
E_y = sum(y * p for y, p in zip(vezes_interrupcao, prob_y))
E_y2 = sum(y**2 * p for y, p in zip(vezes_interrupcao, prob_y))
Var_y = E_y2 - E_y**2

# Criando um DataFrame para mostrar a distribuição de probabilidade de x
df_x = pd.DataFrame({
    'Número de vezes': vezes_interrupcao,
    'Casas de proprietários (milhares)': casas_proprietarios,
    'Probabilidade': prob_x
})

# Criando um DataFrame para mostrar a distribuição de probabilidade de y
df_y = pd.DataFrame({
    'Número de vezes': vezes_interrupcao,
    'Casas de locatários (milhares)': casas_locatarios,
    'Probabilidade': prob_y
})

# Exibir resultados
print("a. Distribuição de probabilidade para x (proprietários):")
print(df_x)
print(f"\nValor esperado E[x] = {E_x:.4f}")
print(f"Variância de x = {Var_x:.4f}")

print("\nc. Distribuição de probabilidade para y (locatários):")
print(df_y)
print(f"\nValor esperado E[y] = {E_y:.4f}")
print(f"Variância de y = {Var_y:.4f}")

# Comparação visual das distribuições de probabilidade
plt.figure(figsize=(10, 6))
plt.bar([x - 0.2 for x in vezes_interrupcao], prob_x, width=0.4, label='Proprietários (x)')
plt.bar([x + 0.2 for x in vezes_interrupcao], prob_y, width=0.4, label='Locatários (y)')
plt.xlabel('Número de interrupções')
plt.ylabel('Probabilidade')
plt.title('Distribuição de probabilidade das interrupções de água')
plt.xticks(vezes_interrupcao)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Observações para o item e
print("\ne. Observações da comparação entre proprietários e locatários:")
if E_x > E_y:
    print(f"- Proprietários têm, em média, mais interrupções ({E_x:.4f} vs {E_y:.4f}).")
elif E_x < E_y:
    print(f"- Locatários têm, em média, mais interrupções ({E_y:.4f} vs {E_x:.4f}).")
else:
    print("- Ambos os grupos têm, em média, o mesmo número de interrupções.")

if Var_x > Var_y:
    print(f"- A variabilidade é maior entre proprietários (variância {Var_x:.4f} vs {Var_y:.4f}).")
elif Var_x < Var_y:
    print(f"- A variabilidade é maior entre locatários (variância {Var_y:.4f} vs {Var_x:.4f}).")
else:
    print("- Ambos os grupos têm a mesma variabilidade.")

# Calcular percentuais de casas sem interrupções
pct_sem_interrupcao_prop = prob_x[0] * 100
pct_sem_interrupcao_loc = prob_y[0] * 100
print(f"- {pct_sem_interrupcao_prop:.1f}% dos proprietários não tiveram interrupções, contra {pct_sem_interrupcao_loc:.1f}% dos locatários.")

# Calculando percentuais para 3 ou mais interrupções
pct_3mais_prop = sum(prob_x[3:]) * 100
pct_3mais_loc = sum(prob_y[3:]) * 100
print(f"- {pct_3mais_prop:.1f}% dos proprietários tiveram 3 ou mais interrupções, contra {pct_3mais_loc:.1f}% dos locatários.")

#Cap5:27

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Dados da tabela
data = np.array([
    [42, 39, 3],  # Qualidade 1, Preços 1, 2, 3
    [33, 63, 54],  # Qualidade 2, Preços 1, 2, 3
    [3, 15, 48]   # Qualidade 3, Preços 1, 2, 3
])

# Total de restaurantes
n_total = np.sum(data)

# a. Desenvolver a distribuição de probabilidade bivariada
# Calculando a matriz de probabilidades P(X=x, Y=y)
prob_bivariada = data / n_total

# Criando um DataFrame para melhor visualização
qualidade = [1, 2, 3]
preco = [1, 2, 3]
df_prob = pd.DataFrame(prob_bivariada, index=qualidade, columns=preco)
df_prob.index.name = 'Qualidade (x)'
df_prob.columns.name = 'Preço (y)'

# b. Calcular o valor esperado e a variância para a classificação de qualidade, x
# Calculando as probabilidades marginais P(X=x)
prob_x = np.sum(prob_bivariada, axis=1)
df_prob_x = pd.DataFrame({'Probabilidade': prob_x}, index=qualidade)
df_prob_x.index.name = 'Qualidade (x)'

# Valor esperado de X
E_X = np.sum(qualidade * prob_x)

# Variância de X
Var_X = np.sum((np.array(qualidade) ** 2) * prob_x) - E_X ** 2

# c. Calcular o valor esperado e a variância para o preço da refeição, y
# Calculando as probabilidades marginais P(Y=y)
prob_y = np.sum(prob_bivariada, axis=0)
df_prob_y = pd.DataFrame({'Probabilidade': prob_y}, index=preco)
df_prob_y.index.name = 'Preço (y)'

# Valor esperado de Y
E_Y = np.sum(preco * prob_y)

# Variância de Y
Var_Y = np.sum((np.array(preco) ** 2) * prob_y) - E_Y ** 2

# d. Calcular a covariância entre x e y
# E[XY] - cálculo do valor esperado do produto
E_XY = 0
for i in range(len(qualidade)):
    for j in range(len(preco)):
        E_XY += qualidade[i] * preco[j] * prob_bivariada[i, j]

# Covariância
Cov_XY = E_XY - E_X * E_Y

# e. Calcular o coeficiente de correlação
Corr_XY = Cov_XY / np.sqrt(Var_X * Var_Y)

# Criando heatmap para visualizar a distribuição bivariada
plt.figure(figsize=(10, 8))
sns.heatmap(df_prob, annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={'label': 'Probabilidade'})
plt.title('Distribuição de Probabilidade Bivariada - Qualidade vs Preço')

# Criando gráfico de barras para as distribuições marginais
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.bar(qualidade, prob_x)
ax1.set_xlabel('Qualidade (x)')
ax1.set_ylabel('Probabilidade')
ax1.set_title('Distribuição Marginal de Qualidade')
ax1.set_xticks(qualidade)

ax2.bar(preco, prob_y)
ax2.set_xlabel('Preço (y)')
ax2.set_ylabel('Probabilidade')
ax2.set_title('Distribuição Marginal de Preço')
ax2.set_xticks(preco)

# Exibindo resultados
print("a. Distribuição de Probabilidade Bivariada P(X=x, Y=y):")
print(df_prob)
print("\n")

print("b. Distribuição Marginal de Qualidade P(X=x):")
print(df_prob_x)
print(f"Valor Esperado E[X] = {E_X:.4f}")
print(f"Variância Var[X] = {Var_X:.4f}")
print("\n")

print("c. Distribuição Marginal de Preço P(Y=y):")
print(df_prob_y)
print(f"Valor Esperado E[Y] = {E_Y:.4f}")
print(f"Variância Var[Y] = {Var_Y:.4f}")
print("\n")

print("d. Covariância:")
print(f"E[XY] = {E_XY:.4f}")
print(f"Covariância entre X e Y = {Cov_XY:.4f}")
enunciado_d = 1.6691
print(f"O enunciado afirma que Var(X + Y) = 1.6691")
var_x_plus_y = Var_X + Var_Y + 2*Cov_XY
print(f"Calculamos Var(X + Y) = Var(X) + Var(Y) + 2*Cov(X,Y) = {var_x_plus_y:.4f}")
print("\n")

print("e. Coeficiente de Correlação:")
print(f"Correlação entre X e Y = {Corr_XY:.4f}")

# Análise da relação entre qualidade e preço
print("\nAnálise da relação entre qualidade e preço:")
if Corr_XY > 0.7:
    forca = "forte"
elif Corr_XY > 0.3:
    forca = "moderada"
else:
    forca = "fraca"

print(f"A correlação entre qualidade e preço é {forca} e positiva ({Corr_XY:.4f}).")
print(f"Isso indica que restaurantes com melhor qualidade tendem a ter preços mais altos.")

# Probabilidade condicional de alta qualidade dado baixo preço
p_alta_dado_baixo = prob_bivariada[2, 0] / prob_y[0]
print(f"\nProbabilidade de encontrar qualidade alta (3) dado preço baixo (1) = {p_alta_dado_baixo:.4f}")
print(f"Isso equivale a {p_alta_dado_baixo*100:.2f}% dos restaurantes de preço baixo.")

if p_alta_dado_baixo < 0.1:
    resposta = "Improvável"
else:
    resposta = "Possível, mas não comum"

print(f"Conclusão: É {resposta} encontrar um restaurante de baixo custo com alta qualidade nesta cidade.")

#Cap.5:34

import numpy as np
from scipy import stats

# Dados do problema
p_adolescentes_pandora = 0.39  # 39% dos adolescentes usam o Pandora Media
n = 10  # número de adolescentes na amostra aleatória

# a. É um experimento binomial?
# Sim, pois temos:
# 1. Número fixo de ensaios (n=10)
# 2. Cada ensaio tem apenas dois resultados possíveis (usa ou não usa o Pandora)
# 3. A probabilidade de sucesso é constante (p=0.39)
# 4. Os ensaios são independentes (assume-se que os adolescentes são selecionados aleatoriamente)

# b. Probabilidade de que nenhum dos 10 adolescentes use o serviço
prob_nenhum = stats.binom.pmf(k=0, n=n, p=p_adolescentes_pandora)
print(f"b. Probabilidade de nenhum usar o Pandora: {prob_nenhum:.6f}")

# c. Probabilidade de que 4 dos 10 adolescentes usem o serviço
prob_quatro = stats.binom.pmf(k=4, n=n, p=p_adolescentes_pandora)
print(f"c. Probabilidade de exatamente 4 usarem o Pandora: {prob_quatro:.6f}")

# d. Probabilidade de que pelo menos 2 dos 10 adolescentes usem o serviço
# Isto é igual a 1 menos a probabilidade de que 0 ou 1 usem
prob_menos_que_dois = stats.binom.pmf(k=0, n=n, p=p_adolescentes_pandora) + stats.binom.pmf(k=1, n=n, p=p_adolescentes_pandora)
prob_pelo_menos_dois = 1 - prob_menos_que_dois
print(f"d. Probabilidade de pelo menos 2 usarem o Pandora: {prob_pelo_menos_dois:.6f}")

# Alternativamente para o item d, podemos usar a função de distribuição cumulativa
# prob_pelo_menos_dois = 1 - stats.binom.cdf(k=1, n=n, p=p_adolescentes_pandora)

#Cap.6:2


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Dados do problema: X ~ U(10, 20)
a = 10  # limite inferior
b = 20  # limite superior

# Função densidade de probabilidade para X ~ U(a, b)
def pdf(x):
    if a <= x <= b:
        return 1/(b-a)
    else:
        return 0

# a. Gráfico da função densidade de probabilidade
x_values = np.linspace(8, 22, 1000)
y_values = [pdf(x) for x in x_values]

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values)
plt.fill_between(x_values, y_values, where=[(x >= a and x <= b) for x in x_values], alpha=0.3)
plt.grid(True)
plt.title('Função Densidade de Probabilidade de X ~ U(10, 20)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=a, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=b, color='r', linestyle='--', alpha=0.5)
plt.text(a-0.5, 0.05, f'a = {a}')
plt.text(b-0.5, 0.05, f'b = {b}')
plt.text((a+b)/2, 1/(b-a)+0.01, f'f(x) = {1/(b-a)}')
plt.show()


# b. Calcule P(x < 15)
prob_less_than_15 = (15 - a)/(b - a)
print(f"b. P(X < 15) = {prob_less_than_15}")

# c. Calcule P(12 ≤ x ≤ 18)
prob_between_12_and_18 = (18 - 12)/(b - a)
print(f"c. P(12 ≤ X ≤ 18) = {prob_between_12_and_18}")

# d. Calcule E(X)
# Para uma distribuição uniforme, E(X) = (a + b)/2
expected_value = (a + b)/2
print(f"d. E(X) = {expected_value}")

# e. Calcule Var(X)
# Para uma distribuição uniforme, Var(X) = (b - a)²/12
variance = (b - a)**2/12
print(f"e. Var(X) = {variance}")

#Cap.6:10

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as patches

# Definir o intervalo de valores para z
z = np.linspace(-4, 4, 1000)

# Calcular a função de densidade de probabilidade da distribuição normal padrão
pdf = stats.norm.pdf(z)

# Criar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(z, pdf, 'b-', lw=2, label='Distribuição Normal Padrão')

# Marcar os pontos específicos no eixo x
pontos_x = [-3, -2, -1, 0, 1, 2, 3]
pontos_y = [0] * len(pontos_x)
plt.scatter(pontos_x, pontos_y, color='red', s=50)

# Adicionar rótulos aos pontos
for i, (x, y) in enumerate(zip(pontos_x, pontos_y)):
    plt.annotate(str(x), (x, y-0.01), ha='center', fontsize=12)

# Adicionar rótulos e título
plt.title('Distribuição Normal Padrão', fontsize=14)
plt.xlabel('z', fontsize=12)
plt.ylabel('Densidade de Probabilidade', fontsize=12)

# Configurar os limites do eixo y
plt.ylim(0, 0.45)

# Calcular as probabilidades solicitadas usando scipy
prob_a = stats.norm.cdf(1.5)
prob_b = stats.norm.cdf(1)
prob_c = stats.norm.cdf(1.5) - stats.norm.cdf(1)
prob_d = stats.norm.cdf(2.5) - stats.norm.cdf(0)

# Adicionar as áreas sombreadas para cada probabilidade
# a. P(z ≤ 1,5)
x_a = np.linspace(-4, 1.5, 500)
y_a = stats.norm.pdf(x_a)
plt.fill_between(x_a, y_a, alpha=0.3, color='red', label=f'a. P(z ≤ 1,5) = {prob_a:.4f}')

# b. P(z ≤ 1)
x_b = np.linspace(-4, 1, 500)
y_b = stats.norm.pdf(x_b)
plt.fill_between(x_b, y_b, alpha=0.3, color='green', label=f'b. P(z ≤ 1) = {prob_b:.4f}')

# c. P(1 ≤ z ≤ 1,5)
x_c = np.linspace(1, 1.5, 500)
y_c = stats.norm.pdf(x_c)
plt.fill_between(x_c, y_c, alpha=0.5, color='purple', label=f'c. P(1 ≤ z ≤ 1,5) = {prob_c:.4f}')

# d. P(0 ≤ z ≤ 2,5)
x_d = np.linspace(0, 2.5, 500)
y_d = stats.norm.pdf(x_d)
plt.fill_between(x_d, y_d, alpha=0.3, color='orange', label=f'd. P(0 ≤ z ≤ 2,5) = {prob_d:.4f}')

# Adicionar legenda
plt.legend(loc='upper right', fontsize=10)

# Resultados
print("Resultados das probabilidades:")
print(f"a. P(z ≤ 1,5) = {prob_a:.4f}")
print(f"b. P(z ≤ 1) = {prob_b:.4f}")
print(f"c. P(1 ≤ z ≤ 1,5) = {prob_c:.4f}")
print(f"d. P(0 ≤ z ≤ 2,5) = {prob_d:.4f}")

# Mostrar o gráfico
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Cap. 6:16

import numpy as np
from scipy import stats

# Função para encontrar z tal que P(Z > z) = área
def encontrar_z_area_direita(area):
    # A função ppf calcula o quantil, mas para área à esquerda
    # Para área à direita, usamos 1-área
    return stats.norm.ppf(1 - area)

# Situações para calcular
areas = [0.01, 0.025, 0.05, 0.10]
letras = ['a', 'b', 'c', 'd']

# Calcular e exibir os resultados
print("Resultados:")
for letra, area in zip(letras, areas):
    z = encontrar_z_area_direita(area)
    print(f"{letra}. Área à direita de z é {area}: z = {z:.4f}")

#Cap. 6:22

import numpy as np
from scipy import stats

# Parâmetros da distribuição normal
media = 8.35  # tempo médio em horas
desvio_padrao = 2.5  # desvio padrão em horas

# Criar a distribuição normal
distribuicao = stats.norm(loc=media, scale=desvio_padrao)

# a. Qual é a probabilidade de que um espectador assista à TV durante 5 a 10 horas por dia?
prob_entre_5_e_10 = distribuicao.cdf(10) - distribuicao.cdf(5)
print(f"a. Probabilidade de assistir entre 5 e 10 horas: {prob_entre_5_e_10:.4f} ou {prob_entre_5_e_10*100:.2f}%")

# b. Por quantas horas um espectador deve assistir à TV para estar entre os 3% que mais assistem TV?
# Os 3% que mais assistem correspondem ao percentil 97
horas_top_3_percent = distribuicao.ppf(0.97)
print(f"b. Horas para estar entre os 3% que mais assistem: {horas_top_3_percent:.2f} horas")

# c. Qual é a probabilidade de que um telespectador assista à TV por mais de 3 horas por dia?
prob_mais_de_3 = 1 - distribuicao.cdf(3)
print(f"c. Probabilidade de assistir mais de 3 horas: {prob_mais_de_3:.4f} ou {prob_mais_de_3*100:.2f}%")

#Cap.6:24

import numpy as np
from scipy import stats

# Parâmetros da distribuição normal
media = 749  # gasto médio em US$
desvio_padrao = 225  # desvio padrão em US$

# Criar a distribuição normal
distribuicao = stats.norm(loc=media, scale=desvio_padrao)

# a. Qual é a probabilidade de as despesas familiares do fim de semana serem inferiores a US$ 400?
prob_menos_400 = distribuicao.cdf(400)
print(f"a. Probabilidade de gastos inferiores a US$ 400: {prob_menos_400:.4f} ou {prob_menos_400*100:.2f}%")

# b. Qual é a probabilidade de as despesas familiares do fim de semana serem de US$ 800 ou mais?
prob_800_ou_mais = 1 - distribuicao.cdf(800)
print(f"b. Probabilidade de gastos de US$ 800 ou mais: {prob_800_ou_mais:.4f} ou {prob_800_ou_mais*100:.2f}%")

# c. Qual é a probabilidade de as despesas familiares do final de semana ficarem entre US$ 500 e US$ 1.000?
prob_entre_500_1000 = distribuicao.cdf(1000) - distribuicao.cdf(500)
print(f"c. Probabilidade de gastos entre US$ 500 e US$ 1.000: {prob_entre_500_1000:.4f} ou {prob_entre_500_1000*100:.2f}%")

# d. Quais seriam as despesas do fim de semana do Dia do Trabalho para os 5% das famílias com planos de viagem mais caros?
# Os 5% mais caros correspondem ao percentil 95
gastos_top_5_percent = distribuicao.ppf(0.95)
print(f"d. Gastos para os 5% das famílias com planos mais caros: US$ {gastos_top_5_percent:.2f} ou mais")

#Cap.6:40

import numpy as np
from scipy import stats

# Parâmetros da distribuição normal
media = 19000  # valor médio da bolsa em US$
desvio_padrao = 2100  # desvio padrão em US$

# Criar a distribuição normal
distribuicao = stats.norm(loc=media, scale=desvio_padrao)

# a. Para os 10% de bolsas de estudos com menor valor para atletas quanto elas valem?
# Isso corresponde ao percentil 10
valor_percentil_10 = distribuicao.ppf(0.10)
print(f"a. Valor para os 10% de bolsas com menor valor: US$ {valor_percentil_10:.2f}")

# b. Qual a porcentagem de bolsas de estudos avaliadas em US$ 22.000 ou mais?
prob_22000_ou_mais = 1 - distribuicao.cdf(22000)
print(f"b. Porcentagem de bolsas de US$ 22.000 ou mais: {prob_22000_ou_mais:.4f} ou {prob_22000_ou_mais*100:.2f}%")

# c. Para os 3% de bolsas de estudos que são mais valiosas quanto elas valem?
# Isso corresponde ao percentil 97
valor_percentil_97 = distribuicao.ppf(0.97)
print(f"c. Valor para os 3% de bolsas mais valiosas: US$ {valor_percentil_97:.2f} ou mais")

