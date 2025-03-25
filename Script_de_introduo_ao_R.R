# Estes são os pacotes que vamos utilizar na nossa disciplina de métodos
# Quantitativos I. Basta rodar estes comandos uma vez para instalar
# os pacotes no seu RStudio.

install.packages('tidymodels')
install.packages('sjPlot')
install.packages('sjmisc')
install.packages('sjlabelled')
install.packages('wooldridge')
install.packages('ggplot2')
install.packages('tidyr')
install.packages('dplyr')

# Estes são os comandos para "ligar" os pacotes que foram instalados
# acima. Sempre que for iniciar uma sessão no Rstudio, lembre
# de rodar os comando abaixo para carregar as livrarias dos pacotes
# desta maneira você terá acesso aos comandos disponíveis em
# cada livraria. A livraria do 'wooldridge' permite que 
# usar o comando data('nomedabase') para carregar
# direto da internet as bases de dados utilizadas no livro.

library(sjPlot)
library(sjmisc)
library(sjlabelled)
library(ggplot2)
library(tidymodels)
library(wooldridge)
library(tidyr)
library(dplyr)

# Vamos utilizar a base de dados 'bwght' do wooldridge para 
# testar alguns comandos iniciais:
# primeiro precisamos carregar a base com o comando a seguir!

data('bwght') # carregamos a base bwght do livro wooldridge

?bwght # este comando abre o help da base e nos mostra a descrição das variáveis contidas nela.

# Agora vamos ver quais são as variáveis que a base de dados possui:
ls(bwght)

# podemos ver que a base de dados possui 14 variáveis diferentes,
# que são listadas ao realizar o comando ls('nomedabase')
# com estas informações vamos analisar algumas estatísticas descritivas!
# os comandos utilizam o seguinte formato: comando(nome_da_base$nome_da_variável)
 
sum(bwght$cigs) # aqui estamos pedindo a soma dos valores da variável cigs na base bwght
min(bwght$cigs) # aqui estamos pedindo o valor mínimo da variável cigs na base bwght
max(bwght$cigs) # aqui estamos pedindo o valor máximo da variável cigs na base bwght
mean(bwght$bwght) # aqui estamos pedindo a média dos valores da variável bwght na base bwght
median(bwght$bwght) # aqui estamos pedindo a mediana dos valores da variável bwght na base bwght
quantile(bwght$bwght) # aqui estamos pedindo os quanits da variável bwght na base bwght
var(bwght$bwght) # aqui estamos pedindo a variância da variável bwght na base bwght
sd(bwght$bwght) # aqui estamos pedindo o desvio padrão da variável bwght na base bwght

# Se bwght é o peso da criança ao nascer e cigs é o número de cigarros fumados
# por dia de uma mãe grávida, você acha que há uma correlação entre essas duas variáveis?
# essa correlação seria positiva ou negativa? vamos testar!

cor(bwght$cigs, bwght$bwght) # com este comando estamos pedindo a correlação de x=cigs e y=bwght
cov(bwght$bwght,bwght$cigs)# aqui estamos pedindo a covariância de x=bwght e y=cigs

# Agora vamos explorar um pouco dos comandos para gerar gráficos usando o ggplot2.
# Os mais utilizados são o Histograma, o Boxplot e o diagrama de dispersão (ScaterPlot)

# Para gerar um Histograma usamos o seguinte comando:

hist(bwght$cigs) # esse comando faz um histograma da variável cigs na base bwght

# Agora vamos gerar um gráfico de dispersão chamado de ScaterPlot
# usando o seguinte comando:

plot(bwght$cigs) # aqui estamos gerando um gráfico de dispersão para uma única variável
plot(bwght$cigs, bwght$bwght) # aqui estamos gerando um gráfico de dispersão para x=cigs e y=bwght

# Para gerar um Boxplot usamos o seguinte comando:

boxplot(bwght$bwght) # geramos um boxplot da variável bwght da base bwght

# Todos os comandos de gráficos podem ser escritos com um código mais "refinado"
# é possível mudar a escala dos eixos, as cores dos pontos etc,
# para isto é necessário ler o arquivo data-visualization que está anexado 
# no sigaa.
