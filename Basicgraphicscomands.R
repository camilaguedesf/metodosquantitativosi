# Aula 03 - Graficos basicos e estatisticas descritivas
# Usando a base de dados "cars", contida no R

## Grafico de dispersao 
data(cars)
?cars
cars
View(cars)
head(cars)
tail(cars)
hist(cars$speed)
hist(cars$dist)

?plot

plot(x = cars$speed, y = cars$dist)
plot(x = cars$dist, y = cars$speed)

plot(x = cars$speed, y = cars$dist, xlab = "Velocidade")
plot(x = cars$speed, y = cars$dist, xlab = "Velocidade", ylab = "Distância de freagem")

plot(cars, main = "Meu Grafico")

plot(cars, sub = "Subtitulo")
plot(cars, col = 2)
plot(cars, xlim = c(10, 15))
plot(cars, pch = 2)

# Usando a base de dados "mtcars", tambem do R
### Boxplot e Histograma

data(mtcars)
head(mtcars)
?boxplot
boxplot(formula = mpg ~ cyl, data = mtcars)
hist(mtcars$mpg)
?mtcars
## Observando a base de dados e elaborando Estatisticas descritivas

str(cars)
class(cars)
dim(cars)
nrow(cars)
ncol(cars)
object.size(cars)
names(cars)
head(cars)
head(cars, 10)
tail(cars, 15)
View(cars)

summary(cars)
mean(cars$speed)
mean(cars$dist)
median(cars$dist)
var(cars$speed)
cov(cars)
cov(cars$speed,cars$dist)

cor(cars)
cor(cars$speed,cars$dist)
?cor
cor(cars,method = c("pearson"))
cor(cars,method = c("spearman"))


### Usando uma base de dados do WOOLDRIDGE: wage1

install.packages("wooldridge")
library(wooldridge)
data("wage1")
?wage1
str(wage1)
names(wage1)
head(wage1)
View(wage1)

plot(wage1$wage,wage1$educ)

plot(x = wage1$wage, y = wage1$educ, xlab = "Educação", ylab = "Salário", main="Relação entre salário e educação")
plot(x = wage1$wage, y = wage1$educ, xlab = "Educação", ylab = "Salário", main="Relação entre salário e educação", col=2)
plot(x = wage1$wage, y = wage1$educ, xlab = "Educação", ylab = "Salário", main="Relação entre salário e educação", xlim = c(5, 20))
plot(x = wage1$wage, y = wage1$educ, xlab = "Educação", ylab = "Salário", main="Relação entre salário e educação",pch = 2)

boxplot(wage1$wage)
boxplot(formula = wage ~ educ, data = wage1)
hist(wage1$wage)
hist(wage1$educ)

## Estatisticas descritivas

summary(wage1)

summary(wage1$wage)
var(wage1)
cov(wage1)

# Alternativamente, pode-se fixar a base e usar os comandos diretamente (sem necessidade de indicar a base$)
attach(wage1)
summary(educ)
summary(female)
sum(female)


### Manipulando a base de dados 
# database: wage1
# dimens?es da base
dim(wage1)
head(wage1)

### Usando o Pacote dplyr; se ainda n?o tiver instalado, install.package()
# esse pacote tem 5 principais comandos, um deles ? o select
library(dplyr)

# para trabalhar com dplyr, tem que colocar a base desejada nesse formato abaixo:
wage1_aula<-tbl_df(wage1)

# para evitar confus?o, pode-se remover a base wage1
rm("wage1")

wage1_aula

?select
## Select pode ser usado para se trabalhar apenas com algumas vari?veis da base (sele??o por colunas)
select(wage1_aula, wage, educ, exper, female, married)
# trabalhando neste pacote, n?o ? preciso digitar mais: wage_aula$educ, etc; e com select as colunas ficam na ordem especificada no comando
select(wage1_aula,wage:exper)
select(wage1_aula,exper:wage)
# pode-se tbm especificar as vari?veis que n?o se quer mais
select(wage1_aula,-wage)
select(wage1_aula,-(female:south))

# Outro dos 5 comandos do Pacote dplyr: filter; usado para selecinar por linhas
filter(wage1_aula,female==1)
# caso a vari?vel estivesse no formato texto:
filter(wage1_aula,female=="mulher")

filter(wage1_aula,female==1,married==0)

# com 2 condi??es, uma ou outra pudendo ser satisfeita
filter(wage1_aula,south==1|west==1)

filter(wage1_aula,wage>5,female==1)

# excluindo os valores missing (NA no R)
filter(wage1_aula,!is.na(educ))

# criando uma subbase
wage1_aula_female<-filter(wage1_aula,female==1)
head(wage1_aula_female)

# Outro dos 5 comandos: arrange; para organizar os dados de formas diversas
# organizando a base anterior por salario crecente e decrescente
arrange(wage1_aula_female, wage)
arrange(wage1_aula_female,desc(wage))

# Segundo duas vari?veis
arrange(wage1_aula_female,married,educ)

# Outro dos 5 comandos do pacote dplyr: mutate
#alterar uma vari?vel, no caso, expressar ela em outra medida;
# criando nova coluna, nova vari?vel a partir de outra
mutate(wage1_aula_female,educ_meses=educ*12)

attach(wage1_aula_female)
summary(educ)

# Voltando a usar a base wage1
summary(wage1)

wage_test<-select(wage1,wage:exper)
wage_test
summary(wage_test)
wage_test_exper<-filter(wage_test,exper>10)
summary(wage_test_exper)

## Lista de exercícios:

## Livro de Estatística (na mão): um estudo de caso do cap. 2 e um do cap. 3  
## Exercícios em Computador, Wooldridge, Cap.1: C2 (base de dados: BWGHT) + outro à escolha
   # (para se acostumar a usar as bases do livro do Wooldridge)

## Resumir as estatísticas descritivas - incluindo gráficos e tabelas - de artigo em tema de interesse que usou regressão
   # Se houver no artigo, tabelas de var e cov





