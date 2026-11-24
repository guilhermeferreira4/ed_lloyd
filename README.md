README · MD
# ED+Lloyd: Evolução Diferencial na inicialização do K-Means
 
Código dos experimentos do trabalho de conclusão de curso **"Aplicação de Evolução Diferencial na Otimização de Agrupamentos via K-Means"**.
 
O trabalho testa se a Evolução Diferencial (ED) consegue escolher centroides iniciais melhores para o K-Means do que o sorteio aleatório e o k-means++. O método proposto, chamado **ED+Lloyd**, usa a ED para minimizar a inércia em função das posições dos centroides e depois refina o melhor indivíduo com o algoritmo de Lloyd. Este repositório contém o código que compara o ED+Lloyd com três configurações de referência em quatro conjuntos de dados reais e um sintético.
