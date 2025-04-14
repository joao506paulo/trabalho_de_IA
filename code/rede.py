# Arquivo para representar a arquitetura da rede neural

#define a quantidade de neurônios em cada camada
Neuronios_Sensorial = 1    #alterar para cada problema
Neuronios_Camada_Escondida = 2
Neuronios_Saida = 1
theta = 2

#função de ativação, pode (e deve ser mudada)
#recebe y_in calculado pelo neuronio e devolve o resultado da função de ativação
def funcao_ativacao(y_in):
    y = 0
    if(y_in >= theta):
        y = 1
    #elif(y_in < theta and y_in > theta*(-1)):
        #y = 0
    elif (y_in < theta):
        y = -1
    return y

#recebe uma lista de pesos e entradas e seu bias e retorna o resultado da função de ativação
def neuronio(numero_de_entradas, Pesos, entradas, bias):
    y_in = bias
    for i in range(numero_de_entradas):
        y_in = y_in + Pesos[i]*entradas[i]
    return funcao_ativacao(y_in)

#recebe uma lista de pesos e entradas para cada neurônio e seu bias, e o número de neurônios e returna uma lista com o resultado de cada neurônio
def camada_da_rede(numero_de_neuronios, Pesos_neuronios, entradas_neuronios, bias_neuronios):
    resultado = []
    for i in range(numero_de_neuronios):
        resultado.append(neuronio(len(entradas_neuronios), Pesos_neuronios[i], entradas_neuronios, bias_neuronios[i]))
    return resultado