# Arquivo para representar a arquitetura da rede neural

#define a quantidade de neurônios em cada camada
Neuronios_Sensorial = 1    #alterar para cada problema
Neuronios_Camada_Escondida = 2
Neuronios_Saida = 1
theta = 2

def neuronio(Pesos_camada_escondida, x_i):
    
    y_in = Pesos_camada_escondida[0]*x_i[0] + Pesos_camada_escondida[1]*x_i[1]
    if(y_in > theta):
        y = 1
    elif(y_in < theta and y_in > theta*(-1)):
        y = 0
    elif (y_in < theta*(-1)):
        y = -1
    print(y)
