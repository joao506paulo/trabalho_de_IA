# Arquivo para implementar o algoritmo de treinamento da rede
from rede import camada_da_rede, neuronio
import numpy as np
training = [-1, -1, -1, 1, 1, -1, 1, 1]
resposta = [-1, 1, 1, 1]
N_teste = 4
#define a quantidade de neurônios em cada camada
Neuronios_Sensorial = 2    #alterar para cada problema
Neuronios_Camada_Escondida = 2
Neuronios_Saida = 1

#Define taxa de aprendizado
alfa = 1
#define número máximo de épocas
Max_epocas = 1000

#Lista de pesos e bias
Pesos_camada_escondida = np.zeros((Neuronios_Camada_Escondida, Neuronios_Sensorial))
#print(Pesos_camada_escondida)
Bias_camada_escondida = np.zeros((Neuronios_Camada_Escondida))
Pesos_saida = []
Bias_saida = []

#step 0
#for i in range(Neuronios_Camada_Escondida):
#   Pesos_camada_escondida.append(0)
#   Pesos_camada_escondida.append(0)
#   Bias_camada_escondida.append(0)

#for i in range(Neuronios_Saida):
#    Pesos_saida.append(0)
#    Bias_saida.append(0)

x_i = []
theta = 2
novos_pesos = []
#step 1
epocas = 0

#stop_condition = False
#while not(stop_condition):
#    mudou = False
#    #step 2
#    for i in range(N_teste):
#        #step 3
#        y_in = Bias_camada_escondida[0]
#        for j in range(Neuronios_Sensorial):
#            x_i.append(training[j+i*Neuronios_Sensorial])
#            #step 4
#            y_in = Pesos_camada_escondida[j]*x_i[j]
#        if(y_in > theta):
#            y = 1
#        elif(y_in < theta and y_in > theta*(-1)):
#            y = 0
#        elif (y_in < theta*(-1)):
#            y = -1
#        if(y != resposta[i]):
#            mudou = True
#            for j in range(Neuronios_Sensorial):
#                Pesos_camada_escondida[j] = Pesos_camada_escondida[j] + alfa*resposta[i]*x_i[j]
#    
#    epocas = epocas+1
#    #step 6
#    if epocas > Max_epocas or not(mudou):
#        stop_condition = True



entrada = []
ultima_alteracao = 0
stop_condition = False
while not(stop_condition):
    for i in range(N_teste):
        entrada.clear()
        for j in range(Neuronios_Sensorial):
            entrada.append(training[j+i*Neuronios_Sensorial])
        resultado = camada_da_rede(Neuronios_Camada_Escondida, Pesos_camada_escondida, entrada, Bias_camada_escondida)
        for a in range(len(resultado)):
            if(resposta[i] == resultado[a]):
                ultima_alteracao = ultima_alteracao+1
            else:
                for j in range(Neuronios_Camada_Escondida):
                    for k in range(Neuronios_Sensorial):
                        Pesos_camada_escondida[j][k] = Pesos_camada_escondida[j][k] + alfa*resposta[i]*entrada[k]
    epocas = epocas+1
    if epocas > Max_epocas or ultima_alteracao > 5:
        stop_condition = True

print(epocas)
print(ultima_alteracao)
print(Pesos_camada_escondida)
print(Bias_camada_escondida)