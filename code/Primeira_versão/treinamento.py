# Arquivo para implementar o algoritmo de treinamento da rede
from rede import camada_da_rede, neuronio
import numpy as np





def treinamento (Max_epocas, alfa, Neuronios_Sensorial, Neuronios_Camada_Escondida, Neuronios_Saida, training, resposta, N_teste):
    #Lista de pesos e bias
    Pesos_camada_escondida = np.zeros((Neuronios_Camada_Escondida, Neuronios_Sensorial))
    Bias_camada_escondida = np.zeros((Neuronios_Camada_Escondida))
    entrada = []
    epocas = 0
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
    return Pesos_camada_escondida, Bias_camada_escondida, resultado

