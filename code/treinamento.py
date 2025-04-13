# Arquivo para implementar o algoritmo de treinamento da rede
from rede import neuronio

training = [-1, -1, -1, 1, 1, -1, 1, 1]
resposta = [-1, -1, -1, 1]
N_teste = 4
#define a quantidade de neurônios em cada camada
Neuronios_Sensorial = 2    #alterar para cada problema
Neuronios_Camada_Escondida = 1
Neuronios_Saida = 1

#Define taxa de aprendizado
alfa = 1
#define número máximo de épocas
Max_epocas = 50

#Lista de pesos e bias
Pesos_camada_escondida = []
Bias_camada_escondida = []
Pesos_saida = []
Bias_saida = []

#step 0
for i in range(Neuronios_Camada_Escondida):
    Pesos_camada_escondida.append(0)
    Pesos_camada_escondida.append(0)
    Bias_camada_escondida.append(0)

for i in range(Neuronios_Saida):
    Pesos_saida.append(0)
    Bias_saida.append(0)

x_i = []
theta = 2
novos_pesos = []
#step 1
epocas = 0

stop_condition = False
while not(stop_condition):
    mudou = False
    #step 2
    for i in range(N_teste):
        #step 3
        y_in = Bias_camada_escondida[0]
        for j in range(Neuronios_Sensorial):
            x_i.append(training[j+i*Neuronios_Sensorial])
            #step 4
            y_in = Pesos_camada_escondida[j]*x_i[j]
        if(y_in > theta):
            y = 1
        elif(y_in < theta and y_in > theta*(-1)):
            y = 0
        elif (y_in < theta*(-1)):
            y = -1
        if(y != resposta[i]):
            mudou = True
            for j in range(Neuronios_Sensorial):
                Pesos_camada_escondida[j] = Pesos_camada_escondida[j] + alfa*resposta[i]*x_i[j]
    
    epocas = epocas+1
    #step 6
    if epocas > Max_epocas or not(mudou):
        stop_condition = True

for i in Pesos_camada_escondida:
    print(i)

neuronio(Pesos_camada_escondida, [1,1])