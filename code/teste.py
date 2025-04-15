from rede import camada_da_rede, neuronio
from treinamento import treinamento

training = [-1, -1, -1, 1, 1, -1, 1, 1]
resposta = [-1, 1, 1, -1]
N_teste = 4
#define a quantidade de neurônios em cada camada
Neuronios_Sensorial = 2    #alterar para cada problema
Neuronios_Camada_Escondida = 2
Neuronios_Saida = 1

#Define taxa de aprendizado
alfa = 1
#define número máximo de épocas
Max_epocas = 1000

p, b, r = treinamento(Max_epocas, alfa, Neuronios_Sensorial, Neuronios_Camada_Escondida, Neuronios_Saida, training, resposta, N_teste)
print(p)
print(b)
print(r)

print(camada_da_rede(2, p, [1, 1], b))