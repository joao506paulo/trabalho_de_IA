#imports
import math
import numpy as np


#definir a função de ativação
def funcao_tanh (net):
    return np.tanh(net)

#definir a derivada da função de ativação
def derivada_tanh (f_net):
    return 1 - (f_net ** 2)

#definir a arquitetura da rede
def mlp_arquiteture (n_entradas, n_escondida, n_saida, f_ativacao, d_ativacao):
    
    modelo = {
        'tamanho_entrada' : n_entradas, 'tamanho_escondida' : n_escondida, 'tamanho_saida' : n_saida
    }

    modelo['pesos_escondida'] = np.random.uniform(low = -0.5, high = 0.5, size = (modelo['tamanho_escondida'], modelo['tamanho_entrada']+1))
    modelo['pesos_saida'] = np.random.uniform(low = -0.5, high = 0.5, size = (modelo['tamanho_saida'], modelo['tamanho_escondida']+1))
    modelo['f'] = f_ativacao
    modelo['df_dnet'] = d_ativacao
    
    return modelo

#teste
teste = mlp_arquiteture(2, 2, 1, funcao_tanh, derivada_tanh)
print(teste)

def mlp_forward (modelo, entrada):
    
    #camada escondida
    entrada_com_bias = np.append(entrada, 1)
    net_h = np.dot(modelo['pesos_escondida'], entrada_com_bias)
    f_net_h = modelo['f'](net_h)

    #camada de saída
    f_net_h_com_bias = np.append(f_net_h, 1)
    net_o = np.dot(modelo['pesos_saida'], f_net_h_com_bias)
    f_net_o = modelo['f'](net_o)
    
    #Resultado
    resultados = {'net_h' : net_h, 'f_net_h' : f_net_h, 'net_o' : net_o, 'f_net_o' : f_net_o}#, 'f_net_h_com_bias' : f_net_h_com_bias, 'entrada_com_bias' : entrada_com_bias}

    return resultados

entrada = np.array([0,1])
resultados_feedforward = mlp_forward(teste, entrada)
print(resultados_feedforward)


eta = 0.1 #acho que esse eta é o alfa
threshold = 0.001 #erro aceitavel
def mlp_backpropagation(modelo, dados, eta, threshold, num_epocas): 
    squaredError = 2 * threshold
    counter = 0
    num_amostras = dados.shape[0]
    while squaredError > threshold or counter < num_epocas:
        squaredError = 0
        for p in range(num_amostras):
            Xp = dados[p, :modelo['tamanho_entrada']].astype(float)
            Yp = dados[p, modelo['tamanho_entrada']:].astype(float)

            results = mlp_forward(modelo, Xp)
            Op = results['f_net_o']
            Error = Yp - Op

            squaredError = squaredError + sum(Error ** 2)

            #treinamento
            delta_o_p = Error * modelo['df_dnet'](results['f_net_o'])

            w_o_kj = modelo['pesos_saida'][:, :-1]
            delta_o_h = modelo['df_dnet'](results['f_net_h']) * np.dot(delta_o_p, w_o_kj)

            #ajuste dos pesos da camada de saida
            gradiente_saida = np.outer(delta_o_p, np.append(results['f_net_h'], 1))
            modelo['pesos_saida'] += eta * gradiente_saida

            #ajuste dos pesos da camada escondida
            gradiente_escondida = np.outer(delta_o_h, np.append(Xp, 1))
            modelo['pesos_escondida'] += eta * gradiente_escondida

        squaredError = squaredError / num_amostras
        print(squaredError)
        counter += 1

        resultados = { 'modelo' : modelo, 'counter' : counter}

    return resultados

#treinamento para XOR
treinado = mlp_backpropagation(teste,np.array([[1,1,-1], [-1,1,1], [1,-1,1], [-1, -1, -1]]), 0.1, 0.0001, 10000)

#testes
print(mlp_forward(treinado['modelo'], np.array([1,1]))['f_net_o'])
print(mlp_forward(treinado['modelo'], np.array([-1,-1]))['f_net_o'])
print(mlp_forward(treinado['modelo'], np.array([-1,1]))['f_net_o'])
print(mlp_forward(treinado['modelo'], np.array([1,-1]))['f_net_o'])