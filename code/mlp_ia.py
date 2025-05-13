import numpy as np
import pandas as pd  # Importar pandas

# Funções de ativação e suas derivadas
def tanh(x):
    return np.tanh(x)

def derivada_tanh(x):
    return 1 - np.tanh(x)**2

# Arquitetura da rede
def mlp_arquitetura(n_entradas, n_camadas_escondidas, n_saida, funcoes_ativacao, derivadas_ativacao):
    modelo = {
        'n_entradas': n_entradas,
        'n_camadas_escondidas': n_camadas_escondidas,
        'n_saida': n_saida,
        'funcoes_ativacao': funcoes_ativacao,
        'derivadas_ativacao': derivadas_ativacao,
        'n_camadas': len(n_camadas_escondidas) + 1
    }

    modelo['pesos'] = []
    tamanhos_camadas = [n_entradas] + n_camadas_escondidas + [n_saida]
    for i in range(modelo['n_camadas']):
        modelo['pesos'].append(np.random.uniform(low=-0.5, high=0.5, size=(tamanhos_camadas[i+1], tamanhos_camadas[i] + 1)))

    return modelo

# Forward propagation
def mlp_forward(modelo, entrada):
    resultados = {'entradas': [np.append(entrada, 1)]}
    for i in range(modelo['n_camadas']):
        net = np.dot(modelo['pesos'][i], resultados['entradas'][-1])
        f_net = modelo['funcoes_ativacao'][i](net)
        resultados['entradas'].append(np.append(f_net, 1) if i < modelo['n_camadas'] - 1 else f_net)

    return resultados

# Backpropagation
def mlp_backpropagation(modelo, dados_treinamento, learning_rate, threshold, num_epocas, dados_validacao=None):
    num_amostras = dados_treinamento.shape[0]
    for epoca in range(num_epocas):
        erro_total = 0
        for p in range(num_amostras):
            Xp = dados_treinamento.iloc[p, :modelo['n_entradas']].values.astype(float) # Usar .iloc e .values
            Yp = dados_treinamento.iloc[p, modelo['n_entradas']:].values.astype(float)

            resultados_forward = mlp_forward(modelo, Xp)
            Op = resultados_forward['entradas'][-1]
            erro = Yp - Op
            erro_total += np.sum(erro**2)

            # Backpropagation
            deltas = [erro * modelo['derivadas_ativacao'][-1](resultados_forward['entradas'][-1])]
            for i in range(modelo['n_camadas'] - 2, -1, -1):
                deltas.insert(0, modelo['derivadas_ativacao'][i](resultados_forward['entradas'][i+1][:-1]) * np.dot(modelo['pesos'][i+1].T[:-1], deltas[-1]))

            # Atualização dos pesos
            for i in range(modelo['n_camadas']):
                modelo['pesos'][i] += learning_rate * np.outer(deltas[i], resultados_forward['entradas'][i])

        mse = erro_total / num_amostras
        print(f"Época {epoca + 1}, Erro Quadrático Médio: {mse}")

        if mse <= threshold:
            print(f"Treinamento interrompido: Erro abaixo do threshold ({threshold})")
            break

        # Validação (opcional)
        if dados_validacao is not None:
            erro_validacao = calcular_erro(modelo, dados_validacao)
            print(f"  Erro no conjunto de validação: {erro_validacao}")

    return {'modelo': modelo, 'epocas_treinadas': epoca + 1}

def calcular_erro(modelo, dados):
    num_amostras = dados.shape[0]
    erro_total = 0
    for p in range(num_amostras):
        Xp = dados.iloc[p, :modelo['n_entradas']].values.astype(float)  # Usar .iloc e .values
        Yp = dados.iloc[p, modelo['n_entradas']:].values.astype(float)
        Op = mlp_forward(modelo, Xp)['entradas'][-1]
        erro_total += np.sum((Yp - Op)**2)
    return erro_total / num_amostras

# Carregar os dados
caminho_dados = "../Caracteres-faussett/"  # Caminho relativo para a pasta de dados
dados_treinamento = pd.read_csv(caminho_dados + "caracteres-limpo.csv", header=None) # Assume que não há cabeçalho
dados_validacao = pd.read_csv(caminho_dados + "caracteres-ruido.csv", header=None)   # Assume que não há cabeçalho

# Parâmetros da rede
n_entradas = dados_treinamento.shape[1] - 7  # Ajuste conforme seus dados (número de features)
n_camadas_escondidas = [10]
n_saida = 7  # Ajuste conforme seus dados (número de saídas)
funcoes_ativacao = [tanh] * len(n_camadas_escondidas) + [tanh]
derivadas_ativacao = [derivada_tanh] * len(n_camadas_escondidas) + [derivada_tanh]

# Criar e treinar o modelo
modelo = mlp_arquitetura(n_entradas, n_camadas_escondidas, n_saida, funcoes_ativacao, derivadas_ativacao)
resultado_treinamento = mlp_backpropagation(modelo, dados_treinamento, learning_rate=0.1, threshold=0.001, num_epocas=100, dados_validacao=dados_validacao)

print("Modelo treinado:", resultado_treinamento['modelo'])

print(mlp_forward(resultado_treinamento['modelo'], np.array([-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,1,-1,1]))['entradas'][-1])