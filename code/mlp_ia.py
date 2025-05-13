import numpy as np

# Funções de ativação e suas derivadas
def tanh(x):
    return np.tanh(x)

def derivada_tanh(x):
    return 1 - np.tanh(x)**2

# Arquitetura da rede
def mlp_arquitetura(n_entradas, n_camadas_escondidas, n_saida, funcoes_ativacao, derivadas_ativacao):
    """
    Define a arquitetura da rede MLP.

    Args:
        n_entradas (int): Número de neurônios na camada de entrada.
        n_camadas_escondidas (list of int): Lista com o número de neurônios em cada camada escondida.
        n_saida (int): Número de neurônios na camada de saída.
        funcoes_ativacao (list of callables): Lista das funções de ativação para cada camada.
        derivadas_ativacao (list of callables): Lista das derivadas das funções de ativação.

    Returns:
        dict: Dicionário contendo a arquitetura da rede.
    """

    modelo = {
        'n_entradas': n_entradas,
        'n_camadas_escondidas': n_camadas_escondidas,
        'n_saida': n_saida,
        'funcoes_ativacao': funcoes_ativacao,
        'derivadas_ativacao': derivadas_ativacao,
        'n_camadas': len(n_camadas_escondidas) + 1  # Total de camadas (escondidas + saída)
    }

    modelo['pesos'] = []
    tamanhos_camadas = [n_entradas] + n_camadas_escondidas + [n_saida]
    for i in range(modelo['n_camadas']):
        modelo['pesos'].append(np.random.uniform(low=-0.5, high=0.5, size=(tamanhos_camadas[i+1], tamanhos_camadas[i] + 1)))  # +1 para o bias

    return modelo

# Forward propagation
def mlp_forward(modelo, entrada):
    """
    Realiza a propagação forward na rede MLP.

    Args:
        modelo (dict): Dicionário contendo a arquitetura da rede.
        entrada (numpy.ndarray): Vetor de entrada.

    Returns:
        dict: Dicionário contendo os resultados de cada camada.
    """

    resultados = {'entradas': [np.append(entrada, 1)]}  # Armazena as entradas com bias
    for i in range(modelo['n_camadas']):
        net = np.dot(modelo['pesos'][i], resultados['entradas'][-1])
        f_net = modelo['funcoes_ativacao'][i](net)
        resultados['entradas'].append(np.append(f_net, 1) if i < modelo['n_camadas'] - 1 else f_net)  # Adiciona bias para camadas escondidas

    return resultados

# Backpropagation
def mlp_backpropagation(modelo, dados, learning_rate, threshold, num_epocas, dados_validacao=None):
    """
    Realiza o algoritmo de backpropagation para treinar a rede MLP.

    Args:
        modelo (dict): Dicionário contendo a arquitetura da rede.
        dados (numpy.ndarray): Dados de treinamento (X, Y).
        learning_rate (float): Taxa de aprendizado.
        threshold (float): Erro mínimo aceitável.
        num_epocas (int): Número máximo de épocas de treinamento.
        dados_validacao (numpy.ndarray, optional): Dados de validação. Defaults to None.

    Returns:
        dict: Dicionário contendo o modelo treinado e informações sobre o treinamento.
    """

    num_amostras = dados.shape[0]
    for epoca in range(num_epocas):
        erro_total = 0
        for p in range(num_amostras):
            Xp = dados[p, :modelo['n_entradas']].astype(float)
            Yp = dados[p, modelo['n_entradas']:].astype(float)

            resultados_forward = mlp_forward(modelo, Xp)
            Op = resultados_forward['entradas'][-1]
            erro = Yp - Op
            erro_total += np.sum(erro**2)

            # Backpropagation
            deltas = [erro * modelo['derivadas_ativacao'][-1](resultados_forward['entradas'][-1])]  # Delta da camada de saída
            for i in range(modelo['n_camadas'] - 2, -1, -1):  # Camadas escondidas (reverso)
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
    """Calcula o erro quadrático médio para um conjunto de dados."""
    num_amostras = dados.shape[0]
    erro_total = 0
    for p in range(num_amostras):
        Xp = dados[p, :modelo['n_entradas']].astype(float)
        Yp = dados[p, modelo['n_entradas']:].astype(float)
        Op = mlp_forward(modelo, Xp)['entradas'][-1]
        erro_total += np.sum((Yp - Op)**2)
    return erro_total / num_amostras

# Exemplo de uso
n_entradas = 2
n_camadas_escondidas = [2]
n_saida = 1
funcoes_ativacao = [tanh, tanh]  # Uma para cada camada (escondidas e saída)
derivadas_ativacao = [derivada_tanh, derivada_tanh]

modelo = mlp_arquitetura(n_entradas, n_camadas_escondidas, n_saida, funcoes_ativacao, derivadas_ativacao)
dados_treinamento = np.array([[1, 1, -1], [-1, 1, 1], [1, -1, 1], [-1, -1, -1]])

# Simulação de dados de validação (substitua pelos seus dados reais)
dados_validacao = np.array([[0.9, 0.8, -0.9], [0.7, -0.7, 0.9]])

resultado_treinamento = mlp_backpropagation(modelo, dados_treinamento, learning_rate=0.1, threshold=0.001, num_epocas=100000, dados_validacao=dados_validacao)

print("Modelo treinado:", resultado_treinamento['modelo'])

#testes
print(mlp_forward(resultado_treinamento['modelo'], np.array([1,1]))['entradas'][-1])
print(mlp_forward(resultado_treinamento['modelo'], np.array([-1,-1]))['entradas'][-1])
print(mlp_forward(resultado_treinamento['modelo'], np.array([-1,1]))['entradas'][-1])
print(mlp_forward(resultado_treinamento['modelo'], np.array([1,-1]))['entradas'][-1])