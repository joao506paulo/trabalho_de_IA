import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

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
        # Inicialização de Xavier para tanh
        limite = np.sqrt(6 / (tamanhos_camadas[i] + tamanhos_camadas[i+1]))
        modelo['pesos'].append(np.random.uniform(low=-limite, high=limite, size=(tamanhos_camadas[i+1], tamanhos_camadas[i] + 1)))

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
def mlp_backpropagation(modelo, X_treinamento, Y_treinamento, learning_rate, threshold, num_epocas, X_validacao=None, Y_validacao=None):
    num_amostras = X_treinamento.shape[0]
    erros_por_epoca = []

    for epoca in range(num_epocas):
        erro_total = 0
        for p in range(num_amostras):
            Xp = X_treinamento[p].astype(float)
            Yp = Y_treinamento[p].astype(float)

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
        erros_por_epoca.append(mse)
        print(f"Época {epoca + 1}, Erro Quadrático Médio (Treino): {mse}")

        if mse <= threshold:
            print(f"Treinamento interrompido: Erro abaixo do threshold ({threshold})")
            break

        # Validação (opcional)
        if X_validacao is not None and Y_validacao is not None:
            erro_validacao = calcular_erro(modelo, X_validacao, Y_validacao)
            print(f"  Erro no conjunto de validação: {erro_validacao}")

    return {'modelo': modelo, 'epocas_treinadas': epoca + 1, 'erros_por_epoca': erros_por_epoca}

def calcular_erro(modelo, X, Y):
    num_amostras = X.shape[0]
    erro_total = 0
    for p in range(num_amostras):
        Xp = X[p].astype(float)
        Yp = Y[p].astype(float)
        Op = mlp_forward(modelo, Xp)['entradas'][-1]
        erro_total += np.sum((Yp - Op)**2)
    return erro_total / num_amostras

def mlp_predict(modelo, X, limiar):
    """Realiza a previsão da rede para um conjunto de entradas."""
    num_amostras = X.shape[0]
    saidas_numericas = []
    saidas_letras = []

    limiar_positivo = limiar
    limiar_negativo = (-1)*limiar - 0.9

    for p in range(num_amostras):
        Xp = X[p].astype(float)
        Op = mlp_forward(modelo, Xp)['entradas'][-1]
        saidas_numericas.append(Op.tolist())

    for a in range(num_amostras):
        max_index = np.argmax(saidas_numericas[a])  # Índice do valor máximo
        letra_encontrada = chr(ord('A') + max_index)  # Converte o índice para a letra
        saidas_letras.append(letra_encontrada)

    return saidas_numericas, saidas_letras

# Carregar os dados
caminho_dados = "../Caracteres-completos/"
X = np.load(caminho_dados + "X.npy")
Y = np.load(caminho_dados + "Y_classe.npy")

print("Formato de X:", X.shape)
print("Formato de Y:", Y.shape)

# Separar os últimos 130 caracteres para teste
X_treino_val = X[:-130]
Y_treino_val = Y[:-130]
X_teste = X[-130:]
Y_teste = Y[-130:]

print("Tamanho de X_treino_val:", len(X_treino_val))
print("Tamanho de Y_treino_val:", len(Y_treino_val))
print("Tamanho de X_teste:", len(X_teste))
print("Tamanho de T_teste:", len(Y_teste))

# Dividir os dados em treinamento e validação
X_treinamento, X_validacao, Y_treinamento, Y_validacao = train_test_split(
    X_treino_val, Y_treino_val, test_size=0.2, random_state=42
)

print("Tamanho de X_treino:", len(X_treinamento))
print("Tamanho de Y_treino:", len(Y_treinamento))
print("Tamanho de X_validacao:", len(X_validacao))
print("Tamanho de Y_validacao:", len(Y_validacao))

# Parâmetros da rede
n_entradas = 120
n_camadas_escondidas = [60]
n_saida = 26
funcoes_ativacao = [tanh] * len(n_camadas_escondidas) + [tanh]
derivadas_ativacao = [derivada_tanh] * len(n_camadas_escondidas) + [derivada_tanh]

# Criar e treinar o modelo
modelo = mlp_arquitetura(n_entradas, n_camadas_escondidas, n_saida, funcoes_ativacao, derivadas_ativacao)
pesos_iniciais = [p.copy().tolist() for p in modelo['pesos']]
resultado_treinamento = mlp_backpropagation(modelo, X_treinamento, Y_treinamento, learning_rate=0.005, threshold=0.001, num_epocas=1000, X_validacao=X_validacao, Y_validacao=Y_validacao)
pesos_finais = [p.copy().tolist() for p in modelo['pesos']]

# Fazer previsões
_, saidas_validacao_l = mlp_predict(modelo, X_validacao, 0.4)
_, saidas_teste_l = mlp_predict(modelo, X_teste, 0.4)

# Converter Y_validacao e Y_teste para letras
def converter_para_letras(Y):
    letras = []
    for one_hot in Y:
        index = np.argmax(one_hot)
        letras.append(chr(ord('A') + index))
    return letras

Y_validacao_letras = converter_para_letras(Y_validacao)
Y_teste_letras = converter_para_letras(Y_teste)

# Avaliação do modelo
def avaliar_modelo(Y_verdadeiro, Y_predito, conjunto_nome="Validação"):
    print(f"--- Resultados no Conjunto de {conjunto_nome} ---")
    print(classification_report(Y_verdadeiro, Y_predito))
    
    matriz_confusao = confusion_matrix(Y_verdadeiro, Y_predito)
    
    # Visualizar a matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(list(set(Y_verdadeiro))), 
                yticklabels=sorted(list(set(Y_verdadeiro))))
    plt.title(f'Matriz de Confusão - Conjunto de {conjunto_nome}')
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Predito')
    plt.show()

    acuracia = accuracy_score(Y_verdadeiro, Y_predito)
    precisao = precision_score(Y_verdadeiro, Y_predito, average='weighted')
    revocacao = recall_score(Y_verdadeiro, Y_predito, average='weighted')
    f1 = f1_score(Y_verdadeiro, Y_predito, average='weighted')

    print(f"Acurácia: {acuracia:.4f}")
    print(f"Precisão: {precisao:.4f}")
    print(f"Revocação: {revocacao:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\n")

# Avaliar o modelo nos conjuntos de validação e teste
avaliar_modelo(Y_validacao_letras, saidas_validacao_l, "Validação")
avaliar_modelo(Y_teste_letras, saidas_teste_l, "Teste")

# Salvar os arquivos de saída
hiperparametros = {
    'n_entradas': n_entradas,
    'n_camadas_escondidas': n_camadas_escondidas,
    'n_saida': n_saida,
    'funcoes_ativacao': [f.__name__ for f in funcoes_ativacao],
    'derivadas_ativacao': [df.__name__ for df in derivadas_ativacao],
    'learning_rate': 0.005,
    'threshold': 0.15,
    'num_epocas': 1000
}

with open("hiperparametros.json", "w") as f:
    json.dump(hiperparametros, f, indent=4)

with open("pesos_iniciais.json", "w") as f:
    json.dump(pesos_iniciais, f, indent=4)

with open("pesos_finais.json", "w") as f:
    json.dump(pesos_finais, f, indent=4)

with open("erros_por_epoca.txt", "w") as f:
    for erro in resultado_treinamento['erros_por_epoca']:
        f.write(str(erro) + "\n")

with open("saidas_validacao.txt", "w") as f:
    for saida in saidas_validacao_l:
        f.write(str(saida) + "\n")

with open("saidas_teste.txt", "w") as f:
    for saida in saidas_teste_l:
        f.write(str(saida) + "\n")