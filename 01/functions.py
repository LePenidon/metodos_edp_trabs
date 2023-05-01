# Importando bibliotecas utilizadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


def imprime_matriz(A):
    format = "%.1f"

    try:
        # Criar um DataFrame pandas a partir da matriz densa
        df = pd.DataFrame(A.toarray())

        # Configurar opções de exibição do pandas para mostrar toda a matriz
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        # Imprimir a matriz usando o método to_string()
        # print(df.to_string(index=False, header=False))
        np.savetxt('matriz.txt', df, fmt=format)
    except:
        df = pd.DataFrame(A)

        # Configurar opções de exibição do pandas para mostrar toda a matriz
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        # Imprimir a matriz usando o método to_string()
        # print(df.to_string(index=False, header=False))
        np.savetxt('matriz.txt', df, fmt=format)

    return


# Função que dá o tamanho da malha de acordo com o tamanho de h escolhido
def tamanho_malha(a, b, h):
    n = (b-a)/h + 1

    return int(n)


def aprox_u(a, b, h, tam_malha, k_2):
    # Matriz identidade
    I = np.identity(tam_malha)

    # Matriz com condições de Dirichlet
    D_h = np.identity(tam_malha)

    # Matriz auxiliar 1
    I_1 = np.identity(tam_malha)
    I_1[0][0] = 0
    I_1[-1][-1] = 0

    # Matriz auxiliar 2
    I_2 = np.zeros((tam_malha, tam_malha))

    # Matriz auxiliar 1
    # I_h = np.zeros((tam_malha, tam_malha))
    # I_h[0][0] = 1/h/h
    # I_h[-1][-1] = 1/h/h
    # I_h += I_1

    I_h = I*(1/h/h)

    # Matriz T com condições de Neumann e com coeficientes do Método de diferenças finitas
    T = np.zeros((tam_malha, tam_malha))
    T[0][0] = (-(4-h))/(h*h) + k_2
    T[-1][-1] = (-(4+h))/(h*h) + k_2
    T[0][1] = 2/h/h
    T[-1][-2] = 2/h/h

    for i in range(1, tam_malha - 1):
        T[i][i - 1] = 1/h/h
        T[i][i] = (-4)/(h*h) + k_2
        T[i][i + 1] = 1/h/h

        I_2[i][i - 1] = 1
        I_2[i][i + 1] = 1

    # Transformando as matrizes criadas em matrizes esparsas
    I = sparse.csr_matrix(I)
    I_1 = sparse.csr_matrix(I_1)
    I_2 = sparse.csr_matrix(I_2)
    T = sparse.csr_matrix(T)
    D_h = sparse.csr_matrix(D_h)

    # Matriz A de resolução do problema linear (A linha)
    A = sparse.kron((I - I_1), D_h) + sparse.kron(I_1, T) + \
        sparse.kron(I_2, I_h)

    imprime_matriz(A)

    F = np.zeros((tam_malha**2, 1))

    for i in range(tam_malha+1):
        F[i] = np.sin(a + i*h)

    F = sparse.csr_matrix(F)

    return (sparse.linalg.spsolve(A, F))


# Função que plota os valores de referência
def plot_referencia(a, b, tam_malha, valores_ref):
    # Cria um conjunto de pontos na superfície
    x = np.linspace(a, b, tam_malha)
    y = np.linspace(a, b, tam_malha)

    z = valores_ref.reshape((tam_malha, tam_malha))

    X, Y = np.meshgrid(x, y)

    # Criando figura e eixos 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotando gráfico de superfície
    surf = ax.plot_surface(X, Y, z, cmap=plt.cm.viridis,
                           rcount=tam_malha, ccount=tam_malha)

    # Configurando eixos e título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Gráfico de Superfície com Interpolação Cúbica')

    # Adicionando barra de cores
    fig.colorbar(surf)

    # plt.show()
    plt.savefig('resultados/ref.png')

    return


# Função que calcula as malhas
def malhas_calc(a, b, h):
    malhas = []

    for i in h:
        malhas.append(tamanho_malha(a, b, i))

    return malhas


# Função que calcula os valores_h
def valores_h_calc(a, b, h, tam_malha, k_2):
    valores_h = []

    for i in h:
        valores_h.append(aprox_u(a, b, i, tam_malha, k_2))

    return valores_h


# Função que calcula o vetor de truncamento
def truncamento_vet(h, valores_referencia, valores_h):
    truncamento = []

    for i in range(0, len(h)):

        erro_absoluto = np.abs(valores_referencia - valores_h[i])
        norma_max = np.amax(np.abs(valores_referencia))

        # Cálculo do erro de aproximação relativo
        erro_relativo = np.amax(erro_absoluto) / norma_max

        # erro = np.linalg.norm(
        #     valores_h[i] - valores_referencia)/np.linalg.norm(valores_referencia)

        truncamento.append(erro_relativo)

    return truncamento


# Função que cria a tabela com os resultados
def tabela_resultados(h, erro):
    print(pd.DataFrame({'h': h, 'Erro': erro}).to_latex(index=False))


# Função que plota o gráfico de convergência
def plot_convergencia(h, erro):

    plt.figure(figsize=(10, 7))
    plt.xlabel('H', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.ylabel('Erro', fontdict={
               'fontsize': 14, 'fontweight': 'bold'})
    plt.title('Erro/H', fontdict={
              'fontsize': 16, 'fontweight': 'bold'})

    # plt.plot(h, erro, c='red')
    plt.loglog(h, erro, color='blue')

    plt.savefig('resultados/convergencia.png')
