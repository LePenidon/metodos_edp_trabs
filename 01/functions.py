# Importando bibliotecas utilizadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d


def imprime_matriz(A):

    try:
        A = sparse.csr_matrix.todense(A)
        A = pd.DataFrame(A)

        pd.set_option('display.max_rows', None)
        # determining the name of the file
        file_name = 'matriz.xlsx'
        # saving the excel
        A.to_excel(file_name)
    except:
        A = pd.DataFrame(A)

        pd.set_option('display.max_rows', None)
        # determining the name of the file
        file_name = 'matriz.xlsx'
        # saving the excel
        A.to_excel(file_name)

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
    I_h = np.zeros((tam_malha, tam_malha))
    I_h[0][0] = 1/h/h
    I_h[-1][-1] = 1/h/h
    I_h += I_1

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

    F = np.zeros((tam_malha**2, 1))

    for i in range(tam_malha+1):
        F[i] = np.sin(a + i*h)

    F = sparse.csr_matrix(F)

    return (sparse.linalg.spsolve(A, F))


# Função que plota os valores de referência
def plot_referencia(a, b, tam_malha, valores_ref):
    # Cria um conjunto de pontos na superfície
    x = np.linspace(a, b, tam_malha*tam_malha)
    y = np.linspace(a, b, tam_malha*tam_malha)

    X, Y = np.meshgrid(x, y)

    # Criamos a interpolacao
    # Definindo os pontos a serem aproximados
    x_spline = np.linspace(a, b, tam_malha*tam_malha)
    y_spline = valores_ref
    f = interp1d(x_spline, y_spline, kind='cubic')

    # Criando a spline cúbica
    Z = f(X)

    # Cria uma figura 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plota a superfície
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Configura os rótulos dos eixos
    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.set_zlabel('Eixo Z')

    # plt.show()
    plt.savefig('resultados/ref.png')


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

        erro = np.linalg.norm(
            valores_h[i] - valores_referencia)/np.linalg.norm(valores_referencia)

        truncamento.append(erro)

    return truncamento


# Função que cria a tabela com os resultados
def tabela_resultados(h, erro):
    print(pd.DataFrame({'h': h, 'Erro': erro}).to_latex(index=False))


# Função que plota o gráfico de convergência
def plot_convergencia(h, erro):
    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.subplots()
    # ax.plot(h, erro,  marker='o')
    # ax.loglog()
    # ax.grid(color='green', linestyle='--', linewidth=0.5,)
    # ax.set_xlabel('h')
    # ax.set_ylabel('erro')

    # for i, j in zip(h, erro):
    #     ax.annotate(str(j), xy=(i, j))

    # posicoes = [i+1 for i in range(len(h))]

    # plt.figure(figsize=(8, 6))
    # plt.scatter(posicoes, erro, c='red', marker='o', label='Pontos')
    # plt.xlabel('H', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    # plt.ylabel('Erro', fontdict={
    #            'fontsize': 14, 'fontweight': 'bold'})
    # plt.title('Valor da Função Objetivo', fontdict={
    #           'fontsize': 16, 'fontweight': 'bold'})
    # plt.grid(color='gray', linestyle='--')

    plt.savefig('resultados/convergencia.png')
