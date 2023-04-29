# Importando bibliotecas utilizadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


# Função que dá o tamanho da malha de acordo com o tamanho de h escolhido
def tamanho_malha(a, b, h):
    n = (b-a)/h + 1

    return int(n)


def aprox_u(a, b, h):
    n = tamanho_malha(a, b, h)
    A = np.identity(n*n)

    for i in range(n):
        for j in range(n):
            if (j == 0 and (i > 0 and i < n)):
                A[i][j] = 3

    return 1


# Função que calcula o valor aproximado da função nos pontos de discretização da malha
# def aprox_u(a, b, h):
#     # Tamanho da malha
#     tam_malha = tamanho_malha(a, b, h)

#     # Matriz identidade
#     I = np.identity(tam_malha)

#     # Matriz com condições de Dirichlet
#     D_h = np.identity(tam_malha)*(h**2)

#     # Matriz auxiliar 1
#     I_1 = np.identity(tam_malha)
#     I_1[0][0] = 0
#     I_1[-1][-1] = 0

#     # Matriz auxiliar 2
#     I_2 = np.zeros((tam_malha, tam_malha))

#     # Matriz auxiliar 1
#     I_h = np.zeros((tam_malha, tam_malha))
#     I_h[0][0] = h/2
#     I_h[-1][-1] = h/2
#     I_h += I_1

#     # Matriz T com condições de Neumann e com coeficientes do Método de diferenças finitas
#     T = np.zeros((tam_malha, tam_malha))
#     T[0][0] = -2*h
#     T[-1][-1] = -2*h
#     T[0][1] = h
#     T[-1][-2] = h

#     for i in range(1, tam_malha - 1):
#         T[i][i - 1] = 1
#         T[i][i] = -4
#         T[i][i + 1] = 1

#         I_2[i][i - 1] = 1
#         I_2[i][i + 1] = 1

#     # Transformando as matrizes criadas em matrizes esparsas
#     I = sparse.csr_matrix(I)
#     I_1 = sparse.csr_matrix(I_1)
#     I_2 = sparse.csr_matrix(I_2)
#     T = sparse.csr_matrix(T)
#     D_h = sparse.csr_matrix(D_h)

#     # Matriz A de resolução do problema linear (A linha)
#     A = sparse.kron((I - I_1), D_h) + sparse.kron(I_1, T) + \
#         sparse.kron(I_2, I_h)
#     A = (1/h**2)*A

#     # Matriz auxiliar para a resolução do sistema linear
#     aux = np.identity(tam_malha)*10
#     aux[0][0] = (1/2)*(1 + 10*h)
#     aux[-1][-1] = (1/2)*(1 + 10*h)
#     aux = sparse.csr_matrix(aux)

#     # Matriz B utilizada para isolar as incógnitas e resolver o sistema linear
#     B = sparse.kron(I_1, aux)

#     # Matriz A final para a resolução do sistema linear
#     A = A - B

#     F = np.zeros((tam_malha**2, 1))
#     F[0: tam_malha] = 1

#     F = sparse.csr_matrix(F)

#     return (sparse.linalg.spsolve(A, F))


# Função que plota os valores de referência
def plot_referencia(valores_referencia, tam_malha):
    x = range(1, tam_malha)
    y = range(1, tam_malha)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    ax.plot_surface((2*X - tam_malha)/tam_malha, (2*Y - tam_malha)/tam_malha,
                    valores_referencia[X + tam_malha*(Y - 1)], cmap='inferno')
    ax.view_init(30, 45)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('U')

    # plt.show()
    plt.savefig('resultados/ref.png')


# Função que calcula as malhas
def malhas_calc(a, b, h):
    malhas = []

    for i in h:
        malhas.append(tamanho_malha(a, b, i))

    return malhas


# Função que calcula os valores_h
def valores_h_calc(h):
    valores_h = []

    for i in h:
        valores_h.append(aprox_u(i))

    return valores_h


# Função que calcula o vetor de truncamento
def truncamento_vet(a, b, h, h_ref, valores_referencia, valores_h):
    truncamento = []
    for t in range(0, len(h)):
        # Indice geral da malha refinada (Solução de referência)
        k_barra = []

        # Índice geral da malha analisada
        k = []

        # Razão entre a divisão da malha analisada e da malha da solução de referência
        q = h[t]/h_ref

        for j in range(0, tamanho_malha(a, b, h[t]) + 2):
            for i in range(0, tamanho_malha(a, b, h[t]) + 2):
                k_barra.append(i*q + (tamanho_malha(a, b, h_ref) + 2)*j*q)
                k.append(i + (tamanho_malha(a, b, h[t]) + 2)*j*q)

        k_barra = np.array(k_barra)
        k_barra = k_barra.astype(int)

        # Variável auxiliar para retirar os zeros(Valores não calculados e sim dados pela condição de contorno de Dirichlet)
        aux = abs(valores_referencia[k_barra] - valores_h[t])
        truncamento.append(aux[aux != 0])

    return truncamento


# Função que calcula o erro
def erro_calc(truncamento):
    erro = []
    for i in truncamento:
        erro.append(i.max())

    return erro


# Função que cria a tabela com os resultados
def tabela_resultados(h, erro):
    print(pd.DataFrame({'h': h, 'Erro': erro}).to_latex(index=False))


# Função que plota o gráfico de convergência
def plot_convergencia(h, erro):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.subplots()
    ax.plot(h, erro,  marker='o')
    ax.loglog()
    ax.grid(color='green', linestyle='--', linewidth=0.5,)
    ax.set_xlabel('h')
    ax.set_ylabel('erro')

    for i, j in zip(h, erro):
        ax.annotate(str(j), xy=(i, j))

    plt.savefig('resultados/convergencia.png')
