import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.interpolate import interp2d


# Imprime a matriz em um arquivo txt
def imprime_matriz(A):
    format = "%.0f"

    try:
        # Criar um DataFrame pandas a partir da matriz densa
        df = pd.DataFrame(A.toarray())

        # Configurar opções de exibição do pandas para mostrar toda a matriz
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        # Imprimir a matriz usando o método to_string()
        # print(df.to_string(index=False, header=False))
        np.savetxt("matriz.txt", df, fmt=format)
    except:
        # Criar um DataFrame pandas a partir da matriz densa
        df = pd.DataFrame(A)

        # Configurar opções de exibição do pandas para mostrar toda a matriz
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        # Imprimir a matriz usando o método to_string()
        # print(df.to_string(index=False, header=False))
        np.savetxt("matriz.txt", df, fmt=format)

    return


# Calcula o tamanho da malha
def tamanho_malha(a, b, h):
    n = (b - a) / h + 1

    return int(n)


# Calcula a aproximação da solução da EDP
def aprox_sol(a, h, tam_malha, k_2):
    # Matriz identidade
    I = sparse.identity(tam_malha, format="lil")

    # Matriz auxiliar
    I_1 = sparse.identity(tam_malha, format="lil")
    I_1[0, 0] = 0
    I_1[-1, -1] = 0

    # Matriz auxiliar
    I_2 = sparse.lil_matrix((tam_malha, tam_malha))

    I_h = I * (1 / h / h)

    # Condicoes de Neumann e diferenças finitas
    T = sparse.lil_matrix((tam_malha, tam_malha))
    T[0, 0] = (-(4 - h)) / (h * h) + k_2
    T[-1, -1] = (-(4 + h)) / (h * h) + k_2
    T[0, 1] = 2 / h / h
    T[-1, -2] = 2 / h / h

    # Condicoes de Dirichlet e diferenças finitas
    for i in range(1, tam_malha - 1):
        T[i, i - 1] = 1 / h / h
        T[i, i] = (-4) / (h * h) + k_2
        T[i, i + 1] = 1 / h / h

        I_2[i, i - 1] = 1
        I_2[i, i + 1] = 1

    # Matriz A
    A = sparse.kron((I - I_1), I) + sparse.kron(I_1, T) + sparse.kron(I_2, I_h)

    # Vetor B
    B = np.zeros((tam_malha**2, 1))
    for i in range(tam_malha + 1):
        B[i] = np.sin(a + i * h)

    B = sparse.csr_matrix(B)

    # Calcula a solução do sistema linear
    return sparse.linalg.spsolve(A, B)


# Plota a superfície da aproximação da solução da EDP
def plot_referencia(a, b, tam_malha, sol_ref):
    # Discretização do domínio
    x = np.linspace(a, b, tam_malha)
    y = np.linspace(a, b, tam_malha)

    z = sol_ref.reshape((tam_malha, tam_malha))

    X, Y = np.meshgrid(x, y)

    # Criando figura e eixos 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plotando gráfico de superfície
    surf = ax.plot_surface(
        X, Y, z, cmap=plt.cm.ocean, rcount=tam_malha, ccount=tam_malha
    )

    # Configurando eixos e título
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Aproximação da solução da EDP")

    # Adicionando barra de cores
    fig.colorbar(surf)

    # Mostrando o gráfico
    plt.savefig("resultados/superficie_ref.png")

    return


# Calcula para cada h a aproximação da solução da EDP
def valores_h_calc(a, b, h, k_2):
    valores_h = []

    for i in h:
        valores_h.append(aprox_sol(a, i, tamanho_malha(a, b, i), k_2))

    return valores_h


# Calcula o erro relativo a partir dos valores de referência e os valores aproximados
def erro_calc(a, b, h, sol_ref, valores_h, tam_malha):
    erros = []

    # Definição dos dados
    x = np.linspace(a, b, tam_malha)
    y = np.linspace(a, b, tam_malha)

    # Interpolação por spline cúbica
    f_ref = interp2d(x, y, sol_ref, kind="cubic")

    for i in range(0, len(h)):
        # Discretização do domínio
        x_h = np.linspace(a, b, tamanho_malha(a, b, h[i]))
        y_h = np.linspace(a, b, tamanho_malha(a, b, h[i]))

        # Interpolação por spline cúbica
        f_h = interp2d(x_h, y_h, valores_h[i], kind="cubic")

        # # Calculando o erro
        erro_absoluto = np.abs(f_ref(x_h, y_h) - f_h(x_h, y_h))

        # Calculando a norma máxima
        norma_max = np.amax(np.abs(erro_absoluto))

        # Calculando o erro relativo
        erro_relativo = norma_max / np.amax(np.abs(f_ref(x_h, y_h)))

        erros.append(erro_relativo)

    return erros


# Plota o erro relativo em função de h
def plot_erros(h, erro):
    # PLotando o gráfico
    plt.figure(figsize=(10, 7))
    plt.xlabel("H", fontdict={"fontsize": 14, "fontweight": "bold"})
    plt.ylabel("Erro", fontdict={"fontsize": 14, "fontweight": "bold"})
    plt.title(
        "Erro/H (escala logXlog)", fontdict={"fontsize": 16, "fontweight": "bold"}
    )

    plt.loglog(h, erro, color="blue", label="Tamanho do H")

    # Plotando a parábola de referência
    quadrados = np.array(h) ** 2
    plt.loglog(h, quadrados, "--", color="red", label="Parábola de referência")

    plt.legend()

    # Salvando o gráfico
    plt.savefig("resultados/erros_h.png")

    return
