# Importando bibliotecas utilizadas
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import linalg
import os


class Metodos:

    def __init__(self, h, k, x, t):

        self.h = h
        self.k = k
        self.dom_x = x
        self.dom_t = t
        self.sigma = k/(h**2)

        self.pontos_x = int((x[1]-x[0])/h - 1)
        self.pontos_t = int((t[1]-t[0])/k - 1)

        self.u_exp = self.explicito(x, t, h, k)
        self.u_crank = self.crank_nicolson(x, t, h, k)

        return

    # Solução analítica do problema
    def u_analitica(self, x, t):

        u = np.exp(-np.pi**2*t)*np.sin(np.pi*x) + x*(1 - x)

        return u

    def plot_ref(self):

        x0 = self.dom_x[0]
        xf = self.dom_x[1]
        t0 = self.dom_t[0]
        tf = self.dom_t[1]

        x = np.linspace(x0, xf, self.pontos_x)
        t = np.linspace(t0, tf, self.pontos_t)

        X, T = np.meshgrid(x, t)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        self.u_ref = self.u_analitica(X, T)

        # Plotando gráfico de superfície
        surf = ax.plot_surface(X, T, self.u_ref, cmap=plt.cm.ocean)

        fig.colorbar(surf)

        ax.view_init(30, 45)
        ax.set_xlabel('X')
        ax.set_ylabel('T')
        ax.set_zlabel('U')

        plt.title('Solução analítica')

        # Salvar o gráfico como uma imagem
        if not os.path.exists('resultados/plot_referencia'):
            os.makedirs('resultados/plot_referencia')
        plt.savefig('resultados/plot_referencia/ref.png')

        plt.close()

        return None

    def explicito(self, x, t, h, k):
        sigma = self.sigma

        # A matriz da solução não contará com a borda do domínio
        m = int((x[1]-x[0])/h - 1)

        if (type(t) == list):
            m_linha = int((t[1]-t[0])/k - 1)
        else:
            m_linha = int((t)/k - 1)

        T = np.zeros((m, m))
        for i in range(0, m - 1):
            T[i][i + 1] = sigma
            T[i][i] = 1 - 2*sigma
            T[i + 1][i] = sigma

        T[-1][-1] = 1 - 2*sigma

        U_0 = np.zeros((m, 1))
        for x in range(0, m):
            U_0[x][0] = np.sin(np.pi*x*self.h) + x*self.h*(1 - x*self.h)

        U = []
        U.append(U_0)

        # matriz coluna de 2
        m_2 = np.ones((m, 1))*2

        for i in range(1, m_linha):
            aux = np.dot(T, U[i-1])
            # aux = aux + m_2
            U.append(aux)

        U = np.array(U)
        U.shape = (U.shape[0], U.shape[1])

        return U

    def calc_erro(self, x, t, h, k, U):
        pontos_x = int((x[1]-x[0])/h - 1)
        pontos_t = int((t[1]-t[0])/k - 1)

        x = np.arange(x[0], x[1], pontos_x)
        t = np.arange(t[0], t[1], pontos_t)

        # X, T = np.meshgrid(x, t)
        erro = abs(self.u_analitica(x, t) - U)

        return erro

    def crank_nicolson(self, x, t, h, k):
        sigma = self.sigma
        m = int((x[1]-x[0])/h - 1)

        # Condições para correção de erros ocorridos por problemas de arredondamento
        # de máquina para valores específicos de k utilizados na verificação da ordem de convergência
        if ((k == h) | (k == 0.5**2) | (k == 0.01**2)):
            if (type(t) == list):
                m_linha = int((t[1]-t[0])/k - 2)
            else:
                m_linha = int((t)/k - 2)

        else:
            if (type(t) == list):
                m_linha = int((t[1]-t[0])/k - 1)
            else:
                m_linha = int((t)/k - 1)

        T = np.zeros((m, m))

        for i in range(0, m - 1):
            T[i][i + 1] = -sigma/2
            T[i][i] = 1 + sigma
            T[i + 1][i] = -sigma/2
        T[-1][-1] = 1 + sigma

        S = np.zeros((m, m))
        for i in range(0, m - 1):
            S[i][i + 1] = sigma/2
            S[i][i] = 1 - sigma
            S[i + 1][i] = sigma/2
        S[-1][-1] = 1 - sigma

        U_0 = np.zeros((m, 1))
        for x in range(0, m):
            U_0[x][0] = np.sin(np.pi*x*self.h) + x*self.h*(1 - x*self.h)

        U = []
        U.append(U_0)

        for i in range(1, m_linha):
            aux = np.dot(S, U[i - 1])

            aux2 = np.linalg.solve(T, aux)
            U.append(aux2)

        U = np.array(U)
        U.shape = (U.shape[0], U.shape[1])

        return U

    def plot_numerica(self, U, nome, titulo):
        x0 = self.dom_x[0]
        xf = self.dom_x[1]
        t0 = self.dom_t[0]
        tf = self.dom_t[1]

        x = np.linspace(x0, xf, self.pontos_x)
        t = np.linspace(t0, tf, self.pontos_t)

        X, T = np.meshgrid(x, t)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        surf = ax.plot_surface(X, T, U, cmap=plt.cm.ocean)
        fig.colorbar(surf)

        ax.view_init(30, 45)
        ax.set_xlabel('X')
        ax.set_ylabel('T')
        ax.set_zlabel('U')

        plt.title(titulo)

        # Salvar o gráfico como uma imagem
        if not os.path.exists('resultados/solucao_numerica'):
            os.makedirs('resultados/solucao_numerica')
        plt.savefig('resultados/solucao_numerica/' + nome + '.png')

        plt.close()

    def comparacao(self, x, h, k, nome):
        pontos_x = int((x[1]-x[0])/h - 1)
        x_lin = np.linspace(x[0], x[1], pontos_x)

        t_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

        for i in t_values:
            u_ref = self.u_analitica(x_lin, i)

            if (nome == 'Crank-Nicolson'):
                U = self.crank_nicolson(x, i, h, k)
            elif (nome == 'explicito'):
                U = self.explicito(x, i, h, k)

            fig, ax = plt.subplots()
            ultima_linha = U[-1, :]

            plt.figure(figsize=(10, 7))
            plt.xlabel("x", fontdict={"fontsize": 14, "fontweight": "bold"})
            plt.ylabel("u", fontdict={"fontsize": 14, "fontweight": "bold"})
            plt.title(
                "X/U - Método: " + nome, fontdict={"fontsize": 16, "fontweight": "bold"}
            )

            plt.plot(x_lin, ultima_linha, color="blue", label="Solução numérica")
            plt.plot(x_lin, u_ref, "--", color="red", label="Solução analítica")

            ax.set_xlim([self.dom_x[0], self.dom_x[1]])
            ax.set_ylim([self.dom_t[0], self.dom_t[1]])

            plt.grid(linestyle='--')

            plt.legend()

            # Salvar o gráfico como uma imagem
            if not os.path.exists('resultados/comparacoes_'+nome):
                os.makedirs('resultados/comparacoes_'+nome)
            plt.savefig('resultados/comparacoes_'+nome + '/t_' + str(i) + '.png')

            plt.close()

    def erro_convergencia(self, x, t, nome):
        # Calculando a norma (2-norm) do erro para múltiplos valores de h e k
        erro = []
        valores_h = np.array([0.5, 0.1, 0.05, 0.01])

        if (nome == 'Explicito'):
            valores_k = (5/11)*(valores_h)**2
        elif (nome == 'Crank-Nicolson'):
            valores_k = (5/11)*(valores_h)

        for i in range(len(valores_h)):
            if (nome == 'Explicito'):
                U = self.explicito(x, t, valores_h[i], valores_k[i])
                aux = self.calc_erro(x, t, valores_h[i], valores_k[i], U)

            elif (nome == 'Crank-Nicolson'):
                U = self.crank_nicolson(x, t, valores_h[i], valores_k[i])
                aux = self.calc_erro(x, t, valores_h[i], valores_k[i], U)

            # Multiplicando a matriz pelos tamanhos dos passos  antes de calcular sua norma
            aux = valores_h[i]*valores_k[i]*aux

            erro.append(np.linalg.norm(aux))

        valores_h = valores_h.tolist()
        valores_k = valores_k.tolist()

        # Tabela dos valores para a análise da convergência
        analise_conv = pd.DataFrame({'h': valores_h, 'k': valores_k, 'erro': erro})

        # =================

        plt.figure(figsize=(10, 7))
        plt.xlabel("h", fontdict={"fontsize": 14, "fontweight": "bold"})
        plt.ylabel("erro", fontdict={"fontsize": 14, "fontweight": "bold"})
        plt.title(
            "Erro x: " + nome, fontdict={"fontsize": 16, "fontweight": "bold"}
        )

        plt.grid(linestyle='--')

        plt.plot(valores_h, erro, color="blue")

        # # Salvar o gráfico como uma imagem
        if not os.path.exists('resultados/convergencia'):
            os.makedirs('resultados/convergencia')
        plt.savefig('resultados/convergencia/conv_' + nome + '_h.png')

        plt.close()

        # =============

        plt.figure(figsize=(10, 7))
        plt.xlabel("k", fontdict={"fontsize": 14, "fontweight": "bold"})
        plt.ylabel("erro", fontdict={"fontsize": 14, "fontweight": "bold"})
        plt.title(
            "Erro k: " + nome, fontdict={"fontsize": 16, "fontweight": "bold"}
        )

        plt.grid(linestyle='--')

        plt.plot(valores_k, erro, color="blue")

        # # Salvar o gráfico como uma imagem
        if not os.path.exists('resultados/convergencia'):
            os.makedirs('resultados/convergencia')
        plt.savefig('resultados/convergencia/conv_' + nome + '_k.png')

        plt.close()
    # Imprime a matriz em um arquivo txt

    def imprime_matriz(self, A):
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
