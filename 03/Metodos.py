# Importando bibliotecas utilizadas
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import linalg
import os


class Metodos:

    def __init__(self, h, k, lim_t, lim_sum):

        self.h = h
        self.k = k
        self.lim_t = lim_t
        self.lim_sum = lim_sum

        self.U_exp = self.U_explicito()
        self.U_cn = self.U_cn()

        return

    # Solução analítica do problema
    def u_analitica(self, x, t):
        lim_sum = self.lim_sum
        aux = 0
        for n in range(1, lim_sum + 1):
            aux += np.sin(n*np.pi/2)*np.sin(n*np.pi*x)*np.exp(-(n**2)*(np.pi**2)*t)/(n**2)

        return (8*aux/(np.pi**2))

    def plot_ref(self):
        # Plotando gráfico da função para avaliar comportamento
        x = np.linspace(0, 1, 100)
        t = np.linspace(0, 2, 100)

        X, T = np.meshgrid(x, t)
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.view_init(30, 45)

        self.u_ref = self.u_analitica(X, T)

        ax.plot_surface(X, T, self.u_ref, cmap='cividis')

        # Salvar o gráfico como uma imagem
        if not os.path.exists('ref'):
            os.makedirs('ref')
        plt.savefig('ref/ref.png')

        plt.close()

        return None

    def U_explicito(self):
        h = self.h
        k = self.k
        lim_t = self.lim_t
        sigma = k/(h**2)

        # A matriz da solução não contará com a borda do domínio
        m = int(1/h - 1)
        m_linha = int(lim_t/k - 1)

        T = np.zeros((m, m))
        for i in range(0, m - 1):
            T[i][i + 1] = sigma
            T[i][i] = 1 - 2*sigma
            T[i + 1][i] = sigma
        T[-1][-1] = 1 - 2*sigma

        U_0 = np.zeros((m, 1))
        for j in range(0, m):
            if (j <= 1/(2*h)):
                U_0[j][0] = 2*j*h
            elif (j <= 1/h):
                U_0[j][0] = 2 - 2*j*h

        U = []
        U.append(U_0)

        for i in range(1, m_linha + 1):
            aux = np.dot(T, U[i-1])
            U.append(aux)

        U = np.array(U)
        U.shape = (U.shape[0], U.shape[1])

        return U

    def erro_explicito(self):
        h = self.h
        k = self.k
        lim_t = self.lim_t

        x = np.arange(h, 1, h)
        t = np.arange(k, lim_t, k)
        X, T = np.meshgrid(x, t)

        erro = abs(self.u_analitica(X, T) - self.U_exp)

        return erro

    def U_cn(self):
        h = self.h
        k = self.k
        lim_t = self.lim_t

        sigma = k/(h**2)

        # A matriz da solução não contará com a borda do domínio
        m = int(1/h - 1)

        # Condições para correção de erros ocorridos por problemas de arredondamento
        # de máquina para valores específicos de k utilizados na verificação da ordem de convergência
        if ((k == h) | (k == 0.5**2) | (k == 0.01**2)):
            m_linha = int(lim_t/k - 2)
        else:
            m_linha = int(lim_t/k - 1)

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
        for j in range(0, m):
            if (j <= 1/(2*h)):
                U_0[j][0] = 2*j*h
            elif (j <= 1/h):
                U_0[j][0] = 2 - 2*j*h

        U = []
        U.append(U_0)

        for i in range(1, m_linha + 1):
            aux = np.dot(S, U[i - 1])

            aux2 = np.linalg.solve(T, aux)
            U.append(aux2)

        U = np.array(U)
        U.shape = (U.shape[0], U.shape[1])

        return U

    def erro_cn(self):
        h = self.h
        lim_t = self.lim_t
        k = self.k

        x = np.arange(h, 1, h)
        t = np.arange(k, lim_t, k)
        X, T = np.meshgrid(x, t)
        erro = abs(self.u_analitica(X, T) - self.U_cn)

        return erro

    def plot_numerica(self, U, nome, titulo):
        h = self.h
        k = self.k
        lim_t = self.lim_t

        # Plotagem 3D da solução numérica
        m = int(1/h - 1)
        m_linha = int(lim_t/k)

        x = np.arange(0, m)
        t = np.arange(0, m_linha)

        X, T = np.meshgrid(x, t)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        ax.plot_surface((-X + m)/m, (T*lim_t)/m_linha, U.flatten()[T*m + X], cmap='cividis')
        ax.view_init(30, 45)
        ax.set_xlabel('X')
        ax.set_ylabel('T')
        ax.set_zlabel('U')

        plt.title(titulo)

        # Salvar o gráfico como uma imagem
        if not os.path.exists('sol_num'):
            os.makedirs('sol_num')
        plt.savefig('sol_num/' + nome + '.png')

        plt.close()

    def comparacao(self, U, nome):
        h = self.h
        k = self.k
        lim_t = self.lim_t
        # Comparação da solução analítica com a solução numérica
        m = int(1/h - 1)
        m_linha = int(lim_t/k)
        x = np.arange(0, m)

        for i in [0, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5]:
            u = self.u_analitica(x*h, i).flatten()

            fig, ax = plt.subplots()
            ax.plot((-x + m)/m, U.flatten()[int(i/k)*m + x])
            ax.plot((-x + m)/m, u[x])
            plt.title("t = %.2f" % i)
            ax.set_ylim([0, 1.05])
            plt.legend(['Solução numérica', 'Solução analítica'])

            # Salvar o gráfico como uma imagem
            if not os.path.exists('comparacoes_'+nome):
                os.makedirs('comparacoes_'+nome)
            plt.savefig('comparacoes_'+nome + '/t_' + str(i) + '.png')

            plt.close()

    def erro_convergencia(self, nome):
        # Calculando a norma (2-norm) do erro para múltiplos valores de h e k
        erro = []
        valores_h = np.array([0.5, 0.1, 0.05, 0.01])
        k = self.k
        valores_k = (5/11)*(valores_h)**2
        for h in [0.5, 0.1, 0.05, 0.01]:
            if (nome == 'Explicito'):
                aux = self.erro_explicito()
            elif (nome == 'Crank-Nicolson'):
                aux = self.erro_cn()

            # Multiplicando a matriz pelos tamanhos dos passos  antes de calcular sua norma
            aux = h*k*aux
            erro.append(np.linalg.norm(aux))

        valores_h = valores_h.tolist()
        valores_k = valores_k.tolist()

        # Tabela dos valores para a análise da convergência
        analise_conv = pd.DataFrame({'h': valores_h, 'k': valores_k, 'erro': erro})

        # Plotando figuras de convergência
        analise_conv.plot.line(x='h', y='erro', loglog=True, legend=False, figsize=(8, 8), color='red')
        plt.ylabel('erro')

        # Salvar o gráfico como uma imagem
        if not os.path.exists('convergencia'):
            os.makedirs('convergencia')
        plt.savefig('convergencia/conv_' + nome + '_h.png')

        analise_conv.plot.line(x='k', y='erro', loglog=True, legend=False, figsize=(8, 8), color='red')
        plt.ylabel('erro')

        if not os.path.exists('convergencia'):
            os.makedirs('convergencia')
        plt.savefig('convergencia/conv_' + nome + '_k.png')
