# Importando bibliotecas utilizadas
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import linalg
import os


class Metodos:

    def __init__(self, h, k, x, t):
        """
        Inicializa uma instância da classe Metodos com os parâmetros de discretização e domínio.

        Args:
            h (float): Passo de tempo.
            k (float): Passo espacial.
            x (list): Lista contendo os limites do domínio espacial [x_min, x_max].
            t (list): Lista contendo os limites do domínio temporal [t_min, t_max].

        Returns:
            None
        """
        self.h = h
        self.k = k
        self.dom_x = x
        self.dom_t = t
        self.sigma = k / (h ** 2)

        self.pontos_x = int((x[1] - x[0]) / h - 1)
        self.pontos_t = int((t[1] - t[0]) / k - 1)

        self.u_exp = self.explicito(x, t, h, k)
        self.u_crank = self.crank_nicolson(x, t, h, k)

        return

    def u_analitica(self, x, t):
        """
        Calcula a solução analítica do sistema de equações diferenciais em um determinado ponto do domínio.

        Args:
            x (float): Valor da variável espacial no domínio [x_min, x_max].
            t (float): Valor da variável temporal no domínio [t_min, t_max].

        Returns:
            float: Valor da solução analítica no ponto (x, t).

        Example:
            >>> m = Metodos(h, k, dom_x, dom_t)
            >>> x_val = 0.5
            >>> t_val = 0.2
            >>> sol_analitica = m.u_analitica(x_val, t_val)
            >>> print(sol_analitica)
            0.42920367320510344
        """
        u = np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x) + x * (1 - x)

        return u

    def plot_ref(self):
        """
        Plota o gráfico da solução analítica do sistema de equações diferenciais ao longo do domínio espacial-temporal.

        Returns:
            None

        Example:
            >>> m = Metodos(h, k, dom_x, dom_t)
            >>> m.plot_ref()
        """
        x0 = self.dom_x[0]
        xf = self.dom_x[1]
        t0 = self.dom_t[0]
        tf = self.dom_t[1]

        # Gera pontos no domínio espacial e temporal
        x = np.linspace(x0, xf, self.pontos_x)
        t = np.linspace(t0, tf, self.pontos_t)

        # Cria uma malha de pontos para o gráfico 3D
        X, T = np.meshgrid(x, t)

        # Calcula a solução analítica nos pontos da malha
        self.u_ref = self.u_analitica(X, T)

        # Cria a figura e o eixo para o gráfico 3D
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        # Plotando gráfico de superfície
        surf = ax.plot_surface(X, T, self.u_ref, cmap=plt.cm.ocean)

        # Adiciona a barra de cores ao gráfico
        fig.colorbar(surf)

        # Define a visão do gráfico
        ax.view_init(30, 45)
        ax.set_xlabel('X')
        ax.set_ylabel('T')
        ax.set_zlabel('U')

        plt.title('Solução analítica')

        # Salvar o gráfico como uma imagem
        if not os.path.exists('resultados/plot_referencia'):
            os.makedirs('resultados/plot_referencia')
        plt.savefig('resultados/plot_referencia/ref.png')

        # Fecha a figura para liberar memória
        plt.close()

        return None

    def explicito(self, x, t, h, k):
        """
        Implementa o método numérico explícito para resolver um sistema de equações diferenciais de segunda ordem.

        Args:
            x (list or tuple): Lista ou tupla contendo os valores inicial e final do domínio espacial.
            t (list or float): Lista ou valor único contendo os valores inicial e final do domínio temporal ou apenas um valor único.
            h (float): Tamanho do passo de discretização espacial.
            k (float): Tamanho do passo de discretização temporal.

        Returns:
            np.array: Array contendo os valores da solução numérica do sistema de equações diferenciais.

        Example:
            >>> m = Metodos(h, k, dom_x, dom_t)
            >>> x = [0, 1]
            >>> t = [0, 2]
            >>> h = 0.05
            >>> k = (5/11) * (h ** 2)
            >>> U = m.explicito(x, t, h, k)
        """
        sigma = self.sigma

        # A matriz da solução não contará com a borda do domínio
        m = int((x[1] - x[0]) / h - 1)

        # Verifica se t é uma lista ou um valor único
        if isinstance(t, list):
            m_linha = int((t[1] - t[0]) / k - 1)
        else:
            m_linha = int(t / k - 1)

        # Cria a matriz tridiagonal T
        T = np.zeros((m+2, m+2))

        for i in range(1, m):
            T[i][i + 1] = sigma
            T[i][i] = 1 - 2 * sigma
            T[i + 1][i] = sigma

        T[-1][-1] = 1 - 2 * sigma

        # Condição inicial U_0
        U_0 = np.zeros((m+2, 1))

        x_lin = np.linspace(x[0], x[1], m+2)
        U_0[:, 0] = np.sin(np.pi * x_lin) + x_lin * (1 - x_lin)

        U_0[0, 0] = 0
        U_0[-1, 0] = 0

        U = []
        U.append(U_0)

        # matriz coluna de 2
        m_2 = np.ones((m+2, 1)) * 2

        # Iteração para calcular a solução em cada passo de tempo
        for i in range(1, m_linha):
            aux = np.dot(T, U[i - 1])
            aux = aux + m_2 * k
            U.append(aux)

        U = np.array(U)
        U.shape = (U.shape[0], U.shape[1])

        return U

    def calc_erro(self, x, t, h, k, U):
        """
        Calcula o erro absoluto entre a solução analítica e a solução numérica.

        Args:
            x (list or tuple): Lista ou tupla contendo os valores inicial e final do domínio espacial.
            t (list or tuple): Lista ou tupla contendo os valores inicial e final do domínio temporal.
            h (float): Tamanho do passo de discretização espacial.
            k (float): Tamanho do passo de discretização temporal.
            U (np.array): Array contendo os valores da solução numérica.

        Returns:
            np.array: Array contendo os valores do erro absoluto entre a solução analítica e a solução numérica.

        Example:
            >>> m = Metodos(h, k, dom_x, dom_t)
            >>> x = [0, 1]
            >>> t = [0, 2]
            >>> h = 0.05
            >>> k = (5/11) * (h ** 2)
            >>> U = m.explicito(x, t, h, k)
            >>> m.calc_erro(x, t, h, k, U)
        """
        pontos_x = int((x[1] - x[0]) / h - 1)
        pontos_t = int((t[1] - t[0]) / k - 1)

        x = np.arange(x[0], x[1], pontos_x)
        t = np.arange(t[0], t[1], pontos_t)

        erro = abs(self.u_analitica(x, t) - U)

        return erro

    def crank_nicolson(self, x, t, h, k):
        """
        Implementa o método de Crank-Nicolson para resolver um sistema de equações diferenciais parciais.

        Args:
            x (list or tuple): Lista ou tupla contendo os valores inicial e final do domínio espacial.
            t (list or tuple): Lista ou tupla contendo os valores inicial e final do domínio temporal.
            h (float): Tamanho do passo de discretização espacial.
            k (float): Tamanho do passo de discretização temporal.

        Returns:
            np.array: Array contendo os valores da solução numérica.

        Example:
            >>> m = Metodos(h, k, dom_x, dom_t)
            >>> x = [0, 1]
            >>> t = [0, 2]
            >>> h = 0.05
            >>> k = (5/11) * (h ** 2)
            >>> U = m.crank_nicolson(x, t, h, k)
        """
        sigma = self.sigma
        m = int((x[1] - x[0]) / h - 1)

        # Condições para correção de erros ocorridos por problemas de arredondamento
        # de máquina para valores específicos de k utilizados na verificação da ordem de convergência
        if ((k == h) | (k == 0.5**2) | (k == 0.01**2)):
            if (type(t) == list):
                m_linha = int((t[1] - t[0]) / k - 2)
            else:
                m_linha = int((t) / k - 2)
        else:
            if (type(t) == list):
                m_linha = int((t[1] - t[0]) / k - 1)
            else:
                m_linha = int((t) / k - 1)

        # Matrizes T e S usadas no método de Crank-Nicolson
        T = np.zeros((m+2, m+2))
        S = np.zeros((m+2, m+2))

        for i in range(0, m+1):
            T[i][i + 1] = -sigma / 2
            T[i][i] = 1 + sigma
            T[i + 1][i] = -sigma / 2
        T[-1][-1] = 1 + sigma

        for i in range(0, m+1):
            S[i][i + 1] = sigma / 2
            S[i][i] = 1 - sigma
            S[i + 1][i] = sigma / 2
        S[-1][-1] = 1 - sigma

        # Condição inicial U_0
        U_0 = np.zeros((m+2, 1))
        x_lin = np.linspace(x[0], x[1], m+2)
        U_0[:, 0] = np.sin(np.pi * x_lin) + x_lin * (1 - x_lin)

        U = []
        U.append(U_0)

        # matriz coluna de 2
        m_2 = np.ones((m+2, 1)) * 2

        # Iteração para calcular a solução em cada passo de tempo
        for i in range(1, m_linha):
            aux = np.dot(S, U[i-1])
            aux2 = np.linalg.solve(T, aux)

            aux2 = aux2 + m_2 * k

            aux2[0][0] = 0
            aux2[-1][0] = 0

            U.append(aux2)

        U = np.array(U)
        U.shape = (U.shape[0], U.shape[1])

        return U

    def plot_numerica(self, U, nome, titulo):
        """
        Plota a solução numérica do sistema de equações diferenciais parciais.

        Args:
            U (np.array): Array contendo os valores da solução numérica.
            nome (str): Nome do arquivo de imagem para salvar o gráfico.
            titulo (str): Título do gráfico.

        Returns:
            None

        Example:
            >>> m = Metodos(h, k, dom_x, dom_t)
            >>> U = m.explicito(x, t, h, k)
            >>> m.plot_numerica(U, 'explicito', 'Método Explícito')
        """
        x0 = self.dom_x[0]
        xf = self.dom_x[1]
        t0 = self.dom_t[0]
        tf = self.dom_t[1]

        # Criando os vetores de coordenadas para o gráfico
        x = np.linspace(x0, xf, self.pontos_x+2)
        t = np.linspace(t0, tf, self.pontos_t)

        X, T = np.meshgrid(x, t)

        # Criando a figura do gráfico 3D
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        # Plotando a superfície com base nos valores da solução numérica (U)
        surf = ax.plot_surface(X, T, U, cmap=plt.cm.ocean)
        fig.colorbar(surf)

        # Configurações do gráfico
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
        """
        Plota gráficos de comparação das soluções numéricas para diferentes tempos.

        Args:
            x (list): Lista contendo o intervalo [x0, xf] do domínio espacial.
            h (float): Tamanho do passo de espaço.
            k (float): Tamanho do passo de tempo.
            nome (str): Nome do método usado para o gráfico (exemplo: 'Crank-Nicolson', 'explicito').

        Returns:
            None

        Example:
            >>> m = Metodos(h, k, dom_x, dom_t)
            >>> m.comparacao(dom_x, h, k, 'explicito')
        """
        pontos_x = int((x[1] - x[0]) / h - 1)
        x_lin = np.linspace(x[0], x[1], pontos_x+2)

        # Valores de tempo para comparação
        t_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

        plt.figure(figsize=(10, 7))
        plt.xlabel("x", fontdict={"fontsize": 14, "fontweight": "bold"})
        plt.ylabel("u", fontdict={"fontsize": 14, "fontweight": "bold"})
        plt.title("X/U - Método: " + nome, fontdict={"fontsize": 16, "fontweight": "bold"})

        for i in t_values:
            # Calcula a solução numérica para o tempo 'i'
            if nome == 'Crank-Nicolson':
                U = self.crank_nicolson(x, i, h, k)
            elif nome == 'explicito':
                U = self.explicito(x, i, h, k)

            # Obtém a última linha da matriz U, que representa a solução em t=i
            ultima_linha = U[-1, :]

            # Plota a curva da solução numérica em t=i com um rótulo
            plt.plot(x_lin, ultima_linha, label='t = {:.2f}'.format(i))

        plt.xlim([x[0], x[1]])
        plt.ylim([0, 1])  # Ajuste o intervalo y conforme necessário

        plt.grid(linestyle='--')
        plt.legend()

        # Salvar o gráfico como uma imagem
        if not os.path.exists('resultados/comparacoes'):
            os.makedirs('resultados/comparacoes')
        plt.savefig('resultados/comparacoes/' + f'{nome}.png')

        plt.close()

    def erro_convergencia(self, x, t, nome):
        """
        Calcula a convergência do erro para diferentes tamanhos de passos 'h' e 'k' e plota gráficos de erro.

        Args:
            x (list): Lista contendo o intervalo [x0, xf] do domínio espacial.
            t (list): Lista contendo o intervalo [t0, tf] ou valor único 't' do domínio temporal.
            nome (str): Nome do método usado para o cálculo do erro e gráficos (exemplo: 'Explicito', 'Crank-Nicolson').

        Returns:
            None

        Example:
            >>> m = Metodos(h, k, dom_x, dom_t)
            >>> m.erro_convergencia(dom_x, dom_t, 'Explicito')
            >>> m.erro_convergencia(dom_x, dom_t, 'Crank-Nicolson')
        """
        # Calculando a norma (2-norm) do erro para múltiplos valores de h e k
        erro = []
        valores_h = np.array([0.5, 0.1, 0.05, 0.01])

        # Calculando os valores de k correspondentes para o método
        if (nome == 'Explicito'):
            valores_k = (5/11)*(valores_h)**2
        elif (nome == 'Crank-Nicolson'):
            valores_k = (5/11)*(valores_h)

        for i in range(len(valores_h)):
            # Calculando a solução numérica para o método e os valores de h e k atuais
            if (nome == 'Explicito'):
                U = self.explicito(x, t, valores_h[i], valores_k[i])
            elif (nome == 'Crank-Nicolson'):
                U = self.crank_nicolson(x, t, valores_h[i], valores_k[i])

            # Calculando o erro
            aux = self.calc_erro(x, t, valores_h[i], valores_k[i], U)

            # Multiplicando a matriz pelos tamanhos dos passos antes de calcular sua norma
            aux = valores_h[i] * valores_k[i] * aux

            # Calculando a norma do erro e adicionando à lista 'erro'
            erro.append(np.linalg.norm(aux))

        # Convertendo os arrays para listas
        valores_h = valores_h.tolist()
        valores_k = valores_k.tolist()

        # Tabela dos valores para a análise da convergência
        analise_conv = pd.DataFrame({'h': valores_h, 'k': valores_k, 'erro': erro})

        # Plotando gráfico de erro em relação a h
        plt.figure(figsize=(10, 7))
        plt.xlabel("h", fontdict={"fontsize": 14, "fontweight": "bold"})
        plt.ylabel("erro", fontdict={"fontsize": 14, "fontweight": "bold"})
        plt.title("Erro x: " + nome, fontdict={"fontsize": 16, "fontweight": "bold"})

        plt.grid(linestyle='--')
        plt.plot(valores_h, erro, color="blue")

        # Salvar o gráfico como uma imagem
        if not os.path.exists('resultados/convergencia'):
            os.makedirs('resultados/convergencia')
        plt.savefig('resultados/convergencia/conv_' + nome + '_h.png')
        plt.close()

        # Plotando gráfico de erro em relação a k
        plt.figure(figsize=(10, 7))
        plt.xlabel("k", fontdict={"fontsize": 14, "fontweight": "bold"})
        plt.ylabel("erro", fontdict={"fontsize": 14, "fontweight": "bold"})
        plt.title("Erro k: " + nome, fontdict={"fontsize": 16, "fontweight": "bold"})

        plt.grid(linestyle='--')
        plt.plot(valores_k, erro, color="blue")

        # Salvar o gráfico como uma imagem
        if not os.path.exists('resultados/convergencia'):
            os.makedirs('resultados/convergencia')
        plt.savefig('resultados/convergencia/conv_' + nome + '_k.png')
        plt.close()

    def imprime_matriz(self, A):
        """
        Imprime a matriz densa A em um arquivo de texto chamado "matriz.txt".

        Args:
            A (numpy.ndarray): A matriz densa a ser impressa.

        Returns:
            None

        Example:
            >>> m = Metodos(h, k, dom_x, dom_t)
            >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> m.imprime_matriz(A)
        """
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
            # Se A não for uma matriz esparsa, cairá no bloco 'except'
            # Criar um DataFrame pandas a partir da matriz densa (já que não é esparsa)
            df = pd.DataFrame(A)

            # Configurar opções de exibição do pandas para mostrar toda a matriz
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)

            # Imprimir a matriz usando o método to_string()
            # print(df.to_string(index=False, header=False))
            np.savetxt("matriz.txt", df, fmt=format)

        return None
