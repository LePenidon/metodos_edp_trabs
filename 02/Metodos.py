import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj  # type: ignore
from scipy.special import ellipk  # type: ignore
from scipy.linalg import solve
from scipy.optimize import newton  # type: ignore
import os
import time


class Metodos:

    def __init__(self, t0, tf, u0, h):
        """
        Inicializa a classe Metodos com os parâmetros necessários para a resolução do sistema de equações diferenciais.

        Args:
            t0 (float): Valor inicial do parâmetro t.
            u0 (list or array): Lista ou array contendo os valores iniciais das variáveis u1 e u2.
            h (float): Tamanho do passo.
            num_passos (int): Número de passos.

        Returns:
            None

        Example:
            >>> obj = Metodos(0.0, [0.5, 1.0], 0.01, 100)

        """
        # Define os parâmetros iniciais
        self.t0 = t0
        self.tf = tf
        self.u0 = u0
        self.h = h
        self.num_passos = round((self.tf-self.t0)/self.h)

        # Cria uma lista de valores de tempo uniformemente espaçados
        self.t_lin = np.linspace(t0, self.tf, self.num_passos+1)

        # Calcula os valores de referência do vetor de estados
        self.u_ref = [self.sol_referencia(t, u0) for t in self.t_lin]

        # Chama os métodos de resolução para cada um dos métodos
        self.t_euler_e, self.u_euler_e = self.euler_explicito(
            t0, u0, h, self.num_passos)
        self.t_taylor_2, self.u_taylor_2 = self.taylor_2(t0, u0, h, self.num_passos)
        self.t_AB_2, self.u_AB_2 = self.adams_bashforth2(t0, u0, h, self.num_passos)
        self.t_euler_i, self.u_euler_i = self.euler_implicito(
            t0, u0, h, self.num_passos)
        self.t_PC, self.u_PC = self.preditor_corretor(t0, u0, h, self.num_passos)

        # Define um dicionário com os métodos de resolução
        self.metodos_dict = {
            "referencia": self.sol_referencia,
            "euler_explicito": self.euler_explicito,
            "taylor_2": self.taylor_2,
            "adams_bashforth2": self.adams_bashforth2,
            "euler_implicito": self.euler_implicito,
            "preditor_corretor": self.preditor_corretor
        }

        # Calcula os tempos de execução para cada método de resolução
        self.tempos_execucao()

        # Plota os gráficos para cada método de resolução
        for i in self.metodos_dict.keys():
            # Plota o gráfico de velocidade versus tempo
            self.plot_vel_pos(i)
            # # Calcula e plota os erros para cada método
            self.erros_metodos(self.metodos_dict[i], i)
            # # Plota o gráfico de fase
            self.plot_grafico_fase(self.metodos_dict[i], i)

        return

    def f(self, u, t):
        """
        Define a função que retorna as derivadas de u1 e u2 em relação a t.

        Args:
            u (list or array): Lista ou array contendo os valores das variáveis u1 e u2.
            t (float): Valor do parâmetro t.

        Returns:
            array: Array contendo as derivadas de u1 e u2 em relação a t.

        Example:
            >>> obj = MyClass()
            >>> u = [0.5, 1.0]
            >>> t = 0.0
            >>> obj.f(u, t)
            array([ 1.        , -0.47942554])
        """
        u1, u2 = u

        # Define as derivadas de u1 e u2 em relação a t
        dudt = np.array([u2, -np.sin(u1)])

        return dudt

    def f_jac(self, u, t):
        """
        Define a função que retorna a matriz jacobiana da função f em relação às variáveis u1 e u2.

        Args:
            u (list or array): Lista ou array contendo os valores das variáveis u1 e u2.
            t (float): Valor do parâmetro t.

        Returns:
            array: Array contendo a matriz jacobiana da função f em relação às variáveis u1 e u2.

        Example:
            >>> obj = MyClass()
            >>> u = [0.5, 1.0]
            >>> t = 0.0
            >>> obj.f_jac(u, t)
            array([[ 0.        ,  1.        ],
                [-0.87758256,  0.        ]])
        """
        # Extrai os valores de u1 e u2 do vetor u
        u1, u2 = u

        # Define a matriz jacobiana
        jac = np.array([[0, 1], [-np.cos(u1), 0]])

        return jac

    def sol_referencia(self, t, u):
        """
        Calcula a solução de referência do sistema de equações diferenciais.

        Esta função recebe dois argumentos: t, que representa o tempo, e u, que é uma lista contendo dois valores numéricos.
        A função realiza uma série de cálculos para determinar a solução de referência do sistema de equações diferenciais.

        Args:
            t (float): O tempo.
            u (list): Lista contendo dois valores numéricos.

        Returns:
            np.array: Array contendo os valores da solução de referência.

        Example:
            >>> obj = MyClass()
            >>> obj.u = [0.5, 1.0]
            >>> obj.t = 2.0
            >>> obj.sol_referencia()
            array([ 3.01190207, -0.15623905])
        """

        # Variável auxiliar para calcular k_0
        k_0 = np.sin(0.5 * u[0])

        # Cálculo da integral elíptica incompleta de primeira espécie
        K = ellipk(k_0 ** 2)

        # Cálculo das funções elípticas de Jacobi: sn, cn, dn e ph
        sn, cn, dn, ph = ellipj(K - t, k_0 ** 2)

        # Cálculo do deslocamento angular
        q = 2 * np.arcsin(k_0 * sn)

        # Cálculo da velocidade angular
        p = -2 * k_0 * cn

        # Vetor de solução de referência contendo os valores do deslocamento angular e da velocidade angular
        u = np.array([q, p])

        return u

    def euler_explicito(self, t0, u0, h, num_passos):
        """
        Implementa o método de Euler explícito para resolver um sistema de equações diferenciais de primeira ordem.

        Args:
            t0 (float): Condição inicial para o tempo.
            u0 (numpy.ndarray): Condição inicial para o vetor de estados.
            h (float): Tamanho do passo de integração.
            num_passos (int): Número de passos de integração.

        Returns:
            tuple: Tupla contendo duas listas, onde a primeira lista contém os valores de tempo e a segunda lista contém
                os valores do vetor de estados ao longo da integração.
        """

        t = [t0]  # Lista para armazenar os valores de t
        u = [u0]  # Lista para armazenar os valores de u

        for i in range(num_passos):
            t_i = t[-1]
            u_i = u[-1]

            # Cálculo do valor da função f(u, t) no ponto atual
            f_i = self.f(u_i, t_i)

            # Cálculo do próximo valor de u usando o método de Euler explícito
            u_prox = u_i + h * f_i

            # Atualização do valor de t
            t.append(t_i + h)

            # Adição do próximo valor de u à lista
            u.append(u_prox)

        return t, u

    def taylor_2(self, t0, u0, h, num_passos):
        """
        Implementa o método de Taylor de ordem 2 para resolver um sistema de equações diferenciais de primeira ordem.

        Args:
            t0 (float): Condição inicial para o tempo.
            u0 (numpy.ndarray): Condição inicial para o vetor de estados.
            h (float): Tamanho do passo de integração.
            num_passos (int): Número de passos de integração.

        Returns:
            tuple: Tupla contendo duas listas, onde a primeira lista contém os valores de tempo e a segunda lista contém
                os valores do vetor de estados ao longo da integração.
        """

        t = [t0]  # Lista para armazenar os valores de t
        u = [u0]  # Lista para armazenar os valores de u

        for i in range(num_passos):
            t_i = t[-1]  # Último valor de t
            u_i = u[-1]  # Último valor de u

            # Calcula o vetor f(u_i, t_i) usando a função f definida anteriormente
            f_i = self.f(u_i, t_i)

            # Calcula a matriz Jacobiana df(u_i, t_i) usando a função f_jac definida anteriormente
            f_jac_i = self.f_jac(u_i, t_i)

            # Cálculo do próximo valor de u usando o método de Taylor de ordem 2
            u_prox = u_i + h * f_i + h**2/2 * (np.dot(f_jac_i, f_i))

            # Atualização do valor de t
            t.append(t_i + h)

            # Adição do próximo valor de u à lista
            u.append(u_prox)

        return t, u

    def adams_bashforth2(self, t0, u0, h, num_passos):
        """
        Implementa o método de Adams-Bashforth de ordem 2 para resolver um sistema de equações diferenciais de primeira ordem.

        Args:
            t0 (float): Condição inicial para o tempo.
            u0 (numpy.ndarray): Condição inicial para o vetor de estados.
            h (float): Tamanho do passo de integração.
            num_passos (int): Número de passos de integração.

        Returns:
            tuple: Tupla contendo duas listas, onde a primeira lista contém os valores de tempo e a segunda lista contém
                os valores do vetor de estados ao longo da integração.
        """

        t = [t0]  # Lista para armazenar os valores de t
        u = [u0]  # Lista para armazenar os valores de u

        # Inicialização usando o método de Euler explícito
        t.append(t[0] + h)
        u.append(u[0] + h * self.f(u[0], t[0]))

        for i in range(2, num_passos + 1):
            # Adição do próximo valor de t e u às listas
            t.append(t[i-1] + h)
            u.append(u[i-1] + (h / 2) * (3 * self.f(u[i-1], t[i-1]) - self.f(u[i-2], t[i-2])))

        return t, u

    def newton(self, f, x0, f_prime, tol=1e-2, max_iter=50):
        x = x0
        for _ in range(max_iter):
            fx = f(x)
            if np.linalg.norm(fx) < tol:
                return x
            dfx = f_prime(x)
            x -= solve(dfx, fx)
        raise ValueError("O método de Newton atingiu o número máximo de iterações sem convergir.")

    def euler_implicito(self, t0, u0, h, num_passos):
        """
        Implementa o método de Euler implícito para resolver um sistema de equações diferenciais de primeira ordem.

        Args:
            t0 (float): Condição inicial para o tempo.
            u0 (numpy.ndarray): Condição inicial para o vetor de estados.
            h (float): Tamanho do passo de integração.
            num_passos (int): Número de passos de integração.

        Returns:
            tuple: Tupla contendo duas listas, onde a primeira lista contém os valores de tempo e a segunda lista contém
                os valores do vetor de estados ao longo da integração.
        """

        t = [t0]  # Lista para armazenar os valores de t
        u = [u0]  # Lista para armazenar os valores de u

        for i in range(num_passos):
            t_i = t[-1]
            u_i = u[-1]

            def f_i(u_prox):
                return u_prox - u_i - h * self.f(u_prox, t_i + h)

            def f_prime_i(u_prox):
                return np.eye(2) - h * self.f_jac(u_prox, t_i + h)

            # Usando o método de Newton para encontrar u_{i+1}
            u_prox = newton(f_i, u_i, f_prime_i)

            t.append(t_i + h)
            u.append(u_prox)

        return t, u

    def preditor_corretor(self, t0, u0, h, num_passos):
        """
        Implementa o método do preditor-corretor para resolver um sistema de equações diferenciais de primeira ordem.

        Args:
            t0 (float): Condição inicial para o tempo.
            u0 (numpy.ndarray): Condição inicial para o vetor de estados.
            h (float): Tamanho do passo de integração.
            num_passos (int): Número de passos de integração.

        Returns:
            tuple: Tupla contendo duas listas, onde a primeira lista contém os valores de tempo e a segunda lista contém
                os valores do vetor de estados ao longo da integração.
        """

        t = [t0]  # Lista para armazenar os valores de t
        u = [u0]  # Lista para armazenar os valores de u

        for i in range(num_passos):
            t_i = t[-1]
            u_i = u[-1]

            # Preditor (Euler Explícito)
            u_pred = u_i + h * self.f(u_i, t_i)

            # Corretor (Método do Trapézio)
            f_pred = self.f(u_pred, t_i + h)
            u_corr = u_i + (h / 2) * (self.f(u_i, t_i) + f_pred)

            t.append(t_i + h)
            u.append(u_corr)

        return t, u

    def print_resultados(self):
        """
        Imprime os resultados das soluções numéricas e da solução analítica.

        Returns:
            None
        """

        # Imprime a solução analítica
        print('\nSolução Analítica: t = {:.2f}, u1 = {:.7f}, u2 = {:.7f}'.format(
            self.tf, self.u_ref[0], self.u_ref[1]))

        # Imprime os resultados do método de Euler Explícito
        print('\nEuler Explícito:  t = {:.2f}, u1 = {:.7f}, u2 = {:.7f}'.format(
            self.t_euler_e[-1], self.u_euler_e[-1][0], self.u_euler_e[-1][1]))

        # Imprime os resultados do método de Taylor de ordem 2
        print('Taylor 2:  t = {:.2f}, u1 = {:.7f}, u2 = {:.7f}'.format(
            self.t_taylor_2[-1], self.u_taylor_2[-1][0], self.u_taylor_2[-1][1]))

        # Imprime os resultados do método de Adams-Bashforth de ordem 2
        print('Adams Bashforth 2:  t = {:.2f}, u1 = {:.7f}, u2 = {:.7f}'.format(
            self.t_AB_2[-1], self.u_AB_2[-1][0], self.u_AB_2[-1][1]))

        # Imprime os resultados do método de Euler Implícito
        print('Euler Implícito: t = {:.2f}, u1 = {:.7f}, u2 = {:.7f}'.format(
            self.t_euler_i[-1], self.u_euler_i[-1][0], self.u_euler_i[-1][1]))

        # Imprime os resultados do método Preditor-Corretor
        print('Preditor-Corretor: t = {:.2f}, u1 = {:.7f}, u2 = {:.7f}'.format(
            self.t_PC[-1], self.u_PC[-1][0], self.u_PC[-1][1])+'\n')

        return

    def plot_vel_pos(self, titulo):
        """
        Gera um gráfico com a posição angular (u1) e a velocidade angular (u2) em função do tempo (t).

        Args:
            titulo (str): Título do gráfico. Deve ser um dos seguintes valores: "euler_explicito", "taylor_2",
                "adams_bashforth2", "euler_implicito", "preditor_corretor", "referencia".

        Returns:
            None
        """

        # Extrair os valores de u1 e u2

        if titulo == "euler_explicito":
            u1_vals = [u[0] for u in self.u_euler_e]
            u2_vals = [u[1] for u in self.u_euler_e]
            t = self.t_euler_e
        elif titulo == "taylor_2":
            u1_vals = [u[0] for u in self.u_taylor_2]
            u2_vals = [u[1] for u in self.u_taylor_2]
            t = self.t_taylor_2
        elif titulo == "adams_bashforth2":
            u1_vals = [u[0] for u in self.u_AB_2]
            u2_vals = [u[1] for u in self.u_AB_2]
            t = self.t_AB_2
        elif titulo == "euler_implicito":
            u1_vals = [u[0] for u in self.u_euler_i]
            u2_vals = [u[1] for u in self.u_euler_i]
            t = self.t_euler_i
        elif titulo == "preditor_corretor":
            u1_vals = [u[0] for u in self.u_PC]
            u2_vals = [u[1] for u in self.u_PC]
            t = self.t_PC
        elif titulo == "referencia":
            u1_vals = [u[0] for u in self.u_ref]
            u2_vals = [u[1] for u in self.u_ref]
            t = self.t_lin
        else:
            print("Titulo invalido")
            return None

        # Plotar o gráfico de u1(t) e u2(t) no mesmo gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(t, u1_vals, 'b-', label='Posição Angular (u1)')
        plt.plot(t, u2_vals, 'r-', label='Velocidade Angular (u2)')
        plt.xlabel('Tempo (t)')
        plt.ylabel('Valor')
        plt.legend()
        plt.legend(loc='upper right', bbox_to_anchor=(
            1, 1.14), fancybox=True, shadow=True)
        plt.title(titulo)

        # Salvar o gráfico como uma imagem
        if not os.path.exists('graficos_vel_pos'):
            os.makedirs('graficos_vel_pos')
        plt.savefig('graficos_vel_pos/' + titulo + '.png')

        plt.close()

        return None

    def erros_metodos(self, metodo, titulo):
        """
        Calcula e plota a convergência do erro de aproximação para um determinado método numérico.

        Args:
            metodo (callable): Função que implementa o método numérico a ser avaliado.
            titulo (str): Título do gráfico. Deve ser um dos seguintes valores: "euler_explicito", "taylor_2",
                "adams_bashforth2", "euler_implicito", "preditor_corretor".

        Returns:
            None
        """

        if titulo == "referencia":
            print("Titulo invalido")
            return None

        # Parâmetros da simulação
        h_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]

        # Cálculo e plotagem da ordem de convergência temporal para o método
        error_u1_list = []
        error_u2_list = []

        for h in h_values:
            num_passos = round((self.tf-self.t0)/h)
            t, u_metodo = metodo(self.t0, self.u0, h, num_passos)

            # Cria uma lista de valores de tempo uniformemente espaçados
            t_lin = np.linspace(self.t0, self.tf, num_passos+1)
            # Calcula os valores de referência do vetor de estados

            u_ref = [self.sol_referencia(t, self.u0) for t in t_lin]

            primeira_coluna_metodo = [linha[0] for linha in u_metodo]
            segunda_coluna_metodo = [linha[1] for linha in u_metodo]

            primeira_coluna_ref = [linha[0] for linha in u_ref]
            segunda_coluna_ref = [linha[1] for linha in u_ref]

            error_u1 = self.calculate_error(primeira_coluna_ref, primeira_coluna_metodo)
            error_u2 = self.calculate_error(segunda_coluna_ref, segunda_coluna_metodo)

            error_u1_list.append(error_u1)
            error_u2_list.append(error_u2)

        # Plotagem do gráfico de convergência do erro de aproximação
        plt.loglog(h_values, error_u1_list, '-o')
        plt.xlabel('Tamanho do Passo (h)')
        plt.ylabel('Erro de Aproximação')
        plt.title('Gráfico de Convergência do Erro da Posição' + titulo)
        plt.grid(True)

        if not os.path.exists('graficos_erros'):
            os.makedirs('graficos_erros')
        plt.savefig('graficos_erros/' + titulo + "_pos" + '.png')

        plt.close()

        # Plotagem do gráfico de convergência do erro de aproximação
        plt.loglog(h_values, error_u2_list, '-o')
        plt.xlabel('Tamanho do Passo (h)')
        plt.ylabel('Erro de Aproximação')
        plt.title('Gráfico de Convergência do Erro da Velocidade' + titulo)
        plt.grid(True)

        if not os.path.exists('graficos_erros'):
            os.makedirs('graficos_erros')
        plt.savefig('graficos_erros/' + titulo + "_vel" + '.png')

        plt.close()

        return None

    def calculate_error(self, u_exact, u_approx):
        """
        Calcula o erro absoluto entre a solução exata e a solução aproximada.

        Args:
            u_exact (list): Lista contendo os valores exatos da solução.
            u_approx (list): Lista contendo os valores aproximados da solução.

        Returns:
            float: Valor do erro absoluto.
        """

        # Calculando o erro absoluto
        erro_absoluto = np.abs(np.subtract(u_exact, u_approx))

        # Calculando a norma máxima
        norma_max = np.amax(np.abs(erro_absoluto))

        return norma_max

    def plot_grafico_fase(self, metodo, titulo):
        """
        Gera um gráfico de fase com a posição angular (u[1]) no eixo x e a velocidade angular (u[2]) no eixo y.

        Args:
            metodo (function): Função que implementa o método numérico.
            titulo (str): Título do gráfico.

        Returns:
            None
        """

        if titulo == "referencia":
            t = self.t_lin
            u = self.u_ref
        else:
            t, u = metodo(self.t0, self.u0, self.h, self.num_passos)

        primeira_coluna = [linha[0] for linha in u]
        segunda_coluna = [linha[1] for linha in u]

        # Plotando o gráfico de fase
        plt.figure(figsize=(10, 6))
        plt.plot(primeira_coluna, segunda_coluna, label=titulo)

        plt.xlabel('Posição Angular (u[1])')
        plt.ylabel('Velocidade Angular (u[2])')
        plt.title('Gráfico de Fase')
        plt.legend()
        plt.grid(True)

        if not os.path.exists('graficos_fase'):
            os.makedirs('graficos_fase')
        plt.savefig('graficos_fase/' + titulo + '.png')
        plt.close()

    def tempos_execucao(self):
        """
        Calcula e plota os tempos de execução para cada método em função do tamanho do passo de tempo.

        Returns:
            None
        """

        h_values = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

        tempos = {"referencia": [], "euler_explicito": [], "taylor_2": [],
                  "adams_bashforth2":  [], "euler_implicito": [],
                  "preditor_corretor":  []}

        for i in h_values:
            for metodo in self.metodos_dict.keys():
                if metodo == "referencia":
                    start = time.time()
                    self.u_ref = [self.sol_referencia(
                        t, self.u0) for t in self.t_lin]
                    end = time.time()
                    tempos[metodo].append(end - start)
                    continue

                start = time.time()
                self.metodos_dict[metodo](self.t0, self.u0, i, self.num_passos)
                end = time.time()
                tempos[metodo].append(end - start)

        # Plotando o gráfico
        plt.figure(figsize=(8, 6))

        for method, times in tempos.items():
            plt.scatter(h_values, times, label=method)

        plt.xlabel('Tamanho do Passo de Tempo (h)')
        plt.ylabel('Tempo de Execução (s)')
        plt.title('Tempo de Execução em Função do Tamanho do Passo de Tempo')
        plt.legend()
        plt.grid(True)

        if not os.path.exists('tempos_execucoes'):
            os.makedirs('tempos_execucoes')
        plt.savefig('tempos_execucoes/tempos.png')
        plt.close()
