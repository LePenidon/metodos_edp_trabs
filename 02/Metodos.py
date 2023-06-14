import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj
from scipy.special import ellipk
from scipy.optimize import newton
import os


class Metodos:

    def __init__(self, t0, u0, h, num_passos, tf) -> None:
        self.t0 = t0
        self.u0 = u0
        self.h = h
        self.num_passos = num_passos
        self.tf = tf

        self.u_ref = self.sol_referencia()

        self.u_ref = self.sol_referencia()
        self.t_euler_e, self.u_euler_e = self.euler_explicito()
        self.t_taylor_2, self.u_taylor_2 = self.taylor_2()
        self.t_AB_2, self.u_AB_2 = self.adams_bashforth2()
        self.t_euler_i, self.u_euler_i = self.euler_implicito()
        self.t_PC, self.u_PC = self.preditor_corretor()

        return None

    def f(self, u, t):
        """
        Função que define o sistema de equações diferenciais.

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
            array([1.0       , -0.47942554])
        """

        u1, u2 = u

        # Define as derivadas de u1 e u2 em relação a t
        dudt = np.array([u2, -np.sin(u1)])

        return dudt

    def f_jac(self, u, t):
        """
        Função que define a matriz jacobiana do sistema de equações diferenciais.

        Args:
            u (list or array): Lista ou array contendo os valores das variáveis u1 e u2.
            t (float): Valor do parâmetro t.

        Returns:
            array: Array contendo a matriz jacobiana.

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

    def sol_referencia(self):
        """
        Calcula a solução de referência do sistema de equações diferenciais.

        Returns:
            array: Array contendo os valores da solução de referência.

        Example:
            >>> obj = MyClass()
            >>> obj.u0 = [0.5, 1.0]
            >>> obj.tf = 2.0
            >>> obj.sol_referencia()
            array([ 3.01190207, -0.15623905])
        """

        # Variável auxiliar
        k_0 = np.sin(0.5*self.u0[0])

        # Integral elíptica incompleta de primeira espécie
        K = ellipk(k_0**2)

        # Funções elípticas de Jacobi: sn e cn
        sn, cn, dn, ph = ellipj(K - self.tf, k_0**2)

        # Deslocamento angular
        q = 2*np.arcsin(k_0*sn)

        # Velocidade angular
        p = -2*k_0*cn

        u = np.array([q, p])

        return u

    def euler_explicito(self):
        """
        Implementa o método de Euler explícito para resolver o sistema de equações diferenciais.

        Returns:
            tuple: Tupla contendo duas listas: a primeira com os valores de t e a segunda com os valores de u.

        Example:
            >>> obj = MyClass()
            >>> obj.t0 = 0.0
            >>> obj.u0 = [0.5, 1.0]
            >>> obj.num_passos = 100
            >>> obj.h = 0.01
            >>> obj.euler_explicito()
            ([0.0, 0.01, 0.02, ..., 0.99, 1.0], [[0.5, 1.0], [0.50734552, 0.99500417], [0.51469103, 0.9900167], ..., [3.00572984, -0.16038433], [3.00622291, -0.15942448]])
        """

        # Lista para armazenar os valores de t
        t = [self.t0]
        # Lista para armazenar os valores de u
        u = [self.u0]

        for i in range(self.num_passos):
            t_i = t[-1]
            u_i = u[-1]
            f_i = self.f(u_i, t_i)
            # Cálculo do próximo valor de u usando o método de Euler explícito
            u_prox = u_i + self.h * f_i
            # Atualização do valor de t
            t.append(t_i + self.h)
            # Adição do próximo valor de u à lista
            u.append(u_prox)

        return t, u

    def taylor_2(self):
        """
        Implementa o método de Taylor de ordem 2 para resolver o sistema de equações diferenciais.

        Returns:
            tuple: Tupla contendo duas listas: a primeira com os valores de t e a segunda com os valores de u.

        Example:
            >>> obj = MyClass()
            >>> obj.t0 = 0.0
            >>> obj.u0 = [0.5, 1.0]
            >>> obj.num_passos = 100
            >>> obj.h = 0.01
            >>> obj.taylor_2()
            ([0.0, 0.01, 0.02, ..., 0.99, 1.0], [[0.5, 1.0], [0.50749998, 0.99500417], [0.51499997, 0.99001666], ..., [3.00624963, -0.16035661], [3.00624962, -0.1603566]])
        """
        t = [self.t0]  # Lista para armazenar os valores de t
        u = [self.u0]  # Lista para armazenar os valores de u

        for i in range(self.num_passos):
            t_i = t[-1]
            u_i = u[-1]
            f_i = self.f(u_i, t_i)  # Calcula o vetor f(u_i, t_i)
            # Calcula a matriz Jacobiana df(u_i, t_i)
            f_jac_i = self.f_jac(u_i, t_i)
            # Cálculo do próximo valor de u usando o método de Taylor de ordem 2
            u_prox = u_i + self.h * f_i + self.h**2 / 2 * np.dot(f_jac_i, f_i)
            # Atualização do valor de t
            t.append(t_i + self.h)
            # Adição do próximo valor de u à lista
            u.append(u_prox)

        return t, u

    def adams_bashforth2(self):
        """
        Implementa o método de Adams-Bashforth de ordem 2 para resolver o sistema de equações diferenciais.

        Returns:
            tuple: Tupla contendo duas listas: a primeira com os valores de t e a segunda com os valores de u.

        Example:
            >>> obj = MyClass()
            >>> obj.t0 = 0.0
            >>> obj.u0 = [0.5, 1.0]
            >>> obj.num_passos = 100
            >>> obj.h = 0.01
            >>> obj.adams_bashforth2()
            ([0.0, 0.01, 0.02, ..., 0.99, 1.0], [[0.5, 1.0], [0.50749998, 0.99500417], [0.51500832, 0.99001496], ..., [3.00625044, -0.1603594], [3.00625225, -0.16035942]])
        """

        # Lista para armazenar os valores de t
        t = [self.t0]
        # Lista para armazenar os valores de u
        u = [self.u0]

        # Inicialização usando o método de Euler explícito
        t_i = self.t0
        u_i = self.u0
        f_i = self.f(u_i, t_i)
        t_prox = t_i + self.h
        u_prox_pred = u_i + self.h * f_i
        t.append(t_prox)
        u.append(u_prox_pred)

        for i in range(2, self.num_passos + 1):
            t_i = t_prox
            u_i = u_prox_pred
            f_i = self.f(u_i, t_i)

            t_prox = t_i + self.h
            u_prox_corr = u_i + (self.h / 2) * \
                (3 * f_i - self.f(u[i-1], t[i-1]))
            # u_prox_corr = u_i + (self.h / 2) * (3 * f_i - self.f(u_i, t_i))

            t.append(t_prox)
            u.append(u_prox_corr)

        return t, u

    def euler_implicito(self):
        """
        Implementa o método de Euler implícito para resolver o sistema de equações diferenciais.

        Returns:
            tuple: Tupla contendo duas listas: a primeira com os valores de t e a segunda com os valores de u.

        Example:
            >>> obj = MyClass()
            >>> obj.t0 = 0.0
            >>> obj.u0 = [0.5, 1.0]
            >>> obj.num_passos = 100
            >>> obj.h = 0.01
            >>> obj.euler_implicito()
            ([0.0, 0.01, 0.02, ..., 0.99, 1.0], [[0.5, 1.0], [0.50166713, 0.99983351], [0.50333427, 0.99966701], ..., [2.234987, -0.423314], [2.2336457, -0.4243009]])
        """
        t = [self.t0]  # Lista para armazenar os valores de t
        u = [self.u0]  # Lista para armazenar os valores de u

        for i in range(self.num_passos):
            t_i = t[-1]
            u_i = u[-1]

            def f_i(u_prox):
                return u_prox - u_i - self.h * self.f(u_prox, t_i + self.h)

            # Usando o método de Newton para encontrar u_{i+1}
            u_prox = newton(f_i, u_i)

            t.append(t_i + self.h)
            u.append(u_prox)

        return t, u

    def preditor_corretor(self):
        # Lista para armazenar os valores de t
        t = [self.t0]
        # Lista para armazenar os valores de u
        u = [self.u0]

        for i in range(self.num_passos):
            t_i = t[-1]
            u_i = u[-1]

            # Preditor (Euler Explícito)
            u_pred = u_i + self.h * self.f(u_i, t_i)

            # Corretor (Método do Trapézio)
            f_pred = self.f(u_pred, t_i + self.h)
            u_corr = u_i + (self.h / 2) * (self.f(u_i, t_i) + f_pred)

            t.append(t_i + self.h)
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

    def plot_vel_pos(u, t, titulo):
        """
        Gera um gráfico com a posição angular (u1) e a velocidade angular (u2) em função do tempo (t).

        Args:
            u (list): Lista contendo os valores de u ao longo do tempo.
            t (list): Lista contendo os valores de tempo correspondentes.
            titulo (str): Título do gráfico.

        Returns:
            None

        Example:
            >>> u = [[0.5, 1.0], [0.50166713, 0.99983351], [0.50333427, 0.99966701], ..., [2.234987, -0.423314], [2.2336457, -0.4243009]]
            >>> t = [0.0, 0.01, 0.02, ..., 0.99, 1.0]
            >>> titulo = 'Gráfico de Posição e Velocidade Angular'
            >>> plot_vel_pos(u, t, titulo)
        """
        # Extrair os valores de u1 e u2
        u1_vals = [u[0] for u in u]
        u2_vals = [u[1] for u in u]

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
        if not os.path.exists('graficos_plot_vel_pos'):
            os.makedirs('graficos_plot_vel_pos')
        plt.savefig('graficos_plot_vel_pos/' + titulo + '.png')

        return None

    def calculate_error(y_exact, y_approx):
        """
        Calcula o erro absoluto entre a solução exata e a solução aproximada.

        Args:
            y_exact: Lista contendo os valores exatos da solução.
            y_approx: Lista contendo os valores aproximados da solução.

        Returns:
            Lista contendo os valores do erro absoluto.
        """
        return np.abs(y_exact - y_approx)

    def plot_convergence_order(errors, step_sizes):
        """
        Plota o gráfico do logaritmo do erro em função do logaritmo do tamanho do passo.

        Args:
            errors: Lista contendo os valores do erro absoluto.
            step_sizes: Lista contendo os tamanhos do passo correspondentes.

        Returns:
            None
        """
        plt.loglog(step_sizes, errors, 'o-', label='Erro')

        # Ajusta a escala do gráfico
        plt.xlabel('Tamanho do Passo (h)')
        plt.ylabel('Erro Absoluto')
        plt.grid(True)
        plt.legend()
        plt.show()
