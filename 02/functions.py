import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj
from scipy.special import ellipk
from scipy.optimize import newton
import os


def f(u, t):
    u1, u2 = u
    return np.array([u2, -np.sin(u1)])


def f_jac(u, t):
    return np.array([[0, 1], [-np.cos(u[0]), 0]])


def RefSolution(q_0, t):

    # Auxiliary variable
    k_0 = np.sin(0.5*q_0)

    # Incomplete elliptic integral of the first kind
    K = ellipk(k_0**2)

    # Jacobi elliptic functions: sn and cn
    sn, cn, dn, ph = ellipj(K - t, k_0**2)

    # Angular displacement
    q = 2*np.arcsin(k_0*sn)

    # Angular velocity
    p = -2*k_0*cn

    u = np.array([q, p])

    return u


def euler_explicito(f, t0, u0, h, num_passos):
    # Lista para armazenar os valores de t
    t = [t0]
    # Lista para armazenar os valores de u
    u = [u0]

    for i in range(num_passos):
        t_i = t[-1]
        u_i = u[-1]
        f_i = f(u_i, t_i)
        # Cálculo do próximo valor de u usando o método de Euler explícito
        u_prox = u_i + h * f_i
        # Atualização do valor de t
        t.append(t_i + h)
        # Adição do próximo valor de u à lista
        u.append(u_prox)

    return t, u


def taylor_2(f, f_jac, t0, u0, h, num_passos):
    t = [t0]  # Lista para armazenar os valores de t
    u = [u0]  # Lista para armazenar os valores de u

    for i in range(num_passos):
        t_i = t[-1]
        u_i = u[-1]
        f_i = f(u_i, t_i)  # Calcula o vetor f(u_i, t_i)
        f_jac_i = f_jac(u_i, t_i)  # Calcula a matriz Jacobiana df(u_i, t_i)
        # Cálculo do próximo valor de u usando o método de Taylor de ordem 2
        u_prox = u_i + h * f_i + h**2/2*(np.dot(f_jac_i, f_i))
        # Atualização do valor de t
        t.append(t_i + h)
        # Adição do próximo valor de u à lista
        u.append(u_prox)

    return t, u


def adams_bashforth2(f, t0, u0, h, num_passos):
    t = [t0]  # Lista para armazenar os valores de t
    u = [u0]  # Lista para armazenar os valores de u

    # Inicialização usando o método de Euler explícito
    t_i = t0
    u_i = u0
    f_i = f(u_i, t_i)
    t_prox = t_i + h
    u_prox_pred = u_i + h * f_i
    t.append(t_prox)
    u.append(u_prox_pred)

    for i in range(2, num_passos + 1):
        t_i = t_prox
        u_i = u_prox_pred
        f_i = f(u_i, t_i)

        t_prox = t_i + h
        u_prox_corr = u_i + (h / 2) * (3 * f_i - f(u[i-1], t[i-1]))

        t.append(t_prox)
        u.append(u_prox_corr)

    return t, u


def euler_implicito(f, t0, u0, h, num_passos):
    t = [t0]  # Lista para armazenar os valores de t
    u = [u0]  # Lista para armazenar os valores de u

    for i in range(num_passos):
        t_i = t[-1]
        u_i = u[-1]
        def f_i(u_prox): return u_prox - u_i - h * f(u_prox, t_i + h)
        # Usando o método de Newton para encontrar u_{i+1}
        u_prox = newton(f_i, u_i)
        t.append(t_i + h)
        u.append(u_prox)

    return t, u


def preditor_corretor(f, t0, u0, h, num_passos):
    t = [t0]  # Lista para armazenar os valores de t
    u = [u0]  # Lista para armazenar os valores de u

    for i in range(num_passos):
        t_i = t[-1]
        u_i = u[-1]

        # Preditor (Euler Explícito)
        u_pred = u_i + h * f(u_i, t_i)

        # Corretor (Método do Trapézio)
        f_pred = f(u_pred, t_i + h)
        u_corr = u_i + (h / 2) * (f(u_i, t_i) + f_pred)

        t.append(t_i + h)
        u.append(u_corr)

    return t, u


def plot_vel_pos(u, t, titulo):
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
