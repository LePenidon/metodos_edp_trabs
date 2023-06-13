from functions import *

# Condição inicial t(0)
t0 = 0
# Condição inicial u(0) = [u1(0), u2(0)]
u0 = np.array([np.pi/4, 0])
# Tamanho do passo
h = 0.1
# Número de passos
num_passos = 300
num_malhas = 5  # Número de malhas diferentes
ordens = [1, 2, 3, 4]  # Ordem dos métodos a serem testados
tf = t0 + h*num_passos


u_ref = RefSolution(u0[0], tf)

t_euler_e, u_euler_e = euler_explicito(f, t0, u0, h, num_passos)
t_taylor_2, u_taylor_2 = taylor_2(f, f_jac, t0, u0, h, num_passos)
t_AB_2, u_AB_2 = adams_bashforth2(f, t0, u0, h, num_passos)
t_euler_i, u_euler_i = euler_implicito(f, t0, u0, h, num_passos)
t_PC, u_PC = preditor_corretor(f, t0, u0, h, num_passos)

plot_vel_pos(u_euler_e, t_euler_e, "Euler Explícito")
plot_vel_pos(u_taylor_2, t_taylor_2, "Taylor 2")
plot_vel_pos(u_AB_2, t_AB_2, "Adams-Bashforth 2")
plot_vel_pos(u_euler_i, t_euler_i, "Euler Implícito")
plot_vel_pos(u_PC, t_PC, "Preditor-Corretor")

print(u_ref)
print(u_PC[-1])

# Parâmetros da simulação
t0 = 0.0
num_meshes = 5
h_values = []

# Realiza a simulação para diferentes tamanhos de malha
for i in range(num_meshes):
    num_steps = 2**(i + 4)  # Define o número de passos como uma potência de 2
    h = (tf - t0) / num_steps
    h_values.append(h)

    # Solução exata
    t_exact = np.linspace(t0, tf, num_steps + 1)

    a, b = preditor_corretor(f, t0, u0, h, num_steps)

    # Cálculo do erro
    error = calculate_error(u_ref, b[-1])

    print(len(error), len(h_values))
    # Plota o gráfico do erro
    # plot_convergence_order(error, 2)
