from Metodos import *

# Condição inicial t(0)
t0 = 0
# Condição inicial u(0) = [u1(0), u2(0)]
u0 = np.array([np.pi/4, 0])
# Tamanho do passo
h = 0.1
# Número de passos
num_passos = 300
# Tempo final
tf = t0 + h*num_passos

metodos = Metodos(t0, u0, h, num_passos, tf)


# Solução exata
metodos.print_resultados()

# # Parâmetros da simulação
# t0 = 0.0
# num_meshes = 5
# h_values = []

# # Realiza a simulação para diferentes tamanhos de malha
# for i in range(num_meshes):
#     num_steps = 2**(i + 4)  # Define o número de passos como uma potência de 2
#     h = (tf - t0) / num_steps
#     h_values.append(h)

#     # Solução exata
#     t_exact = np.linspace(t0, tf, num_steps + 1)

#     a, b = preditor_corretor(f, t0, u0, h, num_steps)

#     # Cálculo do erro
#     error = calculate_error(u_ref, b[-1])

#     print(len(error), len(h_values))
#     # Plota o gráfico do erro
#     # plot_convergence_order(error, 2)
