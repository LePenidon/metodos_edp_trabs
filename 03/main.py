from Metodos import Metodos


# Condições iniciais - CASO A
h = 0.05
k = (5/11)*(h**2)
dom_t = [0, 2]
dom_x = [0, 1]

# # Condições iniciais - CASO B
# h = 0.05
# k = (5/9)*(h**2)
# lim_t = 2
# lim_sum = 1000

m = Metodos(h, k, dom_x, dom_t)

m.plot_ref()

m.plot_numerica(dom_x, 0, h, k 'explicito', 'Método Explícito')

# m.comparacao(m.u_exp, 'explicito')

# m.plot_numerica(m.crank_nicolson, 'crank_nicolson', 'Solução Crank-Nicolson')
# m.comparacao(m.u_crank, 'Crank-Nicolson')

# m.erro_convergencia('Explicito')
# m.erro_convergencia('Crank-Nicolson')
