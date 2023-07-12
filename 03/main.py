from Metodos import Metodos


# Condições iniciais - CASO A
h = 0.05
k = (5/11)*(h**2)
dom_t = [0, 2]
dom_x = [0, 1]

# # Condições iniciais - CASO B
# h = 0.05
# k = (5/9)*(h**2)
# dom_t = [0, 2]
# dom_x = [0, 1]

m = Metodos(h, k, dom_x, dom_t)

m.plot_ref()

m.plot_numerica(m.u_exp, 'explicito', 'Método Explícito')
m.comparacao(dom_x, h, k, 'explicito')

m.plot_numerica(m.u_crank, 'crank_nicolson', 'Crank-Nicolson')
m.comparacao(dom_x, h, k, 'Crank-Nicolson')

m.erro_convergencia(dom_x, dom_t, 'Explicito')
m.erro_convergencia(dom_x, dom_t, 'Crank-Nicolson')
