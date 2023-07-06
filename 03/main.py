from Metodos import Metodos


# Condições iniciais
h = 0.05
k = (5/11)*(h**2)
lim_t = 2
lim_sum = 100

# # Condições iniciais
# h = 0.05
# k = (5/9)*(h**2)
# lim_t = 2
# lim_sum = 1000

m = Metodos(h, k, lim_t, lim_sum)

m.plot_ref()

m.plot_numerica(m.U_exp, 'U_exp', 'Solução numérica explícita')
m.comparacao(m.U_exp, 'Explicito')

m.plot_numerica(m.U_cn, 'U_cn', 'Solução Crank-Nicolson')
m.comparacao(m.U_cn, 'Crank-Nicolson')

m.erro_convergencia('Explicito')
m.erro_convergencia('Crank-Nicolson')
