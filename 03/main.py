from Metodos import *


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

metodos = Metodos(h, k, lim_t, lim_sum)

metodos.plot_ref()

metodos.plot_numerica(metodos.U_exp, 'U_exp', 'Solução numérica explícita')
metodos.comparacao(metodos.U_exp, 'Explicito')

metodos.plot_numerica(metodos.U_cn, 'U_cn', 'Solução Crank-Nicolson')
metodos.comparacao(metodos.U_cn, 'Crank-Nicolson')

metodos.erro_convergencia('Explicito')
metodos.erro_convergencia('Crank-Nicolson')
