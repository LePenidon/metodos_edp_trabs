# Importa a classe Metodos do módulo Metodos
from Metodos import Metodos

# Define as condições iniciais para o problema - CASO A
h = 0.05
k = (5/11) * (h**2)
dom_t = [0, 2]
dom_x = [0, 1]

# # Condições iniciais - CASO B
# h = 0.05
# k = (5/9)*(h**2)
# dom_t = [0, 2]
# dom_x = [0, 1]


# Cria uma instância da classe Metodos com as condições iniciais definidas
m = Metodos(h, k, dom_x, dom_t)

# Gera e plota a solução de referência (solução analítica) para o problema
m.plot_ref()

# Gera e plota a solução numérica usando o método explícito
m.plot_numerica(m.u_exp, 'explicito', 'Método Explícito')

# # Gera e plota a comparação entre as soluções numérica e de referência para o método explícito
m.comparacao(dom_x, h, k, 'explicito')

# Gera e plota a solução numérica usando o método Crank-Nicolson
m.plot_numerica(m.u_crank, 'crank_nicolson', 'Crank-Nicolson')

# # Gera e plota a comparação entre as soluções numérica e de referência para o método Crank-Nicolson
m.comparacao(dom_x, h, k, 'Crank-Nicolson')

# # Gera e plota o gráfico de convergência do erro para o método explícito
m.erro_convergencia(dom_x, dom_t, 'Explicito')

# # Gera e plota o gráfico de convergência do erro para o método Crank-Nicolson
m.erro_convergencia(dom_x, dom_t, 'Crank-Nicolson')
