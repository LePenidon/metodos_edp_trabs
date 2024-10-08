from functions import *

# Parametros do problema
a = -1
b = 1
k_2 = 10
# Solucao da malha fina
h_ref = 1/512
# Valores de h a serem analisados
h = [1/128, 1/64, 1/32, 1/16]

# Calculo da solucao de referencia
tam_malha = tamanho_malha(a, b, h_ref)
sol_ref = aprox_sol(a, h_ref, tam_malha, k_2)

# Plot da solucao de referencia
plot_referencia(a, b, tam_malha, sol_ref)

# Resolucao dos valores de u para cada h
valores_h = valores_h_calc(a, b, h, k_2)

# Calculo dos erros
erros = erro_calc(a, b, h, sol_ref, valores_h, tam_malha)

# Plot dos erros
plot_erros(h, erros)
