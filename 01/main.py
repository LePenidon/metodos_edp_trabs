from functions import *

# Intervalo
a = -1
b = 1
k_2 = 10

# Cálculo da solução de referência
h_ref = 1/32
tam_malha = tamanho_malha(a, b, h_ref)

valores_ref = aprox_u(a, b, h_ref, tam_malha, k_2)

# Grafico dos valores de referencia
plot_referencia(a, b, tam_malha, valores_ref)

# Vetor dos valores de h
h = [1/16, 1/8, 1/2]

# Criação da malha e dos valores_h
valores_h = valores_h_calc(a, b, h, tam_malha, k_2)

# Vetor dos truncamentos dos valores da malha analisada e da malha da solução de referência
erro = truncamento_vet(h, valores_ref, valores_h)

# Gerando tabela de resultados
# tabela_resultados(h, erro)
print(erro)

# Plotagem da convergência do método
plot_convergencia(h, erro)
