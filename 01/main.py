from functions import *

# Intervalo
a = -1
b = 1

# Cálculo da solução de referência
h_ref = 1/256
tam_malha = tamanho_malha(a, b, h_ref)

valores_ref = aprox_u(a, b, h_ref)

# Grafico dos valores de referencia
plot_referencia(valores_ref, tam_malha)

# Vetor dos valores de h
h = [1/2, 1/4, 1/8, 1/64, 1/128]

# Criação da malha e dos valores_h
valores_h = valores_h_calc(h)

# Vetor dos truncamentos dos valores da malha analisada e da malha da solução de referência
truncamento = truncamento_vet(a, b, h, h_ref, valores_ref, valores_h)

# Calcula o erro
erro = erro_calc(truncamento)

# Gerando tabela de resultados
tabela_resultados(h, erro)

# Plotagem da convergência do método
plot_convergencia(h, erro)
