from functions import *

# Cálculo da solução de referência
h_barra = 1/256
m = M(h_barra)

# [(U(h) != 0)&(U(h) != 1)]
valores_referencia = U(h_barra)

plot_referencia(valores_referencia, m)

# Vetor dos valores de h
h = [1/2, 1/4, 1/8, 1/64, 1/128]

malhas = malhas_calc(h)
valores_h = valores_h_calc(h)

# Vetor dos truncamentos dos valores da malha analisada e da malha da solução de referência
truncamento = truncamento_vet(h, h_barra, valores_referencia, valores_h)

erro = erro_calc(truncamento)

# Gerando tabela de resultados
tabela_resultados(h, erro)

# Plotagem da convergência do método
plot_convergencia(h, erro)
