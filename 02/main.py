from Metodos import *

# Condição inicial t(0)
t0 = 0
# Condição inicial u(0) = [u1(0), u2(0)]
u0 = np.array([np.pi/4, 0])
# Tamanho do passo
h = 0.01
# Número de passos
num_passos = 10000

metodos = Metodos(t0, u0, h, num_passos)
