from Metodos import *

# Condição inicial t(0)
t0 = 0
tf = 50

# Condição inicial u(0) = [u1(0), u2(0)]
u0 = np.array([np.pi/4, 0])

# Tamanho do passo
h = 0.01

metodos = Metodos(t0, tf, u0, h)
