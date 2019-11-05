from math import hypot
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def VdPoly (init , t ):#
    """ Attention d ’ abord y ( t) puis t """
    x,y,vx,vy,m = init[0], init[1], init[2], init[3], (6 * 10 ** 24) / (2 * 10 ** 30 + 6 * 10 ** 24)
    r1,r2 = (1 - m -x) ** 2 + y ** 2, (x + m) ** 2 + y ** 2

    ay = -2 * vx - m * y / (r1 ** (3/2)) - (1 - m) * y / (r2 ** (3/2)) + y
    ax = 2 * vy - m * (x - 1 + m) / (r1 ** (3/2)) - (1 - m) * (x + m) / (r2 ** (3/2)) + x

    return  (vx,vy,ax,ay)

plt.scatter(0, 0, s=200, c='purple')
plt.scatter(1, 0, s=200, c='red')

cond_ini = (0.5,0.7,0,0)
t = np.linspace (0 ,20 ,300*20) # on simule sur 1000 points de T = [0; 20]
X,Y,DX,DY = odeint(VdPoly, cond_ini, t).T

plt.plot (np.array(X), np.array(Y))


"""
x = np.linspace ( -1 ,1 ,30) # subdiviser l ’ intervalle de x [−4; 4]
y = np.linspace ( -1 ,1 ,30) # subdiviser l ’ intervalle de y [−4; 4]
X1 , Y1 = np.meshgrid (x ,y )

dx, dy = [], []
for i in x :
    for j in y :
        YX = odeint(VdPoly,( i , j , 0 , 0 ) ,(1,2,2)) # generer les vecteurs tangents
        dx.append(YX[1][0])
        dy.append(YX[1][1])

dX=np.array(dx)
dY=np.array(dy)
M = np.hypot( dX , dY ) # normalisation
M[M == 0] = 1.0 # Normes d ’ eventuels vecteurs nuls
dX /= M # remplaces par 1 avant division
dY /= M
plt.quiver ( Y1 ,X1 ,dY , dX , M ) # generation du champs de vecteurs
"""

plt.show()