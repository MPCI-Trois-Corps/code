import numpy as np
import matplotlib.pyplot as plt


def euler_explicite(f, tinit, Tfinal, yinit, h):
    N = round((Tfinal - tinit) / h)
    if isinstance(yinit, np.ndarray):
        d = len(yinit)
    else:
        d = 1
    y = np.zeros((d, N+1))
    y[:, 0] = yinit
    tn = tinit
    for n in range(N):
        y[:, n+1] = y[:, n]+h*np.dot(f,y[:, n])
        tn = tn + h
    return y


def f0(t, y):
    a,b,c,d = 1,1,1,1
    A = np.array([[a,-y[0]*b],[y[1],-1]])
    B = A.dot(y)
    return B

ti = 0
Tf = 1
h = 0.01
A = np.array([[0,0,1,0],[0,0,0,1],[0.75,1.3,0,2],[1.3,2.25,-2,0]])

"""
yi = np.array([0.5,np.sin(np.pi/3),0,0])
N = round((Tf-ti)/h)
t = (ti + np.arange(N+1)*h).reshape((1, N+1))
yschem = euler_explicite(A, ti, Tf, yi, h)
plt.plot(yschem[0],yschem[1])
"""

plt.scatter(0, 0, s=200, c='purple')
plt.scatter(1, 0, s=200, c='red')


x = np.linspace ( 0 ,1 ,15) # subdiviser l ’ intervalle de x [−4; 4]
y = np.linspace ( 0 ,1 ,15) # subdiviser l ’ intervalle de y [−4; 4]
X1 , Y1 = np.meshgrid (x ,y )

dx, dy = [], []
for i in x :
    for j in y :
        YX = euler_explicite(A,ti,1,np.array([i,j,0,0]),h) # generer les vecteurs tangents
        dx.append(YX[0,1]+0.5)
        dy.append(YX[1,1]+np.sin(np.pi/3))

dX=np.array(dx)
dY=np.array(dy)
M = np.hypot( dX , dY ) # normalisation
M[M == 0] = 1.0 # Normes d ’ eventuels vecteurs nuls
dX /= M # remplaces par 1 avant division
dY /= M
plt.quiver ( Y1 ,X1 ,dY , dX , M ) # generation du champs de vecteurs


plt.show()
