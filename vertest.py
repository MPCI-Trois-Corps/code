import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r', animated=True)

# fonction à définir quand blit=True
# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    ax.set_xlim(-160*10**9,160*10**9)
    ax.set_ylim(-160*10**9, 160*10**9)
    ln.set_data(xdata,ydata)
    return ln,

def euler_pas_trop_expli(xi,yi,h,t_int,T_fin):

    #conditions initiales terre/soleil
    K = 6.7 * 10 ** (-11)
    m0 = 1.9 * 10 ** 30
    m1 = 5.9 * 10 ** 24
    a = 149 * 10 **9
    nu = m1 / (m0+m1)
    n = np.sqrt(K*(m0+m1)/a**3)

    L = []
    M = []
    x = xi
    y = yi

    #plt.scatter(xi, yi, s=150, c='green')

    #vitesse initiale
    xp = 10000
    yp = -25000

    while t_int <= T_fin :
        var1 = xp
        var2 = yp

        xp = var1 + h*(2*n*var2+n**2*(x-nu*a)-K*m0*x*(x**2+y**2)**(-3/2)+K*m1*(a-x)*((a-x)**2+y**2)**(-3/2))
        x += h * xp

        yp = var2 + h*((-2)*n*var1+n**2*y-K*m0*y*(x**2+y**2)**(-3/2)-K*m1*y*((a-x)**2+y**2)**(-3/2))
        y += h * yp

        t_int += h
        L.append(x)
        M.append(y)
    return L,M


def update(frame):
    xdata.append(L[frame])
    ydata.append(M[frame])
    ln.set_data(xdata, ydata)
    return ln,

#terre/soleil
plt.scatter(0,0,s=200,c='purple')
plt.scatter(149 * 10 ** 9,0,s=200,c='red')

global L,M
L,M = euler_pas_trop_expli(75000000000,-90000000000,1440,0,14400000)

#on affiche
#x = np.array(L)
#y = np.array(M)
#plt.plot(x,y)

ani = FuncAnimation(fig, update, frames=10000, init_func=init, blit=True, interval = 0.001,repeat=False)
plt.show()


