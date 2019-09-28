import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# fonction à définir quand blit=True
# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    k = max(max(L),abs(min(L)),max(M),abs(min(M)),160*10**9)
    ax.set_xlim(-k,k)
    ax.set_ylim(-k,k)
    ln.set_data(xdata,ydata)
    tn.set_data(terrexdata,terreydata)
    return ln,  tn,

def euler_pas_trop_expli(xi,yi,h,t_int,T_fin):

    #conditions initiales terre/soleil
    K = 6.7 * 10 ** (-11)

    nu = m1 / (m0+m1)
    n = (K*(m0+m1)/a**3)**(1/2)

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
    xdata.append(carte(polaire(L[frame],M[frame])[0],polaire(L[frame],M[frame])[1]+frame*angle)[0])
    ydata.append(carte(polaire(L[frame],M[frame])[0],polaire(L[frame],M[frame])[1]+frame*angle)[1])
    terrexdata.append(carte(polaire(a, 0)[0], polaire(a, 0)[1] + frame * angle)[0])
    terreydata.append(carte(polaire(a, 0)[0], polaire(a, 0)[1] + frame * angle)[1])
    ln.set_data(xdata, ydata)
    tn.set_data(terrexdata[-1:],terreydata[-1:])
    return ln, tn,

def polaire(x,y):
    rho = (x**2+y**2)**(1/2)
    if y >= 0 :
        theta = np.arccos(x/((x**2+y**2)**(1/2)))
    else :
        theta = 2*np.pi - np.arccos(x/((x**2+y**2)**(1/2)))
    return (rho,theta)

def carte(pho,theta):
    x = pho * np.cos(theta)
    y = pho * np.sin(theta)
    return (x,y)

nombre_image = int(input("Nombre d'images voulues : "))
dureeimage = int(input("Nombre de secondes entre deux images : "))

global m0,m1,a,angle
choix = 2
while choix != 0 :
    choix = int(input("1 pour prendre les 2 points que vous voulez, 0 sinon : "))
    if choix == 0:
        m0 = 1.9 * 10 ** 30
        m1 = 5.9 * 10 ** 24
        a = 149 * 10 ** 9
        angle = dureeimage * (2 * np.pi) / (365 * 24 * 60 * 60)
    elif choix ==1 :
        m0 = int(input("Masse Point0 :"))
        m1 = int(input("Masse Point1 :"))
        a = int(input("Distance Point0-Point1 :"))
        rota = int(input("Nombre de jours pour rotation de P1 autour de P0"))
        angle = dureeimage * (2*np.pi) / (rota * 24 * 60 * 60)
        choix = 0

global L,M
L,M = euler_pas_trop_expli(75000000000,-90000000000,dureeimage,0,nombre_image*dureeimage)

fig, ax = plt.subplots()
xdata, ydata, terrexdata, terreydata = [], [], [149*10**9], [0]
ln, = plt.plot([], [], 'r', animated=True)
tn, = plt.plot([], [], 'ro', animated=True, c='green')
plt.axis("equal")

#terre/soleil
plt.scatter(0,0,s=200,c='purple')
#plt.scatter(149 * 10 ** 9,0,s=200,c='red')

#on affiche
#x = np.array(L)
#y = np.array(M)
#plt.plot(x,y)

ani = FuncAnimation(fig, update, frames=nombre_image, init_func=init, blit=True, interval = 0.001,repeat=False)
plt.show()


