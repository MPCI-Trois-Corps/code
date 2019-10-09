__author__ = 'raymond_jr'
__Filename__ = 'explicit_euler_normalized'
__Creationdate__ = '25/09/2019'

#from decimal import Decimal,getcontext
import numpy as np
import matplotlib.pyplot as plt

#getcontext().prec = 10

def explicit_euler_normalized(t_init, t_final, time_step, r_init, v_init, mass_ratio):
    """Returns the list of [[(x_i, y_i)], [(v_x_i, v_y_i)] ]
    for 0 <= i <= N := (t_final - t_init) / time_step."""

    # Initial conditions
    N = int((t_final - t_init)/time_step)
    dt = time_step

    x , y = r_init

    v_x, v_y = v_init
    v_x = v_x / ((149*10**9)*(2*10**(-7)))
    v_y = v_y / ((149*10**9)*(2*10**(-7)))

    m = mass_ratio
    result = [[(x, y)], [(v_x, v_y)]]

    for i in range(1, N + 1):

        # Computation of X = x(i+1), Y = y(i+1), u_x = v_x(i+1) and u_y = v_y(i+1)
        a_x = 2 * v_y - m * (x + 1 - m)*((x + 1 - m)**2 + y**2)**(-3/2) - (1-m) * (x - m)*((x-m)**2 + y**2)**(-3/2) + x
        a_y = -2 * v_x - m * y * ((x + 1 - m)**2 + y**2)**(-3/2) - (1 - m)*y*((x - m)**2 + y**2)**(-3/2) + y

        u_x = v_x + dt*a_x
        u_y = v_y + dt*a_y

        X = x + dt*v_x
        Y = y + dt*v_y

        # Updating variables
        x, y, v_x, v_y = X, Y, u_x, u_y

        # Updating result
        result[0].append((x, y))
        result[1].append((v_x, v_y))

    return result

plt.scatter(0,0,s=200,c='purple')
plt.scatter(1,0,s=200,c='red')

masse_r = (6*10**24)/(2*10**30 + 6*10**24)

L = explicit_euler_normalized(0,10000,0.1,(0.5,0.8),(0,0),masse_r)


#on affiche
x = np.array(L[0][0])
y = np.array(L[0][1])
plt.plot(x,y)

plt.show()