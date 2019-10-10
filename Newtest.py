#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:50:11 2019

@author: llb
"""

__author__ = 'raymond_jr'
__Filename__ = 'explicit_euler_normalized'
__Creationdate__ = '25/09/2019'

#from decimal import Decimal,getcontext
import math
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
    #v_x = v_x / ((149*10**9)*(2/(10**7)))
    #v_y = v_y / ((149*10**9)*(2/(10**7)))
    
    m = mass_ratio
    result = [[(x, y)], [(v_x, v_y)]]  
    r1 = np.sqrt((x + 1 - m)**(2) + y**(2))
    r2 = np.sqrt((x-m)**(2) + y**(2))
    
    for i in range(1, N + 1):

        # Computation of X = x(i+1), Y = y(i+1), u_x = v_x(i+1) and u_y = v_y(i+1)
        
#        a_x = 2 * v_y - m * (x + 1 - m)*((x + 1 - m)**(2) + y**(2))**(-3/2) - (1-m) * (x - m)*((x-m)**(2) + y**(2))**(-3/2) + x
#        a_y = -2 * v_x - m * y * ((x + 1 - m)**(2) + y**(2))**(-3/2) - (1 - m)*y*((x - m)**(2) + y**(2))**(-3/2) + y
        a_x = 2 * v_y - m * (x + 1 - m)/(r1**3) - (1-m) * (x - m)/(r2**3) + x
        a_y = -2 * v_x - m * y/(r1**3) - (1 - m)*y/(r2**3) + y

        u_x = v_x + dt*a_x
        u_y = v_y + dt*a_y

        X = x + dt*v_x + (dt**2)/2 * a_x
        Y = y + dt*v_y + (dt**2)/2 * a_y

        # Updating variables
        x, y, v_x, v_y = X, Y, u_x, u_y
        r, v = (x, y), (v_x, v_y)
        r1 = np.sqrt((x+1-m)**2 + y**2)
        r2 = np.sqrt((x-m)**2 + y**2)
        # Updating result
        result[0].append(r)
        result[1].append(v)

    return result

plt.scatter(0,0,s=200,c='purple')
plt.scatter(1,0,s=200,c='red')

masse_r = (6*10**24)/(2*10**30 + 6*10**24)
t0, tf = 0, 10
h = 0.01
r0 = (0.51-masse_r, math.sqrt(3)/2)
v0 = (-0.05,0.1)
L = explicit_euler_normalized(t0,tf,h,r0,v0,masse_r) 
print(masse_r)

#on affiche

tab_x = np.array([a for (a,b) in L[0]])
tab_y = np.array([b for (a,b) in L[0]])

# print(tab_x, tab_y)
# X = np.array([x[0], y[0]])
# Y = np.array([x[1], y[1]])
plt.plot(tab_x, tab_y)
plt.scatter(tab_x[0],tab_y[0],s=200,c='green')
plt.scatter(tab_x[-1],tab_y[-1],s=200,c='blue')

plt.show()
