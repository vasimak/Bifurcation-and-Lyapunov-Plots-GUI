from cmath import inf
import matplotlib.pyplot as plt
import math
import csv
import matplotlib as mpl
import time
from numba import njit
import numba as n
import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial
start_time = time.time()


x0=0.1
q0=0.1
begin_r=0
end_r=4
step=0.0001



r=np.arange(begin_r,end_r,step)
X1=[]
Y1=[]     


@njit
def le(q0, x0, r):
    N=1000
    lyapunov =0
    l1= 0
    x=x0
    q=q0
    for i in range(1,N):
        # logistic with extra parameters
        # x = r *(1 + x) * (1 + x) * (2 - x) + q  
        # lyapunov += np.log(np.abs(-3*r*(x**2-1)))      
        # logistic map
        x = x = r * x * (1  - x)
        lyapunov += np.log(np.abs(r - 2*r*x))
        # cheb map
        # x = math.cos(r*math.acos(x))
        # if math.sqrt(1-x**2)==0:
        #     r*math.sin(r*math.acos(x)) ==0
        # else:
        #     lyapunov += np.log(np.abs((r*math.sin(r*math.acos(x))) /
        #                    (math.sqrt(1-x**2))))
        # sine-sinh map
        # x = r*math.sin(math.pi*math.sinh(math.pi*math.sin(math.pi*x)))
        # lyapunov += np.log(np.abs(math.pi**3*r*math.cos(math.pi*x)*math.cosh(
        #     math.pi*math.sin(math.pi*x))*math.cos(math.pi*math.sinh(math.sin(math.pi*x)))))
        #renyi map
        # x = np.mod(r*x, 1)
        # lyapunov += np.log(np.abs(np.mod(r, 1)))
        # sine map
        # x = r*math.sin(math.pi*x)
        # lyapunov += np.log(np.abs(math.pi*r*math.cos(math.pi*x)))
        # cubic-logistic map
        # x = r*x*(1-x)*(2+x)
        # lyapunov += np.log(np.abs(-r*(3*x**2+2*x-2)))
        # cubic map
        # x = r*x*(1-x**2)
        # lyapunov += np.log(np.abs(r-3*r*x**2))
        # cheb with extra parameters
        # x = math.cos(r**q * math.acos(q*x))
        # if math.sqrt(1-(q**2*x**2))==0:
        #     q*r**q*math.sin(r**q * math.acos(q*x))==0
        # else:
        #     lyapunov += np.log(np.abs((q*r**q*math.sin(r**q *
        #                     math.acos(q*x)))/(math.sqrt(1-(q**2*x**2)))))
        # sine-sinh with extra parameters
        # x = r * math.sin(r * math.sinh(q * math.sin(2 * x)))
        # lyapunov += np.log(np.abs(2*q*r**2*math.cos(2*x)
        #                    * math.cosh(q*math.sin(2*x))))
        l1 = lyapunov/N
    return (l1) 
le1 = partial(le, q0, x0)
if __name__ == '__main__':
    with Pool(4) as p:
            for i,ch in enumerate(p.map(le1,r,chunksize=2500)) :
                # x1=np.ones(len(str((ch))))*r[i]
                X1.append(r[i])
                Y1.append(ch)
print("--- %s seconds ---" % (time.time() - start_time))

plt.style.use('dark_background')
plt.plot(X1,Y1, ".r", alpha=1, ms=1.2)
#plt.rcParams.update({"text.usetex": True})
plt.axhline(0)
plt.xlabel("k")
plt.ylabel("LE")
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920 / 40, 1080 / 40)
print("--- %s seconds ---" % (time.time() - start_time))

plt.show()
