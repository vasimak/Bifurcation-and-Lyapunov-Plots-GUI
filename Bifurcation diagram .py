# Bifurcation diagram


from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import matplotlib as mpl
import time
from numba import njit
import multiprocessing 
from multiprocessing import Pool
multiprocessing.cpu_count()
from functools import reduce
import PyQt5
from functools import partial



start_time = time.time()



mpl.use('qtagg')
mpl.rcParams['path.simplify_threshold'] = 1.0


x0=0.1
q0=-0.1
begin_r=0
end_r=1
step=0.000001





# the path where the plots are saved. You can change it with yours.


#filename = "./images/sine q=" + str(q) + "/g" + str(g) + ".jpg"
# filename="./Latex/LateX images/graphs q21/g" + str(g) +".jpg"
# filename="./Latex/LateX images/cheb q="+ str(q)+"/g" + str(g) +".jpg"


# the path where the the data of parameter k and x are saved


#file_path = "./data_folder/data q=" + str(q) + " x=" + str(x[1]) + ".txt"

r=np.arange(begin_r,end_r,step)
X=[]
Y=[]
      
@njit
def bif(q0, x0, r):
    N=1000  
    x = np.zeros(len(range(0, N)))
    x[0]=x0
    q=q0
    for i in range(1,N):
        x[i]= r *(1 + x[i - 1]) * (1 + x[i- 1]) * (2 - x[i - 1]) + q  #logistic with extra parameter
        # x[i] = r * x[i-1] * (1 - x[i-1]) #logistic map
        # x[i] = math.cos(r*math.acos(x[i-1])) #cheb map
        # x[i] = r*math.sin(math.pi*math.sinh(math.pi*math.sin(math.pi*x[i-1]))) #sine-sinh map
        # x[i] = np.mod(r*x[i-1], 1) #renyi map
        # x[i] = r*math.sin(math.pi*x[i-1]) #sine map
        # x[i] = r*x[i-1]*(1-x[i-1])*(2+x[i-1]) #cubic-logistic map
        # x[i] = r*x[i-1]*(1-x[i-1]**2) #cubic map
        # x[i] = math.cos(r**q * math.acos(q*x[i - 1])) #cheb with extra parameter
        # x[i] = r * math.sin(r * math.sinh(q * math.sin(2 * x[i - 1]))) #sine-sinh with extra parameter
    return (x[-130:])

bif1 = partial(bif, q0, x0)
if __name__ == '__main__':
    # create and configure the process pool
    with Pool(4) as p:
        
        for i,ch in enumerate(p.map(bif1,r,chunksize=2500)) :
            x1=np.ones(len(ch))*r[i]
            X.append(x1)
            Y.append(ch)
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.style.use('dark_background')      
    plt.plot(X,Y, ".w", alpha=1, ms=1.2)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(1920 / 40, 1080 / 40)
    # print("--- %s seconds ---" % (time.time() - start_time))
plt.show()

# with open(file_path, "w+", encoding="utf-8", newline="") as f:
#     for i in range(10000):
#         for j in range(130):
#             if np.any(M[i, j] == np.inf) or np.any(M[i, j] == -np.inf):
#                 break
#             else:
#                 f.writelines([f"{k[i]}", f"{M[i,j]}\n"])

# f.close()
# print("--- %s seconds ---" % (time.time() - start_time))
