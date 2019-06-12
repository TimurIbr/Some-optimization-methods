import numpy as np
from numpy.linalg import norm
from scipy.optimize import approx_fprime
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

Y0     = 0
STEP   = 0.15
A = np.array([[1, 2], [2, 3]])
b = np.array([1, 2])
def TriangleDual(target, x0, eps, grad = approx_fprime, step = lambda k: (k+1)/L, alpha = lambda k : 2/(k+1), num_of_steps = 10000):
    index = 0
    y_new = Y0                  
    grad_sum = np.zeros((len(x0),))
    zk = [x0]
    xk = [x0] 
    while True :
        index += 1
        y_old = y_new
        x_new = alpha(index)*y_old + (1 - alpha(index))*zk[index - 1]
        xk.append(x_new)
        grad_sum += grad(x_new, target, 0.0000001)*step(index)
        y_new = (-1/2*grad_sum)
        zk.append(alpha(index)*(y_new) + (1 - alpha(index))*zk[index - 1])
        if index >= num_of_steps :
            return zk
        

def target_func(x):
    eig, v = np.linalg.eig (A.transpose() @ A) 
    e = min (eig)
    res = (np.linalg.norm((A @ x - b), ord=2))**2 - e*0.5*(np.linalg.norm(x, ord = 2))**2 #
    return res

def gradient (x):
    eig, v = np.linalg.eig (A.transpose() @ A)
    eig = min (eig)
    res = 2*A.transpose() @ (A @ x - b) - x * eig
    return res

def USAGETriangleDual(Anew = A, Bnew = B, x0 = np.array([0.1, 0.2]), gen_grad = gradient):
    A = Anew
    B = Bnew
    res = OptTriangleDual(target_func, x0, 0.00001, grad = gen_grad)
    return res


