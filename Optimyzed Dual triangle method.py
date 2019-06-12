import numpy as np
from numpy.linalg import norm
from scipy.optimize import approx_fprime
import numpy as np
from scipy.optimize import minimize

def Dichotom (target, left, right, eps = 0.0001):
    a_arg = left
    b_arg = right
    if a_arg > b_arg :
        print ("error")
        return - 1
    c_arg = (a_arg + b_arg)/2
    y = 0
    z = 0
    while True:
        y = (a_arg + c_arg)/2
        z = (c_arg + b_arg)/2
        if target(y) <= target(c_arg) :
            b_arg = c_arg
            c_arg = y
        else :
            if target(z) <= target(c_arg) :
                a_arg = c_arg
                c_arg = z
            else :
                a_arg = y
                b_arg = z
        if (b_arg - a_arg) <= eps :
            return c_arg

Y0     = 0
STEP   = 0.15
A = np.array([[1, 2], [2, 3]])
b = np.array([1, 2])
L = 36

def temp_alpha(target, alph, x, y):
    return target(alph*x + (1 - alph)*y)

def OptTriangleDual(target, x0, eps = 0.0001, grad = approx_fprime, step = lambda k: (k+1)/L, num_of_steps = 1000):
    index = 0
    y_new = Y0                  
    grad_sum = np.zeros((len(x0),))
    zk = [x0]
    xk = [x0] 
    alpha = lambda k: 2/(k + 1)
#    2pre_alpha = lambda k : Dichotom(lambda alph : target(alph*y_new + (1 - alph)* zk[k - 1]), 0, 1, eps)
#    pre_alpha = lambda alph: temp_alpha(target, alph, )
    while True :
        index += 1
        y_old = y_new
        x_new = alpha(index)*y_old + (1 - alpha(index))*zk[index - 1]
        xk.append(x_new)
        grad_sum += grad(x_new, target, eps)*step(index)
        y_new = (-1/2*grad_sum)
        alpha = lambda k: Dichotom(lambda alph : target(alph*y_new + (1 - alph)*zk[k - 1]), 0, 1, eps)
        zk.append(alpha(index)*(y_new) + (1 - alpha(index))*zk[index - 1])
     #   print(zk[index], index)
        if index >= num_of_steps :
            return zk
        
def target_func(x):
    eig, v = np.linalg.eig ((A.transpose()).dot(A)) 
    e = min (eig)
    res = (np.linalg.norm((A.dot (x) - b), ord=2))**2 - e*0.5*(np.linalg.norm(x, ord = 2))**2 #
    return res


def gradient (x, target, eps):
    eig, v = np.linalg.eig (A.transpose().dot(A))
    eig = min (eig)
    res = 2*A.transpose().dot(A.dot(x) - b) - x * eig
    return res
    
def USAGEOptTriangleDual(Anew = A, bnew = b, x0 = np.array([0.1, 0.2]), gen_grad = gradient):
    A = Anew
    b = bnew
    res = OptTriangleDual(target_func, x0, 0.00001, grad = gen_grad)
    return res
    
rest = minimize(target_func, np.array([0.1, 0.2]), method='BFGS')
print(rest.x)
res = USAGEOptTriangleDual()    
print(res[len(res) - 1])
