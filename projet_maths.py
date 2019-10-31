import autograd
from autograd import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, exp

def find_seed(g, x=0, c=0, n=0,  pas=1, eps=2**(-26)):
    if (c>g(x,n+pas) and c>g(x,n)) or (c<g(x,n+pas) and c<g(x,n)):
        return None
    def f(t): 
        return g(x,t)-c
    def dichotomie(f):
        a=n
        b=n+pas
        while (b-a)/2 > eps and f(a)*f(b) <= 0:
            m=(a+b)/2
            if f(a)*f(m)<=0: 
                b=m
            else: 
                a=m
        return (a+b)/2
    return dichotomie(f) 

def simple_contour_hachures(f, c=0, plot=False, delta=0.01,xo=0, y_borne=0, pas=1):
    X = [xo]
    x=X[0]
    Y = [find_seed(f, X[0], c, y_borne, pas)]
    while x<=xo+pas:
        x = x+delta
        y = find_seed(f, X[-1], c, y_borne, pas)
        X.append(x)
        Y.append(y)
    if plot==True:
        plt.figure(1, figsize=(9,9))
        plt.plot(X, Y)
        plt.show()
    return X, Y

def contour(f, c=0, xc=[0.,1.], yc=[0.,1.], pas=1, delta=0.01):
    X=[]
    Y=[]
    """segx = np.linspace(xc[0], xc[-1], 100)
    segy = np.linspace(yc[0], yc[-1], 100)
    for i in range(99):
        for j in range(99):
            X = X + simple_contour_hachures(f, c, False, delta, segx[i], segy[j])[0]
            Y = Y + simple_contour_hachures(f, c, False, delta, segx[i], segy[j])[1] 
    """
    cx = 0
    while int(xc[0])+cx<int(xc[-1]):
        cy = 0
        while int(yc[0])+cy<int(yc[-1]):
            X = X + simple_contour_hachures(f, c, False, delta, int(xc[0])+cx, int(yc[0])+cy, pas)[0]
            Y = Y + simple_contour_hachures(f, c, False, delta, int(xc[0])+cx, int(yc[0])+cy, pas)[1]            
            cy+=pas
        cx+=pas
    plt.figure(1, figsize=(9,9))
    plt.plot(X, Y, 'ro')
    plt.show()


def ortho_grad(f, x, y):
    g = autograd.grad
    if sqrt(g(f,0)(x,y)**2+g(f,1)(x,y)**2)==0:
        return np.r_[-g(f,1)(x,y), g(f,0)(x,y)]
    tang_1 = np.r_[-g(f,1)(x,y), g(f,0)(x,y)]/sqrt(g(f,1)(x,y)**2 + g(f,0)(x,y)**2)
    tang_2 = np.r_[g(f,1)(x,y), -g(f,0)(x,y)]/sqrt(g(f,1)(x,y)**2 + g(f,0)(x,y)**2)
    return tang_1, tang_2


def grad(f,x,y):
    g = autograd.grad
    if sqrt(g(f,1)(x,y)**2 + g(f,0)(x,y)**2)==0:
        return np.r_[-g(f,1)(x,y), g(f,0)(x,y)]
    return np.r_[g(f,0)(x,y), g(f,1)(x,y)]/sqrt(g(f,0)(x,y)**2 + g(f,1)(x,y)**2)
    
def f(x,y):
    return x*x*x + y*y
    return 5*cos(x)*cos(y)*exp(-0.04*(x*x+y*y))

"""def Newton_grad(f, c, xo, yo, delta):
    def F(X):
        x, y = X
        return X - f(x, y)*grad(f,x,y)
    X = np.array([xo, yo])
    liste_X = [X]
    while norme(F(liste_X[-1])-liste_X[-1])>10**(-5):
        liste_X.append(F(liste_X[-1]))
    return liste_X[-1]"""

def Newton_jacob(f, c, a, b, xo, yo, delta):
    def h(x,y):
        return np.array([f(x,y)-c,(x-a)**2 + (y-b)**2 - delta**2 ])
    def F(X):
        x, y = X[0], X[1]
        return X - np.dot(np.linalg.inv(jacob(h, x, y)),h(x,y))
    X = np.array([xo, yo])
    liste_X = [X]
    while norme(F(liste_X[-1])-liste_X[-1])>10**(-5):
        liste_X.append(F(liste_X[-1]))
    return liste_X[-1]    
   
def simple_contour_grad(f, c=0, delta=0.01):
    X = [0.]
    Y = [find_seed(f, X[0], c)]
    while 0<=X[-1]<1 and 0<=Y[-1]<1:
        xo = X[-1]+delta*ortho_grad(f, X[-1], Y[-1])[0]
        yo = Y[-1]+delta*ortho_grad(f, X[-1], Y[-1])[1]
        print('bonjo')
        X.append(Newton_jacob(f,c,X[-1], Y[-1], xo,yo,delta)[0])
        Y.append(Newton_jacob(f,c,X[-1], Y[-1], xo,yo,delta)[1])
        print('hello')
    plt.figure(1,figsize=(9,9))
    plt.plot(X, Y)
    plt.show()
    
def norme(X):
    x,y = X
    return sqrt(x**2 + y**2)

def jacob(h,x,y):
    j=autograd.jacobian
    return np.c_[j(h,0)(x,y), j(h,1)(x,y)]



"""def f_cercle(f,p,delta,theta):
    xo,yo=p
    x=delta*cos(theta)+xo
    y=delta*sin(theta)+yo
    return [f(x,y),x,y]
    

def h(f, c, x, y, xo, yo, delta):
    return np.array([f(x,y)-c,(x-xo)**2 + (y-yo)**2 - delta**2 ])"""

"""def simple_contour_jacob(f, c=0, delta=0.01):
    X = [0.]
    Y = [find_seed(f, X[0], c)]
    x=X[0]
    y=Y[0]
    for k in range(1000):
        x = X[-1]+delta*ortho_grad(f, X[-1], Y[-1])[0]
        y = Y[-1]+delta*ortho_grad(f, X[-1], Y[-1])[1]
        xn,yn = Newton(f, c, x, y, delta)
        X.append(xn)
        Y.append(yn)
    plt.figure(1,figsize=(9,9))
    plt.plot(X, Y)
    plt.show()"""
    
    
"""np.linalg.inv(A)"""

"""def simple_contour_cercle(f, c=0, delta=0.01):
    X = [0]
    Y = [find_seed(f, X[0], c)]
    x0, y0 = X[0], Y[0]
    for i in range(1):
        k=0
        while abs(f_cercle(f,(x0,y0), delta, k*np.pi/180)[0]-c)>10**(-1):
            k+=1
        X.append(f_cercle(f,(x0,y0), delta, k*np.pi/180)[1])
        Y.append(f_cercle(f,(x0,y0), delta, k*np.pi/180)[2])
    plt.figure(1,figsize=(9,9))
    plt.plot(X, Y)
    plt.show()"""
"""T = []
    xs=[]
    ys=[]
    while ([xc[-1],xc[-2]],[yc[-1],yc[-2]]) not in T:
        xs = xs + simple_contour_hachures(f,c,delta)[0]
        ys = ys + simple_contour_hachures(f,c,delta)[1]
        T.append(([xc[-1],xc[-2]],[yc[-1],yc[-2]]))
        if abs(xs[-1]-xc[-1])<=delta:
            xc.append(xc[-1]+1.)
        elif abs(xs[-1]-xc[-2])<=delta:
            xc[-2]-=1
            xc[-1]-=1
        elif abs(ys[-1]-yc[-1])<=delta:
            xc.append(yc[-1]+1.)
        elif abs(ys[-1]-yc[-2])<=delta:
            yc[-2]-=1
            yc[-1]-=1"""
      

    
    
   
    
    
    
    
    
    
"""def simple_contour(f, c=0, delta=0.01):
        t0 = find_seed(f, 0, delta)
        T=[[]]
        k=0
        while t0 == None:
            t0 = find_seed(f, k*delta, (k+1)*delta)
            if (k+1)*delta>1:
                return [[],[]]
            k+=1
        l=0
        T[l].append(t0)
        e=10**(-5)
        if find_seed(f, t0+delta-e, t0+delta+e) == None:
            l=l+1"""


