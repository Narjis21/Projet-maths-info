import numpy as np

def find_seed(g, n=0, p=1, x=0, c=0,eps=2**(-26)):
    if (c>g(x,1) and c>g(x,0)) or (c<g(x,1) and c<g(x,0)):
        return None
    def f(t): 
        return g(0,t)-c
    def dichotomie(f):
        a=n
        b=p
        while (b-a)/2 > eps and f(a)*f(b) <= 0:
            m=(a+b)/2
            if f(a)*f(m)<=0: 
                b=m
            else: 
                a=m
        return (a+b)/2
    return dichotomie(f) 

def simple_contour(f, c=0, delta=0.01):
    
    

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


