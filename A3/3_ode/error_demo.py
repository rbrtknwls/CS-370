
# error_demo.py

import numpy as np
import matplotlib.pylab as plt
import time
#TkAgg for windows, osx or qt for mac
#get_ipython().magic(u'pylab auto')
plt.ion()


#%% Bigger Global Error

def soln(t, t0, y0):
    c = np.exp(-t0)*(y0+t0+1.)
    return c*np.exp(t) - t - 1.
def my_f(t,y):
    return t+y


#%% Smaller Global Error

def soln(t, t0, y0):
    c = np.exp(t0)*(y0-t0+1.)
    return c*np.exp(-t) + t - 1.
def my_f(t,y):
    return t-y


# In[27]:

t0, tf = 0, 2
tt = np.linspace(t0, tf, 20)
yy = soln(tt, t0, 1) #3.*np.exp(tt) - tt - 1.


# In[28]:

plt.figure(1)
plt.clf()
for a in np.arange(-5, 10, 0.2):
    plt.plot(tt, soln(tt, t0, a), color=[0.9,0.9,0.9])
plt.axis([0, 2, 0, 7])
plt.plot(tt,yy, linewidth=2);


# In[29]:

t0 = 0. # 0 or -0.8
y0 = 1.
tc, yc = t0, y0
h = 0.4
steps = np.int(np.ceil( (tf-h-t0)/h ))
yEuler = [y0]
tEuler = [t0]


# In[30]:

for n in range(steps):
    plt.plot(tt,soln(tt, tc, yc), color=[1,0.7,0.4])
    
    F = my_f(tc, yc)
    ynext = yc + h*F
    ynext_true = soln(tc+h, tc, yc)
    plt.plot([tc, tc+h], [yc, ynext], color=[0.2,0.2,0.1])
    
    time.sleep(1.)
    plt.plot([tc+h, tc+h], [ynext, ynext_true], 'r')
    plt.draw()
    
    yEuler.append(ynext)
    tEuler.append(tc+h)
    
    tc = tc + h
    yc = ynext
plt.plot(tEuler, yEuler, 'bo');



