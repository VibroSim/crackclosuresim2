import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splrep,splev

from crackclosuresim2.fabrikant import K_nondim,u_nondim,u,K

#Penny-shaped crack revisited: Closed-form solutions; V. I. Fabrikant
#https://www.tandfonline.com/loi/tpha20

# align_yaxis from https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)
    pass

from matplotlib import pyplot as pl

E=208e9
nu=0.33
a=5e-3
tauext=100e6

create_surrogate_entries=False
use_surrogate=True

if create_surrogate_entries:
    xnorm = np.linspace(0.0,1.0,30)
    u_nd_eval = np.array([ u_nondim(xval,0.0,nu) for xval in xnorm])
    
    use = ~np.isnan(u_nd_eval)

    (t,c,k)=splrep(xnorm[use],u_nd_eval[use])
    
    print("u_surrogate entry:")
    print("(0.0,%s): (%s,%s,%s)," % (repr(nu),array_repr(t),array_repr(c),repr(k)))
    
    K_nd_val = K_nondim(np.pi,nu)

    print("K_surrogate entry:")
    print("(np.pi,%s): %s," % (repr(nu),repr(K_nd_val)))
    pass


x=np.linspace(0,a,30)
u_eval = np.array([ u(xval,0.0,a,tauext,E,nu,use_surrogate=use_surrogate) for xval in x])


K_eval = K(np.pi,a,tauext,nu,use_surrogate=use_surrogate) # NOTE: K comes out negative if we use 0 for phi (!)



pl.figure()
pl.plot(x*1e3,u_eval*1e6,'-')
#x*1e3,splev(x,(t,c,k))*1e6,'-')
pl.title('%f mm crack' % (a*1e3))
pl.xlabel('x (mm)')
pl.ylabel('u (um)')
pl.show()
