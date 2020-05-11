import sys
import os
import os.path
import tempfile

import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize

from matplotlib import pyplot as pl
pl.rc('text', usetex=True) # Support greek letters in plot legend


from crackclosuresim2 import solve_shearstress
from crackclosuresim2 import ModeII_throughcrack_CSDformula

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


#####INPUT VALUES
E = 200e9    #Plane stress Modulus of Elasticity
sigma_yield = 400e6
tau_yield = sigma_yield/2.0 # limits stress concentration around singularity
nu = 0.33    #Poisson's Ratio

tauext_max = 20e6 # external shear load, Pa

a=2.0e-3  # half-crack length (m)
xmax = 5e-3 # as far out in x as we are calculating (m)
xsteps = 200


# x_bnd represents x coordinates of the boundaries of
# each mesh element 
x_bnd=np.linspace(0,xmax,xsteps,dtype='d')
dx=x_bnd[1]-x_bnd[0]
x = (x_bnd[1:]+x_bnd[:-1])/2.0  # x represents x coordinates of the centers of each mesh element


#Friction coefficient
mu = 0.33

crack_model=ModeII_throughcrack_CSDformula(E,nu,Symmetric_CSD=True)

    
# Closure state (function of position; positive compression)
sigma_closure = 80e6/cos(x/a) -70e6 # Pa
sigma_closure[x > a]=0.0


(effective_length, tau, shear_displ) = solve_shearstress(x,x_bnd,sigma_closure,dx,tauext_max,a,mu,tau_yield,crack_model,verbose=True)

(fig,ax1) = pl.subplots()
(pl1,pl2,pl3)=ax1.plot(x*1e3,sigma_closure/1e6,'-',
                       x*1e3,tau/1e6,'-',
                    x*1e3,(tau-mu*(sigma_closure*(sigma_closure > 0)))/1e6,'-')
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Stress (MPa)')


ax2=ax1.twinx()
(pl4,)=ax2.plot(x*1e3,shear_displ*1e9,'-k')
align_yaxis(ax1,0,ax2,0)
ax2.set_ylabel('Shear displacement (nm)')
pl.legend((pl1,pl2,pl3,pl4),('Closure stress','Shear stress','$ \\tau - \\mu \\sigma_{\\mbox{\\tiny closure}}$','Shear displacement'))
#fig.tight_layout()
pl.title('Closed crack')
pl.savefig(os.path.join(tempfile.gettempdir(),'shear_stickslip_closedcrack.png'),dpi=300)


# Alternate closure state (function of position; positive compression)
sigma_closure2 = 80e6/cos(x/a) -20e6 # Pa
sigma_closure2[x > a]=0.0



(effective_length2, tau2, shear_displ2) = solve_shearstress(x,x_bnd,sigma_closure2,dx,tauext_max,a,mu,tau_yield,crack_model,verbose=True)

(fig2,ax21) = pl.subplots()
(pl21,pl22,pl23)=ax21.plot(x*1e3,sigma_closure2/1e6,'-',
                           x*1e3,tau2/1e6,'-',
                           x*1e3,(tau2-mu*(sigma_closure2*(sigma_closure2 > 0)))/1e6,'-')
ax21.set_xlabel('Position (mm)')
ax21.set_ylabel('Stress (MPa)')


ax22=ax21.twinx()
(pl24,)=ax22.plot(x*1e3,shear_displ2*1e9,'-k')
align_yaxis(ax21,0,ax22,0)
ax22.set_ylabel('Shear displacement (nm)')
pl.legend((pl21,pl22,pl23,pl24),('Closure stress','Shear stress','$ \\tau - \\mu \\sigma_{\\mbox{\\tiny closure}}$','Shear displacement'))
#fig.tight_layout()
pl.title('Tight crack')
pl.savefig(os.path.join(tempfile.gettempdir(),'shear_stickslip_tightcrack.png'),dpi=300)


# Alternate closure state (function of position; positive compression)
sigma_closure3 = 80e6/cos(x/a) -90e6 # Pa
sigma_closure3[x > a]=0.0


    
(effective_length3, tau3, shear_displ3) = solve_shearstress(x,x_bnd,sigma_closure3,dx,tauext_max,a,mu,tau_yield,crack_model,verbose=True)

(fig3,ax31) = pl.subplots()
(pl31,pl32,pl33)=ax31.plot(x*1e3,sigma_closure3/1e6,'-',
                           x*1e3,tau3/1e6,'-',
                        x*1e3,(tau3-mu*(sigma_closure3*(sigma_closure3 > 0)))/1e6,'-')
ax31.set_xlabel('Position (mm)')
ax31.set_ylabel('Stress (MPa)')


ax32=ax31.twinx()
(pl34,)=ax32.plot(x*1e3,shear_displ3*1e9,'-k')
align_yaxis(ax31,0,ax32,0)
ax32.set_ylabel('Shear displacement (nm)')
pl.legend((pl31,pl32,pl33,pl34),('Closure stress','Shear stress','$ \\tau - \\mu \\sigma_{\\mbox{\\tiny closure}}$','Shear displacement'))
#fig.tight_layout()
pl.title('Partially open crack')
pl.savefig(os.path.join(tempfile.gettempdir(),'/tmp/shear_stickslip_opencrack.png'),dpi=300)



    
pl.show()

    
    
