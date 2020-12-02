import sys
import os
import os.path
import tempfile
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize
import scipy as sp
import scipy.interpolate
from matplotlib import pylab as pl
pl.rc('text', usetex=True) # Support greek letters in plot legend

    
from crackclosuresim2 import inverse_closure,inverse_closure2,solve_normalstress
from crackclosuresim2 import crackopening_from_tensile_closure
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline

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
Eeff=E
sigma_yield = 400e6
tau_yield = sigma_yield/2.0 # limits stress concentration around singularity
nu = 0.33    #Poisson's Ratio
specimen_width=25.4e-3

sigmaext_max1 = -50e6 # external tensile load, Pa

a=2.0e-3  # half-crack length (m)
xmax = 5e-3 # as far out in x as we are calculating (m)
xsteps = 200

# x_bnd represents x coordinates of the boundaries of
# each mesh element 
x_bnd=np.linspace(0,xmax,xsteps,dtype='d')
dx=x_bnd[1]-x_bnd[0]
x = (x_bnd[1:]+x_bnd[:-1])/2.0  # x represents x coordinates of the centers of each mesh element

    
##Friction coefficient
#mu = 0.33

weightfun_epsx = dx/8.0

#crack_model = Glinka_ModeI_ThroughCrack(Eeff,x,specimen_width,weightfun_epsx)
#crack_model = ModeI_throughcrack_weightfun(Eeff,x,weightfun_epsx)

#crack_model = ModeI_throughcrack_CODformula(Eeff,Symmetric_COD=True)

crack_model = Tada_ModeI_CircularCrack_along_midline(E,nu)


# Closure state (function of position; positive compression)

observed_reff = np.array([ 0.5e-3,  1e-3, 1.5e-3, 2e-3  ],dtype='d')
observed_seff = np.array([ .00e6, 15e6, 30e6, 150e6  ],dtype='d')

(sigma_closure,
 interp_diagnostic_plot_figure) = inverse_closure2(observed_reff,
                                                   observed_seff,
                                                   x,x_bnd,dx,a,sigma_yield,
                                                   crack_model,
                                                   extrapolate_inward=True,
                                                   extrapolate_outward=True,
                                                   zero_beyond_tip=True,
                                                   interpolate_input=True,
                                                   interpolation_diagnostic_plot=True)

sigma_closure[x > a]=0.0

crack_opening = crackopening_from_tensile_closure(x,x_bnd,sigma_closure,dx,a,sigma_yield,crack_model)


# Forward cross-check of closure
pl.figure()
pl.plot(x*1e3,sigma_closure,'-',
        observed_reff*1e3,observed_seff,'x')

for observcnt in range(len(observed_reff)):        
    (effective_length, sigma, tensile_displ,dsigmaext_dxt) = solve_normalstress(x,x_bnd,sigma_closure,dx,observed_seff[observcnt],a,sigma_yield,crack_model)
    pl.plot(effective_length*1e3,observed_seff[observcnt],'o')
    #pl.plot(x*1e3,tensile_displ*1e15,'-')
    pass

    
#sigma_closure = 80e6/cos(x/a) -70e6 # Pa
#sigma_closure[x > a]=0.0


if True:
    (effective_length, sigma, tensile_displ,dsigmaext_dxt) = solve_normalstress(x,x_bnd,sigma_closure,dx,sigmaext_max1,a,sigma_yield,crack_model,verbose=True,diag_plots=True)
    
    (fig,ax1) = pl.subplots()
    legax=[]
    legstr=[]
    (pl1,pl2)=ax1.plot(x*1e3,-sigma_closure/1e6,'-',
                           x*1e3,sigma/1e6,'-')

    legax.extend([pl1,pl2])
    legstr.extend(['Closure stress','Tensile stress'])
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Stress (MPa)')


    ax2=ax1.twinx()
    (pl5,pl6,)=ax2.plot(x*1e3,crack_opening*1e9,'-k',
                        x*1e3,(crack_opening+tensile_displ)*1e9,'-')
    legax.extend([pl5,pl6])
    legstr.extend(['initial opening (nm)','crack opening (nm)'] )
    
    align_yaxis(ax1,0,ax2,0)
    ax2.set_ylabel('Tensile displacement (nm)')
    pl.legend(legax,legstr)
    #fig.tight_layout()
    pl.title('Partially closed crack')
    pl.savefig(os.path.join(tempfile.gettempdir(),'compressive_peel_initial.png'),dpi=300)
    pass

sigmaext_max2=-50e6

(effective_length, sigma, tensile_displ,dsigmaext_dxt) = solve_normalstress(x,x_bnd,sigma_closure,dx,sigmaext_max2,a,sigma_yield,crack_model,verbose=True,diag_plots=True)
    
(fig,ax1) = pl.subplots()
legax=[]
legstr=[]
(pl1,pl2)=ax1.plot(x*1e3,-sigma_closure/1e6,'-',
                       x*1e3,sigma/1e6,'-')
legax.extend([pl1,pl2])
legstr.extend(['Closure stress','Contact stress'])
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Stress (MPa)')


ax2=ax1.twinx()
(pl5,)=ax2.plot(x*1e3,(crack_opening+tensile_displ)*1e9,'-k')
legax.append(pl5)
legstr.append('uyy (nm)')

align_yaxis(ax1,0,ax2,0)
ax2.set_ylabel('Crack opening (nm)')
pl.legend(legax,legstr)
#fig.tight_layout()
pl.title('Partially closed crack')
pl.savefig(os.path.join(tempfile.gettempdir(),'compressive_peel_pressedclosed.png'),dpi=300)


sigmaext_max3=-150e6

(effective_length, sigma, tensile_displ,dsigmaext_dxt) = solve_normalstress(x,x_bnd,sigma_closure,dx,sigmaext_max3,a,sigma_yield,crack_model,verbose=True,diag_plots=True)
    
(fig,ax1) = pl.subplots()
legax=[]
legstr=[]
(pl1,pl2)=ax1.plot(x*1e3,-sigma_closure/1e6,'-',
                       x*1e3,sigma/1e6,'-')
legax.extend([pl1,pl2])
legstr.extend(['Closure stress','Contact stress'])
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Stress (MPa)')


ax2=ax1.twinx()
(pl5,)=ax2.plot(x*1e3,(crack_opening+tensile_displ)*1e9,'-k')
legax.append(pl5)
legstr.append('uyy (nm)')

align_yaxis(ax1,0,ax2,0)
ax2.set_ylabel('Crack opening (nm)')
pl.legend(legax,legstr)
#fig.tight_layout()
pl.title('Partially closed crack')
pl.savefig(os.path.join(tempfile.gettempdir(),'compressive_peel_pressedclosedharder.png'),dpi=300)


pl.show()
