import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize
import scipy as sp
import scipy.interpolate
from matplotlib import pylab as pl
pl.rc('text', usetex=True) # Support greek letters in plot legend

    
from crackclosuresim2 import inverse_closure,solve_normalstress
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

sigmaext_max = 20e6 # external tensile load, Pa

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

#crack_model = ModeI_throughcrack_CODformula(Eeff)

crack_model = Tada_ModeI_CircularCrack_along_midline(E,nu)


# Closure state (function of position; positive compression)

observed_reff = np.array([ 0.0,  1e-3, 1.5e-3, 2e-3  ],dtype='d')
observed_seff = np.array([ 10e6, 15e6, 30e6, 150e6  ],dtype='d')

sigma_closure = inverse_closure(observed_reff,
                                observed_seff,
                                x,x_bnd,dx,a,sigma_yield,
                                crack_model)


# Forward cross-check of closure
pl.figure()
pl.plot(x*1e3,sigma_closure,'-',
        observed_reff*1e3,observed_seff,'x')

for observcnt in range(len(observed_reff)):        
    (effective_length, sigma, tensile_displ) = solve_normalstress(x,x_bnd,sigma_closure,dx,observed_seff[observcnt],a,sigma_yield,crack_model)
    pl.plot(effective_length*1e3,observed_seff[observcnt],'o')
    #pl.plot(x*1e3,tensile_displ*1e15,'-')
    pass


#sigma_closure = 80e6/cos(x/a) -70e6 # Pa
#sigma_closure[x > a]=0.0

use_crackclosuresim=False
if use_crackclosuresim:
    from scipy.interpolate import splrep
    import crackclosuresim.crack_utils_1D as cu1
    ccs_x = x[10::10] # use every 10th point
    
    pass

if use_crackclosuresim:
    stress_field_spl = splrep(x[x < a],-sigma_closure[x < a],task=-1,t=[.5e-3,1e-3,1.5e-3])
    
    ccs_aeff=cu1.find_length(sigmaext_max,stress_field_spl,a,cu1.weightfun_through,(specimen_width,))
    
    ccs_uyy = np.zeros(ccs_x.shape,dtype='d')
    for xcnt in range(ccs_x.shape[0]):
        print("xcnt=%d" % (xcnt))
        ccs_uyy[xcnt]=cu1.uyy(ccs_x[xcnt],a,sigmaext_max,stress_field_spl,cu1.weightfun_through,(specimen_width,),E,nu,configuration="PLANE_STRESS")
        pass
    
    ccs_closurestate = cu1.effective_stresses_full(ccs_x,a,sigmaext_max,stress_field_spl,cu1.weightfun_through,(specimen_width,))
    
    pass

(effective_length, sigma, tensile_displ) = solve_normalstress(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,verbose=True,diag_plots=True)

(fig,ax1) = pl.subplots()
legax=[]
legstr=[]
(pl1,pl2)=ax1.plot(x*1e3,-sigma_closure/1e6,'-',
                       x*1e3,sigma/1e6,'-')
legax.extend([pl1,pl2])
legstr.extend(['Closure stress','Contact stress'])
if (use_crackclosuresim):
    (pl4,)=ax1.plot(ccs_closurestate[0]*1e3,ccs_closurestate[1]/1e6,'-')
    legax.append(pl4)
    legstr.append('closure (crackclosuresim)')
    pass
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Stress (MPa)')


ax2=ax1.twinx()
(pl5,)=ax2.plot(x*1e3,tensile_displ*1e9,'-k')
legax.append(pl5)
legstr.append('uyy (new)')
if use_crackclosuresim:
    (pl6,)=ax2.plot(ccs_x*1e3,ccs_uyy*1e9,':k')
    legax.append(pl6)
    legstr.append('uyy (crackclosuresim)')
    pass
align_yaxis(ax1,0,ax2,0)
ax2.set_ylabel('Tensile displacement (nm)')
pl.legend(legax,legstr)
#fig.tight_layout()
pl.title('Closed crack')
pl.savefig('/tmp/tensile_peel_closedcrack.png',dpi=300)


# Alternate closure state (function of position; positive compression)
sigma_closure2 = 80e6/cos(x/a) -20e6 # Pa
sigma_closure2[x > a]=0.0

if use_crackclosuresim:
    stress_field_spl2 = splrep(x[x < a],-sigma_closure2[x < a],task=-1,t=[.5e-3,1e-3,1.5e-3])
    
    ccs_aeff2=cu1.find_length(sigmaext_max,stress_field_spl2,a,cu1.weightfun_through,(specimen_width,))
    
    ccs_uyy2 = np.zeros(ccs_x.shape,dtype='d')
    for xcnt in range(ccs_x.shape[0]):
        print("xcnt=%d" % (xcnt))
        ccs_uyy2[xcnt]=cu1.uyy(ccs_x[xcnt],a,sigmaext_max,stress_field_spl2,cu1.weightfun_through,(specimen_width,),E,nu,configuration="PLANE_STRESS")
        pass
    
    ccs_closurestate2 = cu1.effective_stresses_full(ccs_x,a,sigmaext_max,stress_field_spl2,cu1.weightfun_through,(specimen_width,))
    pass



(effective_length2, sigma2, tensile_displ2) = solve_normalstress(x,x_bnd,sigma_closure2,dx,sigmaext_max,a,sigma_yield,crack_model,verbose=True)

(fig2,ax21) = pl.subplots()
legax=[]
legstr=[]
(pl21,pl22)=ax21.plot(x*1e3,-sigma_closure2/1e6,'-',
                           x*1e3,sigma2/1e6,'-')
legax.extend([pl21,pl22])
legstr.extend(['Closure stress','Contact stress'])
if (use_crackclosuresim):
    (pl24,)=ax21.plot(ccs_closurestate2[0]*1e3,ccs_closurestate2[1]/1e6,'-')
    legax.append(pl24)
    legstr.append('closure (crackclosuresim)')
    pass
ax21.set_xlabel('Position (mm)')
ax21.set_ylabel('Stress (MPa)')


ax22=ax21.twinx()
(pl25,)=ax22.plot(x*1e3,tensile_displ2*1e9,'-k')
legax.append(pl25)
legstr.append('uyy (new)')
if use_crackclosuresim:
    (pl26,)=ax22.plot(ccs_x*1e3,ccs_uyy2*1e9,':k')
    legax.append(pl26)
    legstr.append('uyy (crackclosuresim)')
    pass
align_yaxis(ax21,0,ax22,0)
ax22.set_ylabel('Tensile displacement (nm)')
pl.legend(legax,legstr)
#fig.tight_layout()
pl.title('Tight crack')
pl.savefig('/tmp/tensile_peel_tightcrack.png',dpi=300)


# Alternate closure state (function of position; positive compression)
sigma_closure3 = 80e6/cos(x/a) -79e6 # Pa
sigma_closure3[x > a]=0.0

if use_crackclosuresim:
    stress_field_spl3 = splrep(x[x < a],-sigma_closure3[x < a],task=-1,t=[.5e-3,1e-3,1.5e-3])
    
    ccs_aeff3=cu1.find_length(sigmaext_max,stress_field_spl3,a,cu1.weightfun_through,(specimen_width,))
    
    ccs_uyy3 = np.zeros(ccs_x.shape,dtype='d')
    for xcnt in range(ccs_x.shape[0]):
        print("xcnt=%d" % (xcnt))
        ccs_uyy3[xcnt]=cu1.uyy(ccs_x[xcnt],a,sigmaext_max,stress_field_spl3,cu1.weightfun_through,(specimen_width,),E,nu,configuration="PLANE_STRESS")
        pass
    
    ccs_closurestate3 = cu1.effective_stresses_full(ccs_x,a,sigmaext_max,stress_field_spl3,cu1.weightfun_through,(specimen_width,))
    
    pass


(effective_length3, sigma3, tensile_displ3) = solve_normalstress(x,x_bnd,sigma_closure3,dx,sigmaext_max,a,sigma_yield,crack_model,verbose=True)

(fig3,ax31) = pl.subplots()
legax=[]
legstr=[]
(pl31,pl32)=ax31.plot(x*1e3,-sigma_closure3/1e6,'-',
                           x*1e3,sigma3/1e6,'-')
legax.extend([pl31,pl32])
legstr.extend(['Closure stress','Contact stress'])
if (use_crackclosuresim):
    (pl34,)=ax31.plot(ccs_closurestate3[0]*1e3,ccs_closurestate3[1]/1e6,'-')
    legax.append(pl34)
    legstr.append('closure (crackclosuresim)')
    pass
ax31.set_xlabel('Position (mm)')
ax31.set_ylabel('Stress (MPa)')


ax32=ax31.twinx()
(pl35,)=ax32.plot(x*1e3,tensile_displ3*1e9,'-k')
legax.append(pl35)
legstr.append('uyy (new)')
if use_crackclosuresim:
    (pl36,)=ax32.plot(ccs_x*1e3,ccs_uyy3*1e9,':k')
    legax.append(pl36)
    legstr.append('uyy (crackclosuresim)')
    pass
align_yaxis(ax31,0,ax32,0)
ax32.set_ylabel('Tensile displacement (nm)')
pl.legend(legax,legstr)
#fig.tight_layout()
pl.title('Partially open crack')
pl.savefig('/tmp/tensile_peel_opencrack.png',dpi=300)




pl.show()

    
    
