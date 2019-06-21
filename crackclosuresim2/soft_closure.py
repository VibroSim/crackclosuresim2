import copy
import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize
import scipy as sp

from matplotlib import pylab as pl
#pl.rc('text', usetex=True) # Support greek letters in plot legend

from crackclosuresim2 import inverse_closure,crackopening_from_tensile_closure
from crackclosuresim2 import ModeI_throughcrack_CODformula
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline



class sc_params(object):
    # soft closure parameters
    Hm = None # H*m
    E = None # Modulus or effective modulus
    xmax = None
    xsteps = None
    a_input = None
    a = None
    x_bnd = None
    dx = None 
    x = None
    afull_idx = None
    fine_refinement = None
    dx_fine = None
    xbnd_fine = None
    x_fine = None
    afull_idx_fine = None
    sigma_closure = None
    crack_initial_opening = None

    def __init__(self,**kwargs):
        for argname in kwargs:
            if not hasattr(self,argname):
                raise ValueError("Invalid parameter: %s" % (argname))
            setattr(self,argname,kwargs[argname])
            pass
        pass

    def setcrackstate(self,sigma_closure,crack_initial_opening):
        self.sigma_closure=sigma_closure
        self.crack_initial_opening=crack_initial_opening
        pass
    
    
    @classmethod
    def fromcrackgeom(cls,E,xmax,xsteps,a_input,fine_refinement,Hm):
        # x_bnd represents x coordinates of the boundaries of
        # each mesh element 
        x_bnd=np.linspace(0,xmax,xsteps,dtype='d')
        dx=x_bnd[1]-x_bnd[0]
        x = (x_bnd[1:]+x_bnd[:-1])/2.0  # x represents x coordinates of the centers of each mesh element
        
        afull_idx=np.argmin(np.abs(a_input-x_bnd))
        a = x_bnd[afull_idx]
        dx_fine = dx/float(fine_refinement)
        xbnd_fine = np.arange(xsteps*fine_refinement-(fine_refinement-1),dtype='d')*dx_fine
        x_fine=(xbnd_fine[1:]+xbnd_fine[:-1])/2.0
        afull_idx_fine = afull_idx*fine_refinement  # xbnd_fine[afull_idx_fine]==a within roundoff error


        return cls(E=E,
                   xmax=xmax,
                   xsteps=xsteps,
                   a_input=a_input,
                   fine_refinement=fine_refinement,
                   Hm=Hm,
                   x_bnd=x_bnd,
                   dx=dx,
                   x=x,
                   afull_idx=afull_idx,
                   a=a,
                   dx_fine=dx_fine,
                   xbnd_fine=xbnd_fine,
                   x_fine=x_fine,
                   afull_idx_fine=afull_idx_fine)
        pass
    pass

    



def tip_field_integral(scp,param):
    """ Calculate the tip field integral u
from the optimization parameter or result param"""
    # Param represents first elements of u, except for the
    # very first, which is always treated as zero.

    assert(param.shape==(scp.afull_idx+1,))

    # u represents external distributed sigma infinity,
    # defined on the fine centers x_fine.
    # It is extracted from interpolating param,
    # which is defined on the coarse centers x (with the first one zero)
    #
    # The spatial derivative of u is the actual external stresses,
    # defined on xbnd_fine
    # corresponding to partial crack lengths xbnd_fine
    # ... The peak value of u should correspond to the total external stress
    u = np.zeros(scp.x_fine.shape[0],dtype='d') # u defined on the centers x



    xcat = np.concatenate((np.array((-scp.dx/2.0,),dtype='d'),scp.x[:(scp.afull_idx)]))
    paramcat=np.concatenate((np.array((0,),dtype='d'),param[:-1]))

    
    # linear interpolation of param
    paraminterp = scipy.interpolate.interp1d(xcat,paramcat,kind="linear",fill_value="extrapolate") # We extrapolate the last value (prior to tip) forward then manually add the crack tip value, below
    #u[:(fine_refinement//2)]=0.0
    u[0]=0.0
    u[1:(scp.afull_idx_fine)]=paraminterp(scp.x_fine[1:(scp.afull_idx_fine)])
    # u[afull_idx_fine] is the first location past the end of the crack
    # This should bring it up to the full value at the end of param
    u[scp.afull_idx_fine]=param[-1]
    # trailing elements of u are just constant,
    # so there is no derivative at the end
    u[(scp.afull_idx_fine+1):]=u[scp.afull_idx_fine]
    
    return u

# u defined on x positions, represents crack tip stress levels
# du/da defined on x boundaries ... first one always zero
def sigmacontact_from_displacement(scp,u):
    da = scp.dx_fine # same step size

    # integral starts at a0, where sigma_closure > 0 (compressive)
    #first_closureidx = np.where(sigma_closure >0)[0][0]
    # first_closureidx can be thought of as index into x_bnd
    #    last_closureidx = np.where(x_bnd >= a)[0][0]
    
    du_da = np.concatenate((np.array((0.0,),dtype='d'),np.diff(u)/da,np.array((0.0,),dtype='d'))) # defined on the boundaries xbnd_fine
    
    displacement_coarse = scp.crack_initial_opening - (scp.sigma_closure/scp.Hm)**(2.0/3.0)

    displacement=scipy.interpolate.interp1d(scp.x,displacement_coarse,kind="linear",fill_value="extrapolate")(scp.x_fine)
    
    
    for aidx in range(scp.afull_idx_fine):
        #assert(sigma_closure[aidx] > 0)
        #   in next line: sqrt( (a+x) * (a-x) where x >= 0 and
        #   throw out where x >= a
        # diplacement defined on x_fine
        displacement[:aidx] += (4.0/scp.E)*du_da[aidx+1]*np.sqrt((scp.x_fine[aidx]+scp.x_fine[:aidx])*(scp.x_fine[aidx]-scp.x_fine[:aidx]))*da
        # Add in the x=a position
        # Here we have the integral of (4/E)*(du/da)*sqrt( (a+x)* (a-x) )da
        # as a goes from x[aidx] to x[aidx]+da/2
        # ... treat du/da and sqrt(a+x) as constant:
        #  (4/E)*(du/da)*sqrt(a+x) * integral of sqrt(a-x) da
        # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * (  (x[aidx]+da/2-x[aidx])^(3/2) - (x[aidx]-x[aidx])^(3/2) )
        # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * ( (da/2)^(3/2) )
        displacement[aidx] += (4.0/scp.E)*du_da[aidx+1]*np.sqrt(2.0*scp.x_fine[aidx])*(da/2.0)**(3.0/2.0)
        pass
    
    # Displacement calculated....
    # Sigma contact exists where displacement is negative
    # sigmacontact is positive compression

    sigma_contact = np.zeros(scp.x_fine.shape[0],dtype='d')
    sigma_contact[displacement < 0.0] = ((-displacement[displacement < 0.0])**(3.0/2.0)) * scp.Hm

    
    return (sigma_contact,displacement)


def sigmacontact_from_stress(scp,u):
    # sigmacontact is positive compression

    sigma_closure_interp=scipy.interpolate.interp1d(scp.x,scp.sigma_closure,kind="linear",fill_value="extrapolate")(scp.x_fine)
    
    #sigmacontact = copy.copy(sigma_closure)
    sigmacontact = copy.copy(sigma_closure_interp) # copying slightly redundant  no that we no longer use sigma_closure_interp for anything else
    da = scp.dx_fine # same step size
    
    # integral starts at a0, where sigma_closure > 0 (compressive)
    #first_closureidx = np.where(sigma_closure >0)[0][0]
    # first_closureidx can be thought of as index into x_bnd
    #last_closureidx = np.where(x_bnd >= a)[0][0]
    
    du_da = np.concatenate((np.array((0.0,),dtype='d'),np.diff(u)/da,np.array((0.0,),dtype='d'))) # defined on the boundaries xbnd_fine

    for aidx in range(scp.afull_idx_fine+1):
        #assert(sigma_closure[aidx] > 0)

        sigmacontact[(aidx+1):] -= (1.0/sqrt(2.0))*du_da[aidx+1]*sqrt(scp.x_fine[aidx]/(scp.x_fine[(aidx+1):]-scp.x_fine[aidx]))*da

        # Need to include integral from 
        # of (1/sqrt(2))(du/da)sqrt(a/(x-a))da
        # as a goes from x[aidx]-da/2 to x[aidx]
        # approximate sqrt(x-a) as only a dependence
        # (1/sqrt(2))(du/da)*sqrt(a)*integral of sqrt(1/(x-a)) da
        # as a goes from x[aidx]-da/2 to x[aidx]
        # = (1/sqrt(2))(du/da)*sqrt(a) * (-2sqrt(x-x) + 2sqrt(x-a+da/2))
        # = (1/sqrt(2))(du/da)*sqrt(a) * 2sqrt(da/2))

        sigmacontact[aidx] -= (1.0/sqrt(2.0))*du_da[aidx+1]*np.sqrt(scp.x_fine[aidx])*2.0*sqrt(da/2.0)
        pass
    return sigmacontact

#def solve_sigmacontact(sigma_ext):
    # Contraints:
    #  * integral of du/da must equal sigma_ext
    #  * sigmacontact_from_stress = sigmacontact_from_displacement

    ## Normalize each component for the solver

    #sigma_nominal = (np.sqrt(np.mean(sigma_closure**2.0)) + np.abs(sigma_ext))/2.0

def goal_function(param,scp,sigma_ext):

    u = tip_field_integral(scp,param)
    #last_closureidx = np.where(x_bnd >= a)[0][0]

    #  constraint that integral of du/da must equal sigma_ext
    # means that last value of u should match sigma_ext
    
    # sigmacontact is positive compression
    
    (from_displacement,displacement) = sigmacontact_from_displacement(scp,u)
    from_stress = sigmacontact_from_stress(scp,u)
    
    # elements of residual have units of stress^2
    residual = (from_displacement[:scp.afull_idx_fine]-from_stress[:scp.afull_idx_fine])

    average = (from_displacement[:scp.afull_idx_fine]+from_stress[:scp.afull_idx_fine])/2.0
    negative = average[average < 0]  # negative sigmacontact means tension on the surfaces, which is not allowed!
    
    return np.sum(residual**2.0) + np.sum(negative**2.0) + residual.shape[0]*(u[scp.afull_idx_fine]-sigma_ext)**2.0


def calc_contact(scp,sigma_ext):
    iniguess=np.arange(scp.afull_idx+1,dtype='d')/(scp.afull_idx+1) * sigma_ext
    
    res = scipy.optimize.minimize(goal_function,iniguess,args=(scp,sigma_ext),
                                  options={"eps": 10000.0})
    #res = scipy.optimize.minimize(goal_function,iniguess,method='nelder-mead',options={"maxfev": 15000})

    param=res.x
    
    u = tip_field_integral(scp,param)
    (contact_stress,displacement) = sigmacontact_from_displacement(scp,u)

    return (param,contact_stress,displacement)
    

def doplots(scp,param):
    
    u = tip_field_integral(scp,param)
    #last_closureidx = np.where(x_bnd >= a)[0][0]

    #  constraint that integral of du/da must equal sigma_ext
    # means that last value of u should match sigma_ext
    
    # sigmacontact is positive compression

    du_da = np.concatenate((np.array((0.0,),dtype='d'),np.diff(u)/scp.dx_fine,np.array((0.0,),dtype='d'))) # defined on the boundaries xbnd_fine

    (from_displacement,displacement) = sigmacontact_from_displacement(scp,u)
    from_stress = sigmacontact_from_stress(scp,u)

    pl.figure(1)
    pl.clf()
    pl.plot(param)
    pl.grid()
    
    
    pl.figure(2)
    pl.clf()
    pl.plot(scp.x_fine,u)
    pl.title("distributed stress concentration")
    pl.grid()

    pl.figure(3)
    pl.clf()
    pl.plot(scp.x_fine,from_displacement,'-',
            scp.x_fine,from_stress,'-'),
    pl.grid()
    pl.legend(("from displacement","from stress"))
    pl.title("sigmacontact")

    pl.figure(4)
    pl.clf()
    pl.plot(scp.xbnd_fine,du_da,'-')
    pl.grid()
    pl.title("distributed stress concentration derivative")
    pass
    

if __name__=="__main__":
    #####INPUT VALUES
    E = 200e9    #Plane stress Modulus of Elasticity
    Eeff=E
    sigma_yield = 400e6
    tau_yield = sigma_yield/2.0 # limits stress concentration around singularity
    nu = 0.33    #Poisson's Ratio
    
    
    a_input=2.0e-3  # half-crack length (m)
    
    # in this model, we require crack length to line up
    # on an element boundary, so we will round it to
    # the nearest boundary below when we store a
    
    xmax = 5e-3 # as far out in x as we are calculating (m)
    xsteps = 200
    
    
    #fine_refinement=int(4)
    fine_refinement=int(1)

    # 1/Hm has units of m^(3/2)/Pascal
    # Hm has units of Pa/m^(3/2)
    Hm = 10e6/(100e-9**(3.0/2.0))  # rough order of magnitude guess

    scp = sc_params.fromcrackgeom(E,xmax,xsteps,a_input,fine_refinement,Hm)

    
    # Closure state (function of position; positive compression)
    # Use hard closure model to solve for closure state
    #crack_model = ModeI_throughcrack_CODformula(Eeff)
    crack_model = Tada_ModeI_CircularCrack_along_midline(E,nu)
    
    
    
    observed_reff = np.array([ 0.0,  1e-3, 1.5e-3, scp.a  ],dtype='d')
    observed_seff = np.array([ 10e6, 15e6, 30e6, 150e6  ],dtype='d')
    
    sigma_closure = inverse_closure(observed_reff,
                                    observed_seff,
                                    scp.x,scp.x_bnd,scp.dx,scp.a,sigma_yield,
                                    crack_model)

    crack_initial_opening = crackopening_from_tensile_closure(scp.x,scp.x_bnd,sigma_closure,scp.dx,scp.a,sigma_yield,crack_model)
    
    # temporary zero out sigma_closure
    #sigma_closure[:]=0.0
    
    # In the soft closure model, sigma_closure can't be negativej
    # (use crack_initial_opening values instead in that domain)
    sigma_closure[sigma_closure < 0.0]=0.0
    


    scp.setcrackstate(sigma_closure,crack_initial_opening)
    


    sigma_ext=50e6
    
    
    #iniguess = -np.cumsum(sigma_closure)[1:(last_closureidx+1)]*dx
    iniguess=np.arange(scp.afull_idx+1,dtype='d')/(scp.afull_idx+1) * sigma_ext
    
    res = scipy.optimize.minimize(goal_function,iniguess,args=(scp,sigma_ext),
                                  options={"eps": 10000.0})
    #res = scipy.optimize.minimize(goal_function,iniguess,method='nelder-mead',options={"maxfev": 15000})


    doplots(scp,res.x)
    #    pass
