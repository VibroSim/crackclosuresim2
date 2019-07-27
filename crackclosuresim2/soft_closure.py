import copy
import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize
import scipy as sp


from crackclosuresim2 import solve_normalstress
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

    

def interpolate_hardcontact_intensity(scp,dsigmaext_dxt_hardcontact):
    dsigmaext_dxt_hardcontact_interp = scipy.interpolate.interp1d(scp.x,dsigmaext_dxt_hardcontact,kind="linear",fill_value="extrapolate")(scp.x_fine)
    dsigmaext_dxt_hardcontact_interp[np.isnan(dsigmaext_dxt_hardcontact_interp)]=0.0  # No singularity means no concentration

    return dsigmaext_dxt_hardcontact_interp
    

def tip_field_integral_OBSOLETE(scp,param,dsigmaext_dxt_hardcontact):
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


    dsigmaext_dxt_hardcontact_interp = scipy.interpolate.interp1d(scp.x,dsigmaext_dxt_hardcontact,kind="linear",fill_value="extrapolate")(scp.x_fine)
    dsigmaext_dxt_hardcontact_interp[np.isnan(dsigmaext_dxt_hardcontact_interp)]=0.0  # No singularity means no concentration

    return (u,dsigmaext_dxt_hardcontact_interp)

def calc_du_da_OBSOLETE(u,da):
    return np.concatenate((np.array((0.0,),dtype='d'),np.diff(u)/da,np.array((0.0,),dtype='d'))) # defined on the boundaries xbnd_fine

def update_du_da_with_hardcontact(du_da,dsigmaext_dxt_hardcontact_interp,closure_index):
    du_da[:closure_index] = dsigmaext_dxt_hardcontact_interp[:closure_index]  # dsigmaext_dxt_hardcontact_interp is defined on x_fine
    pass


# du/da defined on x positions... represents distributed stress concentrations
def sigmacontact_from_displacement(scp,du_da,dsigmaext_dxt_hardcontact_interp):
    da = scp.dx_fine # same step size

    # integral starts at a0, where sigma_closure > 0 (compressive)
    #first_closureidx = np.where(sigma_closure >0)[0][0]
    # first_closureidx can be thought of as index into x_bnd
    #    last_closureidx = np.where(x_bnd >= a)[0][0]
    
    #du_da = calc_du_da(u,da)
    du_da_corrected=copy.copy(du_da)

    # .. why use du_da[1:...]? !!!
    
    displacement_coarse = scp.crack_initial_opening - (scp.sigma_closure/scp.Hm)**(2.0/3.0)

    displacement=scipy.interpolate.interp1d(scp.x,displacement_coarse,kind="linear",fill_value="extrapolate")(scp.x_fine)
    
    closure_index = None

    # !!!*** Possibly calculation should start at scp.afull_idx_fine instead of
    # afull_idx_fine-1.
    # Then residual and average in goal_function should consider the full range (including afull_idx_fine
    # But this causes various problems:
    #   * displacement assertion failed below
    #   * Convergence failure 'Positive directional derivative for linesearch'

    for aidx in range(scp.afull_idx_fine-1,-1,-1):  
        #assert(sigma_closure[aidx] > 0)
        #   in next line: sqrt( (a+x) * (a-x) where x >= 0 and
        #   throw out where x >= a
        # diplacement defined on x_fine
        displacement[:aidx] += (4.0/scp.E)*du_da_corrected[aidx]*np.sqrt((scp.x_fine[aidx]+scp.x_fine[:aidx])*(scp.x_fine[aidx]-scp.x_fine[:aidx]))*da
        # Add in the x=a position
        # Here we have the integral of (4/E)*(du/da)*sqrt( (a+x)* (a-x) )da
        # as a goes from x[aidx] to x[aidx]+da/2
        # ... treat du/da and sqrt(a+x) as constant:
        #  (4/E)*(du/da)*sqrt(a+x) * integral of sqrt(a-x) da
        # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * (  (x[aidx]+da/2-x[aidx])^(3/2) - (x[aidx]-x[aidx])^(3/2) )
        # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * ( (da/2)^(3/2) )
        displacement[aidx] += (4.0/scp.E)*du_da_corrected[aidx]*np.sqrt(2.0*scp.x_fine[aidx])*(da/2.0)**(3.0/2.0)

        if displacement[aidx] > 0.0 and closure_index is None:
            assert(np.all(displacement[:aidx] > 0.0)) # if this point is open, then all points to the left should be open too

            # Open points to the left of this have distributed crack tip intensities according to the hard contact
            # (actually no contact) model with data in dsigmaext_dxt_hardcontact
            closure_index = aidx
            update_du_da_with_hardcontact(du_da_corrected,dsigmaext_dxt_hardcontact_interp,closure_index)
            pass
        pass
    
    # Displacement calculated....
    # Sigma contact exists where displacement is negative
    # sigmacontact is positive compression

    sigma_contact = np.zeros(scp.x_fine.shape[0],dtype='d')
    sigma_contact[displacement < 0.0] = ((-displacement[displacement < 0.0])**(3.0/2.0)) * scp.Hm

    
    return (sigma_contact,displacement,closure_index,du_da_corrected) # returned du_da is actually du_da_corrected...


def sigmacontact_from_stress(scp,du_da_corrected,dsigmaext_dxt_hardcontact_interp):
    # sigmacontact is positive compression

    sigma_closure_interp=scipy.interpolate.interp1d(scp.x,scp.sigma_closure,kind="linear",fill_value="extrapolate")(scp.x_fine)
    
    #sigmacontact = copy.copy(sigma_closure)
    sigmacontact = copy.copy(sigma_closure_interp) # copying slightly redundant  no that we no longer use sigma_closure_interp for anything else
    da = scp.dx_fine # same step size
    
    # integral starts at a0, where sigma_closure > 0 (compressive)
    # OR closure state with external load is compressive

    #first_closureidx = np.where(sigma_closure >0)[0][0]
    # first_closureidx can be thought of as index into x_bnd
    #last_closureidx = np.where(x_bnd >= a)[0][0]
   
    #du_da = calc_du_da(u,da)

    #update_du_da_with_hardcontact(du_da,dsigmaext_dxt_hardcontact_interp,closure_index)
    
    
    for aidx in range(scp.afull_idx_fine+1):
        #assert(sigma_closure[aidx] > 0)

        sigmacontact[(aidx+1):] -= du_da_corrected[aidx]*((1.0/sqrt(2.0))*sqrt(scp.x_fine[aidx]/(scp.x_fine[(aidx+1):]-scp.x_fine[aidx])) + 1.0)*da # + 1.0 represents stress when a large distance away from effective tip

        # Need to include integral 
        # of (du/da)[(1/sqrt(2))sqrt(a/(x-a)) + 1]da
        # as a goes from x[aidx]-da/2 to x[aidx]
        # approximate sqrt(x-a) as only a dependence
        # (du/da)*(1/sqrt(2))*sqrt(a)*integral of sqrt(1/(x-a)) da + integral of du/da da
        # as a goes from x[aidx]-da/2 to x[aidx]
        # = (du/da)(1/sqrt(2))*sqrt(a) * (-2sqrt(x-x) + 2sqrt(x-a+da/2))  + (du/da)(da/2)
        # = (1/sqrt(2))(du/da)*sqrt(a) * 2sqrt(da/2)) + (du/da)(da/2)

        sigmacontact[aidx] -= (du_da_corrected[aidx]*(1.0/sqrt(2.0))*np.sqrt(scp.x_fine[aidx])*2.0*sqrt(da/2.0) + du_da_corrected[aidx]*da/2.0)
        pass
    return sigmacontact

#def solve_sigmacontact(sigma_ext):
    # Contraints:
    #  * integral of du/da must equal sigma_ext
    #  * sigmacontact_from_stress = sigmacontact_from_displacement

    ## Normalize each component for the solver

    #sigma_nominal = (np.sqrt(np.mean(sigma_closure**2.0)) + np.abs(sigma_ext))/2.0

def soft_closure_goal_function_opening(du_da_shortened,scp,sigma_ext,dsigmaext_dxt_hardcontact):

    du_da = np.concatenate((du_da_shortened,np.zeros(scp.xsteps*scp.fine_refinement - scp.afull_idx_fine - 2 ,dtype='d')))
    
    #(u,dsigmaext_dxt_hardcontact_interp) = tip_field_integral(scp,param,dsigmaext_dxt_hardcontact)
    dsigmaext_dxt_hardcontact_interp=interpolate_hardcontact_intensity(scp,dsigmaext_dxt_hardcontact)
    #last_closureidx = np.where(x_bnd >= a)[0][0]

    #  constraint that integral of du/da must equal sigma_ext
    # means that last value of u should match sigma_ext
    
    # sigmacontact is positive compression
    
    (from_displacement,displacement,closure_index,du_da_corrected) = sigmacontact_from_displacement(scp,du_da,dsigmaext_dxt_hardcontact_interp)

    u_corrected = np.cumsum(du_da_corrected)*scp.dx_fine
    # u_corrected nominally on position basis x_fine+dx_fine/2.0
    
    from_stress = sigmacontact_from_stress(scp,du_da_corrected,dsigmaext_dxt_hardcontact_interp)
    
    # elements of residual have units of stress^2
    # !!!*** should from_displacement consider to afull_idx_fine+1???
    residual = (from_displacement[:(scp.afull_idx_fine)]-from_stress[:(scp.afull_idx_fine)])

    average = (from_displacement[:(scp.afull_idx_fine)]+from_stress[:(scp.afull_idx_fine)])/2.0
    negative = average[average < 0]  # negative sigmacontact means tension on the surfaces, which is not allowed!
    
    #return np.sum(residual**2.0) + 1.0*np.sum(negative**2.0) + residual.shape[0]*(u[scp.afull_idx_fine]-sigma_ext)**2.0
    return np.sum(residual**2.0) + 1.0*np.sum(negative**2.0) + 10*residual.shape[0]*(u_corrected[scp.afull_idx_fine]-sigma_ext)**2.0 


def calc_contact(scp,sigma_ext):

    if sigma_ext >= 0: # Tensile
        #iniguess=np.arange(scp.afull_idx+1,dtype='d')/(scp.afull_idx+1) * sigma_ext  # This iniguess was for u

        # now we use du_da
        iniguess=np.ones(scp.x_fine.shape[0],dtype='d')*(1.0/(scp.afull_idx+1)) * sigma_ext/scp.dx_fine  #

        # sigma_yield = scp.sigma_yield  # not a parameter yet... just use infinity for now
        sigma_yield=np.inf
        
        #crack_model=scp.crack_model  # not a parameter yet...
        crack_model = ModeI_throughcrack_CODformula(scp.E)

        # call solve_normalstress with an infinite tensile stress to get dsigmaext_dxt_hardcontact
        (effective_length, sigma, tensile_displ, dsigmaext_dxt_hardcontact) = solve_normalstress(scp.x,scp.x_bnd,scp.sigma_closure,scp.dx,np.inf,scp.a,sigma_yield,crack_model)

        
        # Calculate x

        #nonnegative_constraint = scipy.optimize.NonlinearConstraint(lambda du_da: du_da,0.0,np.inf)
        nonnegative_constraint = { "type": "ineq",
                                   "fun": lambda du_da: du_da }
        
        dsigmaext_dxt_hardcontact_interp=interpolate_hardcontact_intensity(scp,dsigmaext_dxt_hardcontact)
        
        def load_constraint_fun(du_da):
            # NOTE: Could make faster by accelerating sigmacontact_from_displacement to not do anything once it figures out du_da_corrected
            # NOTE: NOT CURRENTLY USED!!!
            (sigma_contact,displacement,closure_index,du_da_corrected) = sigmacontact_from_displacement(scp,du_da,dsigmaext_dxt_hardcontact_interp)
            return np.cumsum(du_da_corrected)[scp.afull_idx_fine]*scp.dx_fine-sigma_ext
        
        load_constraint = { "type": "eq",
                            "fun": load_constraint_fun }
        
        res = scipy.optimize.minimize(soft_closure_goal_function_opening,iniguess[:(scp.afull_idx_fine+1)],args=(scp,sigma_ext,dsigmaext_dxt_hardcontact),
                                      constraints = [ nonnegative_constraint ], # , load_constraint ],
                                      method="SLSQP",
                                      options={"eps": 10000.0,
                                               "maxiter": 100000})
        #res = scipy.optimize.minimize(goal_function,iniguess,method='nelder-mead',options={"maxfev": 15000})
        if not res.success and res.status != 4:
            # (ignore incompatible constraint, because our constraints are
            # compatible by definition, and scipy 1.2 seems to diagnose
            # this incorrectly... should file bug report)
            print("minimize error %d: %s" % (res.status,res.message))
            import pdb
            pdb.set_trace()
            pass
        
        du_da_shortened=res.x
        du_da = np.concatenate((du_da_shortened,np.zeros(scp.xsteps*scp.fine_refinement - scp.afull_idx_fine - 2 ,dtype='d')))
        pass
    else:
        # Compressive
        assert(0)
        pass
    
    #(u,dsigmaext_dxt_hardcontact_interp) = tip_field_integral(scp,param,dsigmaext_dxt_hardcontact)
    dsigmaext_dxt_hardcontact_interp=interpolate_hardcontact_intensity(scp,dsigmaext_dxt_hardcontact)
    (contact_stress,displacement,closure_index,du_da_corrected) = sigmacontact_from_displacement(scp,du_da,dsigmaext_dxt_hardcontact_interp)

    return (du_da,du_da_corrected,contact_stress,displacement,dsigmaext_dxt_hardcontact)
    

def soft_closure_plots(scp,du_da,dsigmaext_dxt_hardcontact):
    from matplotlib import pyplot as pl
    #pl.rc('text', usetex=True) # Support greek letters in plot legend
    
    #(u,dsigmaext_dxt_hardcontact_interp) = tip_field_integral(scp,param,dsigmaext_dxt_hardcontact)
    #last_closureidx = np.where(x_bnd >= a)[0][0]

    #  constraint that integral of du/da must equal sigma_ext
    # means that last value of u should match sigma_ext
    
    # sigmacontact is positive compression

    #du_da = calc_du_da(u,scp.dx_fine)

    dsigmaext_dxt_hardcontact_interp=interpolate_hardcontact_intensity(scp,dsigmaext_dxt_hardcontact)

    
    (from_displacement,displacement,closure_index,du_da_corrected) = sigmacontact_from_displacement(scp,du_da,dsigmaext_dxt_hardcontact_interp)
    from_stress = sigmacontact_from_stress(scp,du_da_corrected,dsigmaext_dxt_hardcontact_interp)

    #update_du_da_with_hardcontact(du_da,dsigmaext_dxt_hardcontact_interp,closure_index)

    
    
    #pl.figure()
    #pl.clf()
    #pl.plot(scp.x_fine,u)
    #pl.title("distributed stress concentration")
    #pl.grid()

    pl.figure()
    pl.clf()
    pl.plot(scp.x_fine*1e3,from_displacement/1e6,'-',
            scp.x_fine*1e3,from_stress/1e6,'-'),
    pl.grid()
    pl.legend(("from displacement","from stress"))
    pl.ylabel("Stress (MPa)")
    pl.xlabel('Position (mm)')
    pl.title("sigmacontact")

    u_corrected = np.cumsum(du_da_corrected)*scp.dx_fine
    # u_corrected nominally on position basis x_fine+dx_fine/2.0

    pl.figure()
    pl.clf()
    pl.plot(scp.x_fine*1e3,du_da_corrected/1e6,'-')
    pl.grid()
    pl.title("distributed stress concentration derivative (corrected)\ntotal load=%g" % (u_corrected[scp.afull_idx_fine]))
    pl.xlabel('Position (mm)')
    pl.ylabel("Distributed stress concentration (MPa/m)")

    pl.figure()
    pl.clf()
    pl.plot(scp.x*1e3,(scp.crack_initial_opening-(scp.sigma_closure/scp.Hm)**(2.0/3.0))*1e6,'-')
    pl.plot(scp.x_fine*1e3,displacement*1e6,'-')
    pl.grid()
    pl.legend(('Initial displacement','Final displacement'))
    pl.xlabel('Position (mm)')
    pl.ylabel('Displacement (um)')
    pass
    
