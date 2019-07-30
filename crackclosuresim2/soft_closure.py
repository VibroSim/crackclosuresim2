import copy
import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize
import scipy as sp


from crackclosuresim2 import solve_normalstress
from crackclosuresim2 import inverse_closure,crackopening_from_tensile_closure
from crackclosuresim2 import ModeI_Beta_COD_Formula
from crackclosuresim2 import ModeI_throughcrack_CODformula
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline



class sc_params(object):
    # soft closure parameters
    crack_model = None # crackclosuresim ModeI_crack_model object  (for now should be derived from ModeI_Beta_COD_Formula)
    Hm = None # H*m
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

        # For now we only support crack models with simple
        # COD formulas... specifically the throughcrack and Tada models
        assert(isinstance(self.crack_model,ModeI_Beta_COD_Formula))
        
        pass

    def setcrackstate(self,sigma_closure,crack_initial_opening):
        self.sigma_closure=sigma_closure
        self.crack_initial_opening=crack_initial_opening
        pass

    def initialize_contact(self,sigma_closure,crack_initial_opening):

        assert(np.all(sigma_closure >= 0.0)) # given sigma_closure should not have any tensile component
        
        #self.crack_initial_opening=copy.copy(crack_initial_opening)
        self.crack_initial_opening=np.zeros(self.x.shape[0],dtype='d')
        self.sigma_closure=np.zeros(self.x.shape[0],dtype='d')

        ## closure index is the last point prior to closure
        closure_index = np.where(sigma_closure[:(self.afull_idx)]!=0.0)[0][0]*self.fine_refinement - 1
        #if closure_index < 0:
        #    closure_index = None
        #    pass
        assert(np.all(sigma_closure[:(closure_index//self.fine_refinement+1)]==0.0))

        
        
        # ***!!!! will need to change this integration for a different
        # crack model!!!***
        #thickness=2.0e-3 # ***!!! should get from crack model... but we divide it out so it really doesn't matter
        #sigma_closure_avg_stress = np.sum(sigma_closure[sigma_closure > 0.0])*self.dx*thickness/(self.a*thickness)
        
        # now we use du_da_shortened
        # which is  afull_idx_fine+1 numbers representing
        # the distributed stress concentration
        # ... Note that the distributed stress concentration indices
        # have no effect to the left of the closure point 

        sigma_closure_interp = scipy.interpolate.interp1d(self.x,sigma_closure,kind="linear",fill_value="extrapolate")(self.x_fine)


        # du_da_shortened has the initial open region removed, as we
        # disallow any concentration in this open region --
        # crack should not close. 
        du_da_shortened_iniguess=np.zeros(self.afull_idx_fine+1-(closure_index+1),dtype='d')*(1.0/(self.afull_idx+1)) # * (-sigma_closure_avg_stress)/self.dx_fine  #
        du_da_shortened_iniguess[:(self.afull_idx_fine+1-(closure_index+1))] = -sigma_closure_interp[(closure_index+1):(self.afull_idx_fine+1)]

        
        
        # sigma_yield = self.sigma_yield  # not a parameter yet... just use infinity for now
        sigma_yield=np.inf
        
        crack_model=self.crack_model  # not a parameter yet...
        
        
        #def load_constraint_fun(uniform_du_da_shortened):
        #    # NOTE: Could make faster by accelerating sigmacontact_from_displacement to not do anything once it figures out du_da_corrected
        #    
        #    uniform_load=uniform_du_da_shortened[0]*self.dx_fine  # Uniform stress in Pa across crack
        #    uniform_load = 0.0
        #    
        #    du_da = np.concatenate((uniform_du_da_shortened[1:],np.zeros(self.xsteps*self.fine_refinement - self.afull_idx_fine - 2 ,dtype='d')))        
        
        #    (sigma_contact,displacement,closure_index,du_da_corrected) = sigmacontact_from_displacement(self,uniform_load,du_da,dsigmaext_dxt_hardcontact_interp)
            
        #   if closure_index is not None or uniform_load > 0.0:
        #       # Open crack cannot hold uniform load; no tensile uniform load possible
        #       uniform_load = 0.0
        #       pass
        #   
        #   return (uniform_load + np.cumsum(du_da_corrected)[self.afull_idx_fine]*self.dx_fine)-(-sigma_closure_avg_stress)
        
        #load_constraint = { "type": "eq",
        #                    "fun": load_constraint_fun }
        
        
        #def crack_open_constraint_fun(uniform_du_da_shortened):
        #    """ Represents that crack should be open up
        #       through closure_index and closed thereafter"""
        #    
        #    uniform_load=uniform_du_da_shortened[0]*self.dx_fine  # Uniform stress in Pa across crack
        #    uniform_load = 0.0
            
        #    du_da = np.concatenate((uniform_du_da_shortened[1:],np.zeros(self.xsteps*self.fine_refinement - self.afull_idx_fine - 2 ,dtype='d')))        
            
        #    (sigma_contact,displacement,closure_index_junk,du_da_corrected) = sigmacontact_from_displacement(self,uniform_load,du_da,dsigmaext_dxt_hardcontact_interp)
        #    
        #    return np.concatenate((displacement[:(closure_index+1)],-displacement[(closure_index+1):]))  # require displacement >= 0 in open region and <= 0 in closed region

        #crack_open_constraint = { "type": "ineq",
        #                          "fun": crack_open_constraint_fun }
        

               
        

        constraints = [] #[ load_constraint, crack_open_constraint ]
        
        res = scipy.optimize.minimize(initialize_contact_goal_function,du_da_shortened_iniguess,args=(self,sigma_closure_interp,closure_index),
                                      constraints = constraints,
                                      method="SLSQP",
                                      options={"eps": 10000.0,
                                               "maxiter": 10000,
                                               "ftol": self.afull_idx_fine*(abs(np.mean(sigma_closure))+20e6)**2.0/1e19})
        
        if not res.success: #  and res.status != 4:
            # (ignore incompatible constraint, because our constraints are
            # compatible by definition, and scipy 1.2 seems to diagnose
            # this incorrectly... should file bug report)
            print("minimize error %d: %s" % (res.status,res.message))
            import pdb
            pdb.set_trace()
            pass
        
        du_da_shortened=res.x
        du_da = np.concatenate((np.zeros(closure_index+1,dtype='d'),du_da_shortened,np.zeros(self.xsteps*self.fine_refinement - self.afull_idx_fine - 2 ,dtype='d')))
        
        (contact_stress,displacement) = sigmacontact_from_displacement(self,du_da)

        from_stress = sigmacontact_from_stress(self,du_da)

        # displacement = crack_initial_opening - (sigma_closure_stored/Hm)^(2/3) + displacement_due_to_distributed_stress_concentrations
        # sigmacontact_from_displacement = 0 when displacement > 0
        # sigmacontact_from_displacement = (-displacement)^(3/2)*Hm when displacement < 0
        
        # we made sigmacontact_from_stress match sigma_closure...

        # So let's set sigmacontact_from_displacement equal and solve for displacement
        # sigma_closure = (-displacement)^(3/2)*Hm
        # (-displacement)^(3/2) = sigma_closure/Hm = 
        # displacement = -(sigma_closure/Hm)^(2/3)
        # crack_initial_opening + displacement_due_to_distributed_stress_concentrations = -(sigma_closure/Hm)^(2/3)
        # crack_initial_opening = -displacement_due_to_distributed_stress_concentrations -(sigma_closure/Hm)^(2/3)

        # To evaluate modeled residual stressing operation starting
        # at zero initial opening and zero closure, 
        # stop here, and run soft_closure_plots()
        self.crack_initial_opening = crack_initial_opening -displacement - (sigma_closure/self.Hm)**(2.0/3.0)  # Through this line we have evaluated an initial opening displacement

        #sys.modules["__main__"].__dict__.update(globals())
        #sys.modules["__main__"].__dict__.update(locals())
        #raise ValueError("Foo!")

        # Now we apply our sigma_closure
        (contact_stress,displacement) = sigmacontact_from_displacement(self,du_da)

        from_stress = sigmacontact_from_stress(self,du_da)

        # This becomes our new initial state
        self.crack_initial_opening = displacement + (sigma_closure/self.Hm)**(2.0/3.0) # we add sigma_closure displacement back in because sigmacontact_from_displacement() will subtract it out
        self.sigma_closure = sigma_closure
        du_da[:]=0.0
        
        
        #sys.modules["__main__"].__dict__.update(globals())
        #sys.modules["__main__"].__dict__.update(locals())
        #raise ValueError("Foo!")
    
        pass
    


    
    @classmethod
    def fromcrackgeom(cls,crack_model,xmax,xsteps,a_input,fine_refinement,Hm):
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


        return cls(crack_model=crack_model,
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



def initialize_contact_goal_function(du_da_shortened,scp,sigma_closure_interp,closure_index):

    du_da = np.concatenate((np.zeros(closure_index+1,dtype='d'),du_da_shortened,np.zeros(scp.xsteps*scp.fine_refinement - scp.afull_idx_fine - 2 ,dtype='d')))

    

    #last_closureidx = np.where(x_bnd >= a)[0][0]

    #  constraint that integral of du/da must equal sigma_ext
    # means that last value of u should match sigma_ext
    
    # sigmacontact is positive compression
    
    (from_displacement,displacement) = sigmacontact_from_displacement(scp,du_da)

    #u = np.cumsum(du_da)*scp.dx_fine
    # u nominally on position basis x_fine+dx_fine/2.0
    
    from_stress = sigmacontact_from_stress(scp,du_da)
    
    # elements of residual have units of stress^2
    # !!!*** should from_displacement consider to afull_idx_fine+1???
    residual = (sigma_closure_interp[:(scp.afull_idx_fine+1)]-from_stress[:(scp.afull_idx_fine+1)])

    #average = (from_displacement[:(scp.afull_idx_fine+1)]+from_stress[:(scp.afull_idx_fine+1)])/2.0
    #negative = average[average < 0]  # negative sigmacontact means tension on the surfaces, which is not allowed!
    
    #return np.sum(residual**2.0) + 1.0*np.sum(negative**2.0) + residual.shape[0]*(u[scp.afull_idx_fine]-sigma_ext)**2.0
    return np.sum(residual**2.0) # + 1.0*np.sum(negative**2.0) #  + 10*residual.shape[0]*(u[scp.afull_idx_fine]-sigma_ext)**2.0 








# du/da defined on x positions... represents distributed stress concentrations
def sigmacontact_from_displacement(scp,du_da):
    # 
    da = scp.dx_fine # same step size

    # integral starts at a0, where sigma_closure > 0 (compressive)
    #first_closureidx = np.where(sigma_closure >0)[0][0]
    # first_closureidx can be thought of as index into x_bnd
    #    last_closureidx = np.where(x_bnd >= a)[0][0]
    

    
    displacement_coarse = scp.crack_initial_opening - (scp.sigma_closure/scp.Hm)**(2.0/3.0)

    displacement=scipy.interpolate.interp1d(scp.x,displacement_coarse,kind="linear",fill_value="extrapolate")(scp.x_fine)
    
    

    # !!!*** Possibly calculation should start at scp.afull_idx_fine instead of
    # afull_idx_fine-1.
    # Then residual and average in goal_function should consider the full range (including afull_idx_fine
    # But this causes various problems:
    #   * displacement assertion failed below -- caused by sigma_closure not having a value @ afull_idx_fine
    #   * Convergence failure 'Positive directional derivative for linesearch' ... possibly not a problem (need to adjust tolerances?)

    if isinstance(scp.crack_model,ModeI_throughcrack_CODformula):
    
        for aidx in range(scp.afull_idx_fine,-1,-1): 
            #assert(sigma_closure[aidx] > 0)
            #   in next line: sqrt( (a+x) * (a-x) where x >= 0 and
            #   throw out where x >= a
            # diplacement defined on x_fine
            displacement[:aidx] += (4.0/scp.crack_model.Eeff)*du_da[aidx]*np.sqrt((scp.x_fine[aidx]+scp.x_fine[:aidx])*(scp.x_fine[aidx]-scp.x_fine[:aidx]))*da
            # Add in the x=a position
            # Here we have the integral of (4/E)*(du/da)*sqrt( (a+x)* (a-x) )da
            # as a goes from x[aidx] to x[aidx]+da/2
            # ... treat du/da and sqrt(a+x) as constant:
            #  (4/E)*(du/da)*sqrt(a+x) * integral of sqrt(a-x) da
            # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * (  (x[aidx]+da/2-x[aidx])^(3/2) - (x[aidx]-x[aidx])^(3/2) )
            # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * ( (da/2)^(3/2) )
            displacement[aidx] += (4.0/scp.crack_model.Eeff)*du_da[aidx]*np.sqrt(2.0*scp.x_fine[aidx])*(da/2.0)**(3.0/2.0)
            
            pass
        pass
    elif isinstance(scp.crack_model,Tada_ModeI_CircularCrack_along_midline):
        for aidx in range(scp.afull_idx_fine,-1,-1): 
            #assert(sigma_closure[aidx] > 0)
            #   in next line: sqrt( (a+x) * (a-x) where x >= 0 and
            #   throw out where x >= a
            # displacement defined on x_fine
            # NOTE: displacement is 2* usual formula because we are concerned with total distance between both crack faces

            # Tada turns out to have sqrt(xt^2-x^2) === sqrt((xt-x)(xt+x)) form
            # as throughcrack, so same integral calculation applies.
            # We just change the leading factors
            displacement[:aidx] += (8.0*(1.0-scp.crack_model.nu**2.0)/(np.pi*scp.crack_model.E))*du_da[aidx]*np.sqrt((scp.x_fine[aidx]+scp.x_fine[:aidx])*(scp.x_fine[aidx]-scp.x_fine[:aidx]))*da
            # Add in the x=a position
            # Here we have the integral of (4/E)*(du/da)*sqrt( (a+x)* (a-x) )da
            # as a goes from x[aidx] to x[aidx]+da/2
            # ... treat du/da and sqrt(a+x) as constant:
            #  (4/E)*(du/da)*sqrt(a+x) * integral of sqrt(a-x) da
            # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * (  (x[aidx]+da/2-x[aidx])^(3/2) - (x[aidx]-x[aidx])^(3/2) )
            # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * ( (da/2)^(3/2) )
            displacement[aidx] += (8.0*(1.0-scp.crack_model.nu**2.0)/(np.pi*scp.crack_model.E))*du_da[aidx]*np.sqrt(2.0*scp.x_fine[aidx])*(da/2.0)**(3.0/2.0)
            pass

        
        pass
    else:
        raise ValueError("Unsupported crack model for soft_closure")
    
    # Displacement calculated....
    # Sigma contact exists where displacement is negative
    # sigmacontact is positive compression

    
    sigma_contact = np.zeros(scp.x_fine.shape[0],dtype='d')
    sigma_contact[displacement < 0.0] = ((-displacement[displacement < 0.0])**(3.0/2.0)) * scp.Hm
    
    return (sigma_contact,displacement) 


def sigmacontact_from_stress(scp,du_da):
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

    betaval = scp.crack_model.beta(scp.crack_model)
    
    for aidx in range(scp.afull_idx_fine+1):
        #assert(sigma_closure[aidx] > 0)

        sigmacontact[(aidx+1):] -= du_da[aidx]*((betaval/sqrt(2.0))*sqrt(scp.x_fine[aidx]/(scp.x_fine[(aidx+1):]-scp.x_fine[aidx])) + 1.0)*da # + 1.0 represents stress when a large distance away from effective tip

        # Need to include integral 
        # of (du/da)[(1/sqrt(2))sqrt(a/(x-a)) + 1]da
        # as a goes from x[aidx]-da/2 to x[aidx]
        # approximate sqrt(x-a) as only a dependence
        # (du/da)*(1/sqrt(2))*sqrt(a)*integral of sqrt(1/(x-a)) da + integral of du/da da
        # as a goes from x[aidx]-da/2 to x[aidx]
        # = (du/da)(1/sqrt(2))*sqrt(a) * (-2sqrt(x-x) + 2sqrt(x-a+da/2))  + (du/da)(da/2)
        # = (1/sqrt(2))(du/da)*sqrt(a) * 2sqrt(da/2)) + (du/da)(da/2)

        sigmacontact[aidx] -= (du_da[aidx]*(betaval/sqrt(2.0))*np.sqrt(scp.x_fine[aidx])*2.0*sqrt(da/2.0) + du_da[aidx]*da/2.0)
        pass

    return sigmacontact

#def solve_sigmacontact(sigma_ext):
    # Contraints:
    #  * integral of du/da must equal sigma_ext
    #  * sigmacontact_from_stress = sigmacontact_from_displacement

    ## Normalize each component for the solver

    #sigma_nominal = (np.sqrt(np.mean(sigma_closure**2.0)) + np.abs(sigma_ext))/2.0

def soft_closure_goal_function(du_da_shortened,scp,sigma_ext,closure_index):
    # closure_index is used in tension to shorten du_da, disallowing any stresses or concentration to left of initial opening distance
    


    du_da = np.concatenate((np.zeros(closure_index+1,dtype='d'),du_da_shortened,np.zeros(scp.xsteps*scp.fine_refinement - scp.afull_idx_fine - 2 ,dtype='d')))        

    

    
    #last_closureidx = np.where(x_bnd >= a)[0][0]

    #  constraint that integral of du/da must equal sigma_ext
    # means that last value of u should match sigma_ext
    
    # sigmacontact is positive compression
    
    (from_displacement,displacement) = sigmacontact_from_displacement(scp,du_da)

    #u = np.cumsum(du_da)*scp.dx_fine
    # u nominally on position basis x_fine+dx_fine/2.0
    
    from_stress = sigmacontact_from_stress(scp,du_da)
    
    # elements of residual have units of stress^2
    # !!!*** should from_displacement consider to afull_idx_fine+1???
    residual = (from_displacement[:(scp.afull_idx_fine+1)]-from_stress[:(scp.afull_idx_fine+1)])

    average = (from_displacement[:(scp.afull_idx_fine+1)]+from_stress[:(scp.afull_idx_fine+1)])/2.0
    negative = average[average < 0]  # negative sigmacontact means tension on the surfaces, which is not allowed!

    displaced = average[displacement[:(scp.afull_idx_fine+1)] > 0.0] # should not have stresses with positive displacement 
    
    #return np.sum(residual**2.0) + 1.0*np.sum(negative**2.0) + residual.shape[0]*(u[scp.afull_idx_fine]-sigma_ext)**2.0
    return 1.0*np.sum(residual**2.0) + 1.0*np.sum(negative**2.0) + 1.0*np.sum(displaced**2.0) #  + 10*residual.shape[0]*(u[scp.afull_idx_fine]-sigma_ext)**2.0 




def calc_contact(scp,sigma_ext):

    #iniguess=np.arange(scp.afull_idx+1,dtype='d')/(scp.afull_idx+1) * sigma_ext  # This iniguess was for u
    
    # now we use du_da_shortened
    # which is afull_idx_fine+1 numbers representing
    # the distributed stress concentration
    # ... Note that the distributed stress concentration indices
    # have no effect to the left of the closure point
    
    # sigma_yield = scp.sigma_yield  # not a parameter yet... just use infinity for now
    sigma_yield=np.inf
    
    #crack_model=scp.crack_model  # not a parameter yet...
    #crack_model = ModeI_throughcrack_CODformula(scp.E)

    if sigma_ext > 0: # Tensile

        # closure_index is used to disallow any stresses or stress concentration
        # added to region that was open prior to applying load
        closure_index = np.where(scp.sigma_closure[:(scp.afull_idx)]!=0.0)[0][0]*scp.fine_refinement - 1
        pass
    else:
        closure_index = -1
        pass

    #du_da_shortened_iniguess=np.ones(scp.afull_idx_fine+1,dtype='d')*(1.0/(scp.afull_idx+1)) * sigma_ext/scp.dx_fine  #
    du_da_shortened_iniguess=np.ones(scp.afull_idx_fine+1-(closure_index+1),dtype='d')*(1.0/(scp.afull_idx+1))* sigma_ext/scp.dx_fine  #


    def load_constraint_fun(du_da_shortened):

        du_da = np.concatenate((np.zeros(closure_index+1,dtype='d'),du_da_shortened,np.zeros(scp.xsteps*scp.fine_refinement - scp.afull_idx_fine - 2 ,dtype='d')))        
        

        
        return (np.cumsum(du_da)[scp.afull_idx_fine]*scp.dx_fine)-sigma_ext  # !!!*** should go to afull_idx_fine+1??? 
        
    load_constraint = { "type": "eq",
                        "fun": load_constraint_fun }
        

    
    if sigma_ext > 0: # Tensile

        #nonnegative_constraint = scipy.optimize.NonlinearConstraint(lambda du_da: du_da,0.0,np.inf)
        nonnegative_constraint = { "type": "ineq",
                                   "fun": lambda du_da_shortened: du_da_shortened }

        

        
        # Calculate x

        
        
        res = scipy.optimize.minimize(soft_closure_goal_function,du_da_shortened_iniguess,args=(scp,sigma_ext,closure_index),
                                      constraints = [ load_constraint ], #[ nonnegative_constraint, load_constraint ],
                                      method="SLSQP",
                                      options={"eps": 10000.0,
                                               "maxiter": 100000,
                                               "ftol": scp.afull_idx_fine*(np.abs(sigma_ext)+20e6)**2.0/1e19})
        #res = scipy.optimize.minimize(goal_function,du_da_shortened_iniguess,method='nelder-mead',options={"maxfev": 15000})
        if not res.success: # and res.status != 4:
            # (ignore incompatible constraint, because our constraints are
            # compatible by definition, and scipy 1.2 seems to diagnose
            # this incorrectly... should file bug report)
            print("minimize error %d: %s" % (res.status,res.message))
            import pdb
            pdb.set_trace()
            pass
        
        du_da_shortened=res.x
        du_da = np.concatenate((np.zeros(closure_index+1,dtype='d'),du_da_shortened,np.zeros(scp.xsteps*scp.fine_refinement - scp.afull_idx_fine - 2 ,dtype='d')))        


        pass
    else:
        # Compressive or zero load
        #nonpositive_constraint = scipy.optimize.NonlinearConstraint(lambda du_da: du_da,0.0,np.inf)
        #nonpositive_constraint = { "type": "ineq",
        #                           "fun": lambda du_da_shortened: -du_da_shortened }

        
        #def crack_open_constraint_fun(uniform_du_da_shortened):
        #    """ Represents that no stress is permitted up
        #       through generated closure_index"""
        #    
        #    uniform_load=uniform_du_da_shortened[0]*self.dx_fine  # Uniform stress in Pa across crack
        #    uniform_load = 0.0
            
        #    du_da = np.concatenate((uniform_du_da_shortened[1:],np.zeros(self.xsteps*self.fine_refinement - self.afull_idx_fine - 2 ,dtype='d')))        
            
        #    (sigma_contact,displacement,closure_index_generated,du_da_corrected) = sigmacontact_from_displacement(self,uniform_load,du_da,dsigmaext_dxt_hardcontact_interp)
            
        #    return np.concatenate((displacement[:(closure_index_generated+1)],-displacement[(closure_index_generated+1):]))  # require displacement >= 0 in open region and <= 0 in closed region

        #crack_open_constraint = { "type": "ineq",
        #                          "fun": crack_open_constraint_fun }
        

        
        constraints = [ load_constraint ]
        
        #if sigma_ext < 0.0:
        #    # if we are applying compressive external load,
        #    # net shift of concentration should always be compressive
        #    constraints.append(nonpositive_constraint)
        #    pass

        res = scipy.optimize.minimize(soft_closure_goal_function,du_da_shortened_iniguess,args=(scp,sigma_ext,closure_index),
                                      constraints = constraints,
                                      method="SLSQP",
                                      options={"eps": 10000.0,
                                               "maxiter": 100000,
                                               "ftol": scp.afull_idx_fine*(np.abs(sigma_ext)+20e6)**2.0/1e19})
        
        #res = scipy.optimize.minimize(goal_function,du_da_shortened_iniguess,method='nelder-mead',options={"maxfev": 15000})
        if not res.success: #  and res.status != 4:
            # (ignore incompatible constraint, because our constraints are
            # compatible by definition, and scipy 1.2 seems to diagnose
            # this incorrectly... should file bug report)
            print("minimize error %d: %s" % (res.status,res.message))
            import pdb
            pdb.set_trace()
            pass
        
        du_da_shortened=res.x
        du_da = np.concatenate((du_da_shortened,np.zeros(scp.xsteps*scp.fine_refinement - scp.afull_idx_fine - 2 ,dtype='d')))

        (contact_stress,displacement) = sigmacontact_from_displacement(scp,du_da)
        

        

        #sys.modules["__main__"].__dict__.update(globals())
        #sys.modules["__main__"].__dict__.update(locals())
        #raise ValueError("Foo!")
        pass


    
    (contact_stress,displacement) = sigmacontact_from_displacement(scp,du_da)


    
    return (du_da,contact_stress,displacement)
    









def soft_closure_plots(scp,du_da,titleprefix=""):
    from matplotlib import pyplot as pl
    #pl.rc('text', usetex=True) # Support greek letters in plot legend
    
    #last_closureidx = np.where(x_bnd >= a)[0][0]

    #  constraint that integral of du/da must equal sigma_ext
    # means that last value of u should match sigma_ext
    
    # sigmacontact is positive compression

    #du_da = calc_du_da(u,scp.dx_fine)


    
    (from_displacement,displacement) = sigmacontact_from_displacement(scp,du_da)
    from_stress = sigmacontact_from_stress(scp,du_da)


    
    
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
    pl.title(titleprefix+"sigmacontact")

    u = np.cumsum(du_da)*scp.dx_fine
    # u nominally on position basis x_fine+dx_fine/2.0

    pl.figure()
    pl.clf()
    pl.plot(scp.x_fine*1e3,du_da/1e6,'-')
    pl.grid()
    pl.title(titleprefix+"distributed stress concentration derivative (corrected)\ntotal load=%g" % (u[scp.afull_idx_fine]))
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
    pl.title(titleprefix+"displacement")
    pass
    


