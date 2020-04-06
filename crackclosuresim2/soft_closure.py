import copy
import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan,exp
import numpy.random
from scipy.special import erf
import scipy.optimize
import scipy as sp


from . import solve_normalstress
from . import inverse_closure,crackopening_from_tensile_closure
from . import indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay
from . import ModeI_Beta_COD_Formula
from . import ModeI_throughcrack_CODformula
from . import Tada_ModeI_CircularCrack_along_midline
from .soft_closure_accel import initialize_contact_goal_function_with_gradient_accel
from .soft_closure_accel import soft_closure_goal_function_with_gradient_accel
from .soft_closure_accel import initialize_contact_goal_function_with_gradient_normalized_accel
from .soft_closure_accel import soft_closure_goal_function_with_gradient_normalized_accel


# IMPORTANT:
# du_da terminology

# Full length: starts with element corresponding to uniform load, no displacement. Then
# followed by distributed stress concentration starting at x center offset by dx/2 from
# crack center, may extend well beyond crack tip. Only meaningful up through crack tip.
#
# du_da_short: Same as full length but does not extend significantly beyond crack tip. Goes up to afull_idx_fine+1 as last element (afull_idx_fine+2 elements total)
#
# du_da_shortened: representation of du_da_short consisting of first element,
#  closure_index+1 implicit zeros, followed by afull_idx_fine-closure_index
# remaining elements from du_da_short. Total number of (non-implicit)
#  elements: afull_idx_fine-closure_index+1


def duda_short__from_duda_shortened(du_da_shortened,closure_index):
    du_da_short = np.concatenate(((du_da_shortened[0],),np.zeros(closure_index+1,dtype='d'),du_da_shortened[1:]))  # short version goes only up to afull_idx_fine+1 as last element (afull_idx_fine+2 elements total)
    return du_da_short

def duda_shortened__from_duda(du_da,afull_idx,closure_index):
    """Evaluate du_da_shortened from duda or duda_short"""

    assert((du_da[1:(closure_index+2)]==0).all())
    du_da_shortened_ext = np.concatenate(((du_da[0],),du_da[(closure_index+2):]))
    du_da_shortened = du_da_shortened_ext[:(afull_idx+2)]
    return du_da_shortened
    
    
def duda__from_duda_shortened(scp,du_da_shortened,closure_index):
    du_da = np.concatenate(((du_da_shortened[0],),np.zeros(closure_index+1,dtype='d'),du_da_shortened[1:],np.zeros(scp.xsteps - scp.afull_idx - 2, dtype='d')))  # full version goes all the way up to and including scp.xsteps. 
    return du_da


class sc_params(object):
    # soft closure parameters
    crack_model = None # crackclosuresim ModeI_crack_model object  (for now should be derived from ModeI_Beta_COD_Formula)
    Lm = None # L*m
    xmax = None
    xsteps = None
    a_input = None
    a = None
    x_bnd = None
    dx = None 
    x = None
    afull_idx = None
    #fine_refinement = None
    #dx_fine = None
    #xbnd_fine = None
    #x_fine = None
    #afull_idx_fine = None
    sigma_closure = None
    #sigma_closure_interp = None
    crack_initial_opening = None
    #crack_initial_opening_interp = None

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

    def save_debug_pickle(self,sigma_ext,du_da,closure_index,du_da_normalization,goal_function_normalization,sigma_closure=None,load_constraint_fun_normalization=None,filename=None):
        import os
        import os.path
        import tempfile
        import pickle
        import inspect
        import copy
          
        scp_copy=copy.deepcopy(self)
  
        caller_lineno = inspect.getframeinfo(inspect.stack()[1][0]).lineno

        if filename is None:
            filename=os.path.join(tempfile.gettempdir(),"scdebug%d_line_%d.pickle" % (os.getpid(),caller_lineno))
            pass
            
        picklefh=open(filename,"wb")

        (contact_stress_from_displacement,displacement,contact_stress_from_displacement_gradient,displacement_gradient) = sigmacontact_from_displacement(self,du_da,closure_index_for_gradient=closure_index)
        (contact_stress_from_stress,contact_stress_from_stress_gradient) = sigmacontact_from_stress(self,du_da,closure_index_for_gradient=closure_index)


        scp_copy.crack_model = None
        to_pickle = {
            "scp": scp_copy,
            "E": self.crack_model.E,
            "nu": self.crack_model.nu,
            "crack_model_class": self.crack_model.__class__.__name__,
            "sigma_ext": sigma_ext,

            "du_da": du_da,
            "closure_index": closure_index,
            "contact_stress_from_stress": contact_stress_from_stress,
            "contact_stress_from_stress_gradient": contact_stress_from_stress_gradient,
            "contact_stress_from_displacement": contact_stress_from_displacement,
            "displacement": displacement, 
            "contact_stress_from_displacement_gradient": contact_stress_from_displacement_gradient,
            "displacement_gradient": displacement_gradient, 

            "du_da_normalization": du_da_normalization,
            "goal_function_normalization": goal_function_normalization,
        }

        if sigma_closure is not None:
            to_pickle["sigma_closure"]=sigma_closure
            pass

        if load_constraint_fun_normalization is not None:
            to_pickle["load_constraint_fun_normalization"]=load_constraint_fun_normalization
            pass
            

        pickle.dump(to_pickle,picklefh)
        picklefh.close()
        pass


    def setcrackstate(self,sigma_closure,crack_initial_opening):
        self.sigma_closure=sigma_closure
        #self.sigma_closure_interp=scipy.interpolate.interp1d(self.x,self.sigma_closure,kind="linear",fill_value="extrapolate")(self.x_fine)

        
        self.crack_initial_opening=crack_initial_opening
        #self.crack_initial_opening_interp=scipy.interpolate.interp1d(self.x,self.crack_initial_opening,kind="linear",fill_value="extrapolate")(self.x_fine)
        pass

    def initialize_contact(self,sigma_closure,crack_initial_opening):

        assert(np.all(sigma_closure >= 0.0)) # given sigma_closure should not have any tensile component
        
        #self.crack_initial_opening=copy.copy(crack_initial_opening)
        self.crack_initial_opening=np.zeros(self.x.shape[0],dtype='d')
        #self.crack_initial_opening_interp=np.zeros(self.x_fine.shape[0],dtype='d')
        self.sigma_closure=np.zeros(self.x.shape[0],dtype='d')
        #self.sigma_closure_interp=np.zeros(self.x_fine.shape[0],dtype='d')

        ## closure index is the last point prior to closure
        closure_index = np.where(sigma_closure[:(self.afull_idx)]!=0.0)[0][0] - 1
        #if closure_index < 0:
        #    closure_index = None
        #    pass
        assert(np.all(sigma_closure[:(closure_index+1)]==0.0))

        
        
        # ***!!!! will need to change this integration for a different
        # crack model!!!***
        #thickness=2.0e-3 # ***!!! should get from crack model... but we divide it out so it really doesn't matter
        #sigma_closure_avg_stress = np.sum(sigma_closure[sigma_closure > 0.0])*self.dx*thickness/(self.a*thickness)
        
        # now we use du_da_shortened
        # which is  afull_idx_fine+2 numbers representing
        # a uniform load followed by the distributed stress concentration
        # ... Note that the distributed stress concentration indices
        # have no effect to the left of the closure point 

        #sigma_closure_interp = scipy.interpolate.interp1d(self.x,sigma_closure,kind="linear",fill_value="extrapolate")(self.x_fine)


        # du_da_shortened has the initial open region removed, as we
        # disallow any concentration in this open region --
        # crack should not close. 
        du_da_shortened_iniguess=np.zeros(self.afull_idx+2-(closure_index+1),dtype='d')*(1.0/(self.afull_idx+1)) # * (-sigma_closure_avg_stress)/self.dx_fine  #
        if closure_index==-1:
            du_da_shortened_iniguess[0] = -sigma_closure[0]/self.dx
            pass
        
        du_da_shortened_iniguess[1:(self.afull_idx+2-(closure_index+1))] = -sigma_closure[(closure_index+1):(self.afull_idx+1)]/self.dx

        
        
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
        

               
        # Check gradient
    
        #try: 
        grad_eval = initialize_contact_goal_function_with_gradient(du_da_shortened_iniguess,self,sigma_closure,closure_index)[1]
        #    pass
        #except:
        #    import pdb
        #    import traceback
        #    exctype, value, tb = sys.exc_info()
        #    traceback.print_exc()
        #    pdb.post_mortem(tb)
        #    pass

        grad_approx = scipy.optimize.approx_fprime(du_da_shortened_iniguess,lambda x: initialize_contact_goal_function_with_gradient(x,self,sigma_closure,closure_index)[0],(abs(np.mean(sigma_closure))+10e6)/(self.dx*self.afull_idx)/3e3)
        grad_sumsquareddiff = np.sqrt(np.sum((grad_eval-grad_approx)**2.0))
        grad_sumsquared = np.sqrt(np.sum(grad_eval**2.0))
    
        
        #print("grad_sumsquared=%g; grad_sumsquareddiff=%g" % (grad_sumsquared,grad_sumsquareddiff))
        
        if (grad_sumsquareddiff/grad_sumsquared >= 1e-4):
            self.save_debug_pickle(sigma_ext,duda__from_duda_shortened(self,du_da_shortened_iniguess,closure_index),closure_index,du_da_normalization=None,goal_function_normalization=None,sigma_closure=sigma_closure)
            raise ValueError("Grad error too high: FAIL grad_sumsquared=%g; grad_sumsquareddiff=%g" % (grad_sumsquared,grad_sumsquareddiff))
            
        #assert(grad_sumsquareddiff/grad_sumsquared < 1e-4) # NOTE: In the obscure case where our initial guess is at a relative minimum, this might fail extraneously

        # check accelerated gradient
        grad_eval_accel = initialize_contact_goal_function_with_gradient_accel(du_da_shortened_iniguess,self,sigma_closure,closure_index)[1]
        grad_sumsquareddiff_accel = np.sqrt(np.sum((grad_eval_accel-grad_approx)**2.0))
        grad_sumsquared_accel = np.sqrt(np.sum(grad_eval_accel**2.0))
        
        assert(grad_sumsquareddiff_accel/grad_sumsquared_accel < 1e-4) # NOTE: In the obscure case where our initial guess is at a relative minimum, this might fail extraneously
        

        constraints = [] #[ load_constraint, crack_open_constraint ]
        
        # Allow total iterations to be broken into pieces separated by failures with minimize error 9 (Iteration limit exceeded)
        # (for some reason, restarting the minimizer where it left off seems to help get it to the goal)
        total_maxiter=10000000
        niter = 0
        #epsval1 = 50e6/self.a/5000.0
        #epsval2 = np.max(np.abs(sigma_closure))/self.a/5000.0
        #epsval = max(epsval1,epsval2)
        epsval=1e-5
        epsvalscaled = epsval
        terminate=False
        starting_value=du_da_shortened_iniguess
        goal_stress_fit_error_pascals = 150e3 # Amount of stress error to allow in fitting process. If we have more than this we keep trying to minimize
        goal_residual = (goal_stress_fit_error_pascals**2.0)*self.afull_idx
        
        du_da_normalization = max(np.max(np.abs(sigma_closure)),10e6)/self.dx
        goal_function_normalization = goal_residual

        while niter < total_maxiter and not terminate: 
            this_niter=10000
            res = scipy.optimize.minimize(initialize_contact_goal_function_with_gradient_normalized_accel,starting_value/du_da_normalization,args=(self,sigma_closure,closure_index,du_da_normalization,goal_function_normalization), # was initialize_contact_goal_function_accel
                                          constraints = constraints,
                                          method="SLSQP",
                                          jac=True,
                                          options={"eps": epsvalscaled,
                                                   "maxiter": this_niter,
                                                   "ftol": 1e-12}) #self.afull_idx*(abs(np.mean(sigma_closure))+20e6)**2.0/1e14})
            
            res_fun_denormalized = res.fun*goal_function_normalization

            # Variables accessible from gdb
            status = res.status
            fun_str = str(res_fun_denormalized)

            if res_fun_denormalized <= goal_residual:
                terminate=True
                pass
 
            if res.status != 9 and res.status != 7:  # anything but reached iteration limit or eps increase
                if res_fun_denormalized <= goal_residual or res.nit==0:
                    terminate=True
                    pass 
                else:
                    ## ... otherwise keep trying!
                    #if res.nit==1:
                    #    import pdb
                    #    pdb.set_trace()
                    #    pass
                    epsvalscaled = epsval # reset eps to nominal value
                    starting_value = res.x*du_da_normalization # Next iteration starts where this one left off
                    pass
                pass
            elif res.status==7:
                # Rank-deficient equality constraint subproblem HFTI 
                # Generally indicates too fine epsilon...
                if epsvalscaled/epsval < 10:
                    epsvalscaled *= 2 
                    starting_value = res.x*du_da_normalization # Next iteration starts where this one left off
                    pass
                else:
                    print("WARNING: initialize_contact(): repeated rank-deficient equality constraint subproblem HFTI... Terminating!\n")
                    sys.stdout.flush()
                    terminate=True  # Don't allow eps to grow too much
                    pass
                pass
            else:
                epsvalscaled = epsval # reset eps to nominal value
                starting_value = res.x*du_da_normalization # Next iteration starts where this one left off
                pass
            niter += this_niter #res.nit
            pass

        if niter >= total_maxiter and res_fun_denormalized > goal_residual:
            print("soft_closure/initialize_contact: WARNING Maximum number of iterations (%d) reached and residual (%g) exceeds goal (%g)" % (total_maxiter,res_fun_denormalized,goal_residual))
            sys.stdout.flush()
            self.save_debug_pickle(sigma_ext,duda__from_duda_shortened(self,res.x*du_da_normalization,closure_index),closure_index,du_da_normalization,goal_function_normalization,sigma_closure=sigma_closure)
            pass

        
        if res_fun_denormalized > goal_residual and not res.success: #  and res.status != 4:
            # (ignore incompatible constraint, because our constraints are
            # compatible by definition, and scipy 1.2 seems to diagnose
            # this incorrectly... should file bug report)
            self.save_debug_pickle(sigma_ext,duda__from_duda_shortened(self,res.x*du_da_normalization,closure_index),closure_index,du_da_normalization,goal_function_normalization,sigma_closure=sigma_closure)
            print("minimize error %d: %s" % (res.status,res.message))
            sys.stdout.flush()
            import pdb
            pdb.set_trace()
            pass

        # Verify proper operation of accelerated code
        (slowcalc,slowcalc_grad) = initialize_contact_goal_function_with_gradient(res.x*du_da_normalization,self,sigma_closure,closure_index)
        (fastcalc,fastcalc_grad) = initialize_contact_goal_function_with_gradient_accel(res.x*du_da_normalization,self,sigma_closure,closure_index)

        if abs((slowcalc-fastcalc)/slowcalc) >= 1e-4 and (slowcalc > goal_residual/100.0 or fastcalc > goal_residual/100.0):
            #from VibroSim_Simulator.function_as_script import scriptify
            #(slowcalc2,slowcalc2_grad) = scriptify(initialize_contact_goal_function_with_gradient)(res.x*du_da_normalization,self,sigma_closure,closure_index)
            raise ValueError("Accelerated initialize contact calculation mismatch: %g vs %g" % (slowcalc,fastcalc))
            
        #assert(abs((slowcalc-fastcalc)/slowcalc) < 1e-4)
        
        du_da_shortened=res.x*du_da_normalization
        #du_da = np.concatenate(((du_da_shortened[0],),np.zeros(closure_index+1,dtype='d'),du_da_shortened[1:],np.zeros(self.xsteps - self.afull_idx - 2 ,dtype='d')))
        du_da = duda__from_duda_shortened(self,res.x*du_da_normalization,closure_index)
        
        (contact_stress,displacement) = sigmacontact_from_displacement(self,du_da)

        from_stress = sigmacontact_from_stress(self,du_da)

        # displacement = crack_initial_opening - (sigma_closure_stored/Lm)^(2/3) + displacement_due_to_distributed_stress_concentrations
        # sigmacontact_from_displacement = 0 when displacement > 0
        # sigmacontact_from_displacement = (-displacement)^(3/2)*Lm when displacement < 0
        
        # we made sigmacontact_from_stress match sigma_closure...

        # So let's set sigmacontact_from_displacement equal and solve for displacement
        # sigma_closure = (-displacement)^(3/2)*Lm
        # (-displacement)^(3/2) = sigma_closure/Lm = 
        # displacement = -(sigma_closure/Lm)^(2/3)
        # crack_initial_opening + displacement_due_to_distributed_stress_concentrations = -(sigma_closure/Lm)^(2/3)
        # crack_initial_opening = -displacement_due_to_distributed_stress_concentrations -(sigma_closure/Lm)^(2/3)

        # To evaluate modeled residual stressing operation starting
        # at zero initial opening and zero closure, 
        # stop here, and run soft_closure_plots()
        self.crack_initial_opening = crack_initial_opening -displacement - (sigma_closure/self.Lm)**(2.0/3.0)  # Through this line we have evaluated an initial opening displacement
        #self.crack_initial_opening_interp=scipy.interpolate.interp1d(self.x,self.crack_initial_opening,kind="linear",fill_value="extrapolate")(self.x_fine)

        
        #sys.modules["__main__"].__dict__.update(globals())
        #sys.modules["__main__"].__dict__.update(locals())
        #raise ValueError("Foo!")

        # Now we apply our sigma_closure
        (contact_stress,displacement) = sigmacontact_from_displacement(self,du_da)

        from_stress = sigmacontact_from_stress(self,du_da)

        # This becomes our new initial state
        self.crack_initial_opening = displacement + (sigma_closure/self.Lm)**(2.0/3.0) # we add sigma_closure displacement back in because sigmacontact_from_displacement() will subtract it out
        #self.crack_initial_opening_interp=scipy.interpolate.interp1d(self.x,self.crack_initial_opening,kind="linear",fill_value="extrapolate")(self.x_fine)

        self.sigma_closure = sigma_closure
        #self.sigma_closure_interp=scipy.interpolate.interp1d(self.x,self.sigma_closure,kind="linear",fill_value="extrapolate")(self.x_fine)
        du_da[:]=0.0
        
        
        #sys.modules["__main__"].__dict__.update(globals())
        #sys.modules["__main__"].__dict__.update(locals())
        #raise ValueError("Foo!")
    
        pass
    


    
    @classmethod
    def fromcrackgeom(cls,crack_model,xmax,xsteps,a_input,fine_refinement,Lm):
        # x_bnd represents x coordinates of the boundaries of
        # each mesh element
        assert(fine_refinement==1)
        x_bnd=np.linspace(0,xmax,xsteps,dtype='d')
        dx=x_bnd[1]-x_bnd[0]
        x = (x_bnd[1:]+x_bnd[:-1])/2.0  # x represents x coordinates of the centers of each mesh element
        
        afull_idx=np.argmin(np.abs(a_input-x_bnd))
        a = x_bnd[afull_idx]


        return cls(crack_model=crack_model,
                   xmax=xmax,
                   xsteps=xsteps,
                   a_input=a_input,
                   Lm=Lm,
                   x_bnd=x_bnd,
                   dx=dx,
                   x=x,
                   afull_idx=afull_idx,
                   a=a)
        pass
    pass



def initialize_contact_goal_function_with_gradient(du_da_shortened,scp,sigma_closure,closure_index):
    """ NOTE: This should be kept identical functionally to initialize_contact_goal_function_accel in soft_closure_accel.pyx"""

    #du_da = np.concatenate((np.zeros(closure_index+1,dtype='d'),du_da_shortened,np.zeros(scp.xsteps*scp.fine_refinement - scp.afull_idx_fine - 2 ,dtype='d')))
    du_da_short = duda_short__from_duda_shortened(du_da_shortened,closure_index)

    #last_closureidx = np.where(x_bnd >= a)[0][0]

    #  constraint that integral of du/da must equal sigma_ext
    # means that last value of u should match sigma_ext
    
    # sigmacontact is positive compression
    
    #(from_displacement,displacement) = sigmacontact_from_displacement(scp,du_da_short)

    
    (from_stress,from_stress_gradient) = sigmacontact_from_stress(scp,du_da_short,closure_index_for_gradient=closure_index)
    
    # elements of residual have units of stress^2
    residual = (sigma_closure[:(scp.afull_idx+1)]-from_stress)
    dresidual = - from_stress_gradient


    goal_function = np.sum(residual**2.0) # + 1.0*np.sum(negative**2.0) #  + 10*residual.shape[0]*(u[scp.afull_idx_fine]-sigma_ext)**2.0 

    gradient = np.sum(2.0*residual[:,np.newaxis]*dresidual,axis=0)
    
    return (goal_function,gradient)




def initialize_contact_goal_function_with_gradient_normalized(du_da_shortened_normalized,scp,sigma_closure,closure_index,du_da_normalization,goal_function_normalization):
    (goal_function,gradient)=initialize_contact_goal_function_with_gradient(du_da_shortened_normalized*du_da_normalization,scp,sigma_closure,closure_index)
    goal_function_normalized = goal_function / goal_function_normalization
    
    
    # d_gfn/d_dudasn = d_gfn/d_gf * d_gf/d_dudas * d_dudas/d_dudasn  
    #  ... where d_gfn/d_gf = 1/goal_function_normalization
    #  dudasn = dudas/du_da_normalization so d_dudasn/d_dudas = 1/du_da_normalization so d_dudas/d_dudasn = du_da_normalization
    # so d_gfn/d_dudasn = du_da_normalization/goal_function_normalization * d_gf/d_dudas
    gradient_normalized = (du_da_normalization/goal_function_normalization) * gradient
    return (goal_function_normalized,gradient_normalized)

# du/da defined on x positions... represents distributed stress concentrations
def sigmacontact_from_displacement(scp,du_da,closure_index_for_gradient=None):
    """ NOTE: This should be kept functionally identical to sigmacontact_from_displacement() in soft_closure_accel_ops.h"""
    # 
    da = scp.dx # same step size

    x = scp.x[:(du_da.shape[0]-1)]

    # integral starts at a0, where sigma_closure > 0 (compressive)
    #first_closureidx = np.where(sigma_closure >0)[0][0]
    # first_closureidx can be thought of as index into x_bnd
    #    last_closureidx = np.where(x_bnd >= a)[0][0]
    

    
    displacement = scp.crack_initial_opening[:(du_da.shape[0]-1)] - (scp.sigma_closure[:(du_da.shape[0]-1)]/scp.Lm)**(2.0/3.0)

    #displacement=scipy.interpolate.interp1d(scp.x,displacement_coarse,kind="linear",fill_value="extrapolate")(x)

    if closure_index_for_gradient is not None:
        # displacement gradient axis zero is position along crack, axis one is du_da_shortened element 
        displacement_gradient = np.zeros((du_da.shape[0]-1,scp.afull_idx-closure_index_for_gradient+1),dtype='d')
        pass

    # !!!*** Possibly calculation should start at scp.afull_idx_fine instead of
    # afull_idx_fine-1.
    # Then residual and average in goal_function should consider the full range (including afull_idx_fine
    # But this causes various problems:
    #   * displacement assertion failed below -- caused by sigma_closure not having a value @ afull_idx_fine
    #   * Convergence failure 'Positive directional derivative for linesearch' ... possibly not a problem (need to adjust tolerances?)

    if isinstance(scp.crack_model,ModeI_throughcrack_CODformula):
    
        for aidx in range(scp.afull_idx,-1,-1):
            
            #assert(sigma_closure[aidx] > 0)
            #   in next line: sqrt( (a+x) * (a-x) where x >= 0 and
            #   throw out where x >= a
            # diplacement defined on x
            displacement[:aidx] += (4.0/scp.crack_model.Eeff)*du_da[aidx+1]*np.sqrt((x[aidx]+x[:aidx])*(x[aidx]-x[:aidx]))*da
            
            # Add in the x=a position
            # Here we have the integral of (4/E)*(du/da)*sqrt( (a+x)* (a-x) )da
            # as a goes from x[aidx] to x[aidx]+da/2
            # ... treat du/da and sqrt(a+x) as constant:
            #  (4/E)*(du/da)*sqrt(a+x) * integral of sqrt(a-x) da
            # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * (  (x[aidx]+da/2-x[aidx])^(3/2) - (x[aidx]-x[aidx])^(3/2) )
            # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * ( (da/2)^(3/2) )
            displacement[aidx] += (4.0/scp.crack_model.Eeff)*du_da[aidx+1]*np.sqrt(2.0*x[aidx])*(da/2.0)**(3.0/2.0)

            if closure_index_for_gradient is not None:
                if aidx+1 >= closure_index_for_gradient+2:
                    du_da_shortened_index = aidx -  closure_index_for_gradient
                    displacement_gradient[:aidx,du_da_shortened_index] += (4.0/scp.crack_model.Eeff)*np.sqrt((x[aidx]+x[:aidx])*(x[aidx]-x[:aidx]))*da
                    displacement_gradient[aidx,du_da_shortened_index] += (4.0/scp.crack_model.Eeff)*np.sqrt(2.0*x[aidx])*(da/2.0)**(3.0/2.0)
                    pass
                
                pass
            
            
            pass
        pass
    elif isinstance(scp.crack_model,Tada_ModeI_CircularCrack_along_midline):
        for aidx in range(scp.afull_idx,-1,-1): 
            #assert(sigma_closure[aidx] > 0)
            #   in next line: sqrt( (a+x) * (a-x) where x >= 0 and
            #   throw out where x >= a
            # displacement defined on x_fine
            # NOTE: displacement is 2* usual formula because we are concerned with total distance between both crack faces

            # Tada turns out to have sqrt(xt^2-x^2) === sqrt((xt-x)(xt+x)) form
            # as throughcrack, so same integral calculation applies.
            # We just change the leading factors
            displacement[:aidx] += (8.0*(1.0-scp.crack_model.nu**2.0)/(np.pi*scp.crack_model.E))*du_da[aidx+1]*np.sqrt((x[aidx]+x[:aidx])*(x[aidx]-x[:aidx]))*da
            # Add in the x=a position
            # Here we have the integral of (4/E)*(du/da)*sqrt( (a+x)* (a-x) )da
            # as a goes from x[aidx] to x[aidx]+da/2
            # ... treat du/da and sqrt(a+x) as constant:
            #  (4/E)*(du/da)*sqrt(a+x) * integral of sqrt(a-x) da
            # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * (  (x[aidx]+da/2-x[aidx])^(3/2) - (x[aidx]-x[aidx])^(3/2) )
            # = (4/E)*(du/da)*sqrt(a+x) * (2/3) * ( (da/2)^(3/2) )
            displacement[aidx] += (8.0*(1.0-scp.crack_model.nu**2.0)/(np.pi*scp.crack_model.E))*du_da[aidx+1]*np.sqrt(2.0*x[aidx])*(da/2.0)**(3.0/2.0)

            if closure_index_for_gradient is not None:
                if aidx+1 >= closure_index_for_gradient+2:
                    du_da_shortened_index = aidx - closure_index_for_gradient
                    displacement_gradient[:aidx,du_da_shortened_index] += (8.0*(1.0-scp.crack_model.nu**2.0)/(np.pi*scp.crack_model.E))*np.sqrt((x[aidx]+x[:aidx])*(x[aidx]-x[:aidx]))*da
                    displacement_gradient[aidx,du_da_shortened_index] += (8.0*(1.0-scp.crack_model.nu**2.0)/(np.pi*scp.crack_model.E))*np.sqrt(2.0*x[aidx])*(da/2.0)**(3.0/2.0)
                    pass
                
                pass
            

            
            pass

        
        pass
    else:
        raise ValueError("Unsupported crack model for soft_closure")
    
    # Displacement calculated....
    # Sigma contact exists where displacement is negative
    # sigmacontact is positive compression

    
    sigma_contact = np.zeros(x.shape[0],dtype='d')
    sigma_contact[displacement < 0.0] = ((-displacement[displacement < 0.0])**(3.0/2.0)) * scp.Lm

    if closure_index_for_gradient is not None:
        # sigma_contact gradient axis zero is position along crack, axis one is du_da_shortened element 
        sigma_contact_gradient = np.zeros((du_da.shape[0]-1,scp.afull_idx-closure_index_for_gradient+1),dtype='d')
        sigma_contact_gradient[displacement < 0.0,:] = -(3.0/2.0)*((-displacement[displacement < 0.0,np.newaxis])**(1.0/2.0))*scp.Lm*displacement_gradient[displacement < 0.0,:]
        pass
    

    
    if closure_index_for_gradient is None:
        return (sigma_contact,displacement) 
    else:
        return (sigma_contact,displacement,sigma_contact_gradient,displacement_gradient) 
    pass

def sigmacontact_from_stress(scp,du_da,closure_index_for_gradient=None):
    """ NOTE: This should be kept functionally identical to sigmacontact_from_stress() in soft_closure_accel_ops.h"""
    # sigmacontact is positive compression

    #sigma_closure_interp=scipy.interpolate.interp1d(scp.x,scp.sigma_closure,kind="linear",fill_value="extrapolate")(scp.x)
    
    #sigmacontact = copy.copy(sigma_closure)
    sigmacontact = copy.copy(scp.sigma_closure[:(du_da.shape[0]-1)])

    if closure_index_for_gradient is not None:
        # sigma_contact gradient axis zero is position along crack, axis one is du_da_shortened element 
        sigma_contact_gradient = np.zeros((du_da.shape[0]-1,scp.afull_idx-closure_index_for_gradient+1),dtype='d')
        pass
    
    da = scp.dx # same step size

    x = scp.x[:(du_da.shape[0]-1)]
    
    # integral starts at a0, where sigma_closure > 0 (compressive)
    # OR closure state with external load is compressive

    #first_closureidx = np.where(sigma_closure >0)[0][0]
    # first_closureidx can be thought of as index into x_bnd
    #last_closureidx = np.where(x_bnd >= a)[0][0]
   
    #du_da = calc_du_da(u,da)

    betaval = scp.crack_model.beta(scp.crack_model)

    sigmacontact -= du_da[0]*da # constant term 
    if closure_index_for_gradient is not None:
        sigma_contact_gradient[:,0] -= da
        pass
    
    for aidx in range(scp.afull_idx+1):
        #assert(sigma_closure[aidx] > 0)

        #sigmacontact[(aidx+1):] -= du_da[aidx+1]*((betaval/sqrt(2.0))*sqrt(x_fine[aidx]/(x_fine[(aidx+1):]-x_fine[aidx])) + 1.0)*da # + 1.0 represents stress when a large distance away from effective tip

        # !!!*** Could improve this approximation by using the explicit integration for more than just the
        # term on top of the crack tip ***!!! 
        
        a = x[aidx]
        r = (x[(aidx+1):]-a)
        r0 = scp.crack_model.r0_over_a(a)*a
        sigmacontact[(aidx+1):] -= du_da[aidx+1]*((sqrt(betaval)/sqrt(2.0))*sqrt(a/r)*( (r0)**2.0/(r + r0)**2.0 ) + 1.0)*da # + 1.0 represents stress when a large distance away from effective tip

        # Need to include integral 
        # of (du/da)[(1/sqrt(2))sqrt(a/(x-a))*(r0^2)/(r0 + (x-a))^2 + 1]da
        # as a goes from x[aidx]-da/2 to x[aidx]

        # Pulling out the constant term and leading factors, we have 
        # (du/da)(sqrt(beta)/sqrt(2))integral sqrt(a/(x-a))*(r0^2)/(r0 + (x-a))^2 + integral (du/da)da
        # with integrals as a goes from x[aidx]-da/2 to x[aidx]
        
        # The integral on the right is trivial. The solution to the integral on the left is given
        # by crackclosure.indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x,xt)
        
        sigmacontact[aidx] -= du_da[aidx+1]*(sqrt(betaval)/sqrt(2.0))*(indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(scp.crack_model,x[aidx],x[aidx])-indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(scp.crack_model,x[aidx],x[aidx]-da/2.0)) + du_da[aidx+1]*da/2.0 
        
        # = (du/da)*(1/sqrt(2))*sqrt(a)* [ (1/(2*sqrt(2))) * sqrt(r0) * {ln(r0) - ln(r0) + 2*atan(1) - 2*atan(1)} - [ (1/(2*sqrt(2))) * sqrt(r0) * {ln(-sqrt(2*r0*da/2) + r0 + da/2) - ln(sqrt(2*r0*da/2) + r0+da/2) + 2*atan(1-sqrt(2*(da/2)/r0)) - 2*atan(sqrt(2*(da/2)/r0)+1)} ]  + (du/da)*(da/2)
        # sigmacontact[aidx]  -= (du_da[aidx+1])*(sqrt(betaval)/sqrt(2.0))*sqrt(a)* ( (1.0/(2.0*sqrt(2))) * sqrt(r0) * (-(log(-sqrt(2*(r0)*da/2) + r0 + da/2) - log(sqrt(2*(r0)*da/2) + r0+da/2) + 2.0*arctan(1-sqrt(2*(da/2.0)/(r0))) - 2*arctan(sqrt(2*(da/2.0)/(r0))+1)))) + du_da[aidx+1]*da/2.0
        
        # print(" sigmacontact update: New: %g Old: %g VeryOld: %g VeryVeryOld: %g" % (du_da[aidx+1]*(sqrt(betaval)/sqrt(2.0))*(indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(scp.crack_model,x[aidx],x[aidx])-indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(scp.crack_model,x[aidx],x[aidx]-da/2.0)) + du_da[aidx+1]*da/2.0,(du_da[aidx+1])*(sqrt(betaval)/sqrt(2.0))*sqrt(a)* ( (1.0/(2*sqrt(2))) * sqrt(r0) * -(log(-sqrt(2*(r0)*da/2) + r0 + da/2) - log(sqrt(2*(r0)*da/2) + r0+da/2) + 2.0*arctan(1-sqrt(2*(da/2.0)/(r0))) - 2*arctan(sqrt(2*(da/2.0)/(r0))+1))),du_da[aidx+1]*(sqrt(betaval)/sqrt(2.0))*np.sqrt(x[aidx])*sqrt(np.pi*scp.crack_model.r0_over_a(a)*a)*erf(sqrt(da/(2.0*scp.crack_model.r0_over_a(a)*a))),du_da[aidx+1]*(sqrt(betaval)/sqrt(2.0))*np.sqrt(x[aidx])*2.0*sqrt(da/2.0)))

        #sigmacontact[aidx] -= (du_da[aidx+1]*(sqrt(betaval)/sqrt(2.0))*np.sqrt(x[aidx])*sqrt(np.pi*scp.crack_model.r0_over_a*a)*erf(sqrt(da/(2.0*scp.crack_model.r0_over_a*a))) + du_da[aidx+1]*da/2.0)
        
        # # *** OBSOLETE: 
        # # Need to include integral 
        # # of (du/da)[(1/sqrt(2))sqrt(a/(x-a))exp(-(x-a)/r0) + 1]da
        # # as a goes from x[aidx]-da/2 to x[aidx]
        # # approximate sqrt(x-a)*exp(-(x-a)/r0) as only a dependence
        # # (du/da)*(1/sqrt(2))*sqrt(a)*integral of sqrt(1/(x-a))*exp(-(x-a)/r0) da + integral of du/da da
        # # as a goes from x[aidx]-da/2 to x[aidx]
        # # = (du/da)(1/sqrt(2))*sqrt(a) * (-sqrt(pi*r0)*erf(sqrt(x-a)/sqrt(r0))) as a goes from x[aidx]-da/2 to x[aidx]  ...  + (du/da)(da/2)
        #
        # # = (du/da)(1/sqrt(2))*sqrt(a) * [ -sqrt(pi*r0)*erf(sqrt(x-x[aidx])/sqrt(r0)) + sqrt(pi*r0)*erf(sqrt(x-x[aidx]+da/2)/sqrt(r0)) ] + (du/da)(da/2)
        #
        # # = (du/da)(1/sqrt(2))*sqrt(a) * [ -sqrt(pi*r0)*erf(sqrt(0)/sqrt(r0)) + sqrt(pi*r0)*erf(sqrt(da/2)/sqrt(r0)) ] + (du/da)(da/2)
        # 
        # = (du/da)(1/sqrt(2))*sqrt(a) * [ sqrt(pi*r0)*erf(sqrt(da/2)/sqrt(r0)) ] + (du/da)(da/2)
        #
        # sigmacontact[aidx] -= (du_da[aidx+1]*(sqrt(betaval)/sqrt(2.0))*np.sqrt(x[aidx])*sqrt(np.pi*scp.crack_model.r0_over_a*a)*erf(sqrt(da/(2.0*scp.crack_model.r0_over_a*a))) + du_da[aidx+1]*da/2.0)
        
        ## OBSOLETE
        ## = (du/da)(1/sqrt(2))*sqrt(a) * (-2sqrt(x-x) + 2sqrt(x-a+da/2))  + (du/da)(da/2)
        ## = (1/sqrt(2))(du/da)*sqrt(a) * 2sqrt(da/2)) + (du/da)(da/2)

        ##sigmacontact[aidx] -= (du_da[aidx+1]*(sqrt(betaval)/sqrt(2.0))*np.sqrt(x_fine[aidx])*2.0*sqrt(da/2.0) + du_da[aidx+1]*da/2.0)

        if closure_index_for_gradient is not None:
            if aidx+1 >= closure_index_for_gradient+2:
                du_da_shortened_index = aidx - closure_index_for_gradient
                sigma_contact_gradient[(aidx+1):,du_da_shortened_index] -= ((sqrt(betaval)/sqrt(2.0))*sqrt(a/r)*( (r0)**2.0/(r + r0)**2.0 )  + 1.0)*da # + 1.0 represents stress when a large distance away from effective tip

                sigma_contact_gradient[aidx,du_da_shortened_index] -= (sqrt(betaval)/sqrt(2.0))*(indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(scp.crack_model,x[aidx],x[aidx])-indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(scp.crack_model,x[aidx],x[aidx]-da/2.0)) + da/2.0
        

                # # OBSOLETE
                # sigma_contact_gradient[(aidx+1):,du_da_shortened_index] -= ((sqrt(betaval)/sqrt(2.0))*sqrt(a/r)*exp(-r/(scp.crack_model.r0_over_a*a)) + 1.0)*da
                # sigma_contact_gradient[aidx,du_da_shortened_index] -= ( (sqrt(betaval)/sqrt(2.0))*np.sqrt(x[aidx])*sqrt(np.pi*scp.crack_model.r0_over_a*a)*erf(sqrt(da/(2.0*scp.crack_model.r0_over_a*a))) + da/2.0 )
                # sigma_contact_gradient[aidx,du_da_shortened_index] -= (sqrt(betaval)/sqrt(2.0))*sqrt(a)* ( (1.0/(2.0*sqrt(2))) * sqrt(r0) * (-(log(-sqrt(2*(r0)*da/2) + r0 + da/2) - log(sqrt(2*(r0)*da/2) + r0+da/2) + 2.0*arctan(1-sqrt(2*(da/2.0)/(r0))) - 2*arctan(sqrt(2*(da/2.0)/(r0))+1)))) + da/2.0

                pass
                
            pass

        pass

    if closure_index_for_gradient is None:
        return sigmacontact
    else:
        return (sigmacontact,sigma_contact_gradient)
    pass

#def solve_sigmacontact(sigma_ext):
    # Contraints:
    #  * integral of du/da must equal sigma_ext
    #  * sigmacontact_from_stress = sigmacontact_from_displacement

    ## Normalize each component for the solver

    #sigma_nominal = (np.sqrt(np.mean(sigma_closure**2.0)) + np.abs(sigma_ext))/2.0

def soft_closure_goal_function_with_gradient(du_da_shortened,scp,closure_index):
    """ NOTE: This should be kept identical functionally to soft_closure_goal_function_accel in soft_closure_accel.pyx"""
    # closure_index is used in tension to shorten du_da, disallowing any stresses or concentration to left of initial opening distance
    


    #du_da = np.concatenate((np.zeros(closure_index+1,dtype='d'),du_da_shortened,np.zeros(scp.xsteps*scp.fine_refinement - scp.afull_idx_fine - 2 ,dtype='d')))
    du_da_short = duda_short__from_duda_shortened(du_da_shortened,closure_index)

    

    
    #last_closureidx = np.where(x_bnd >= a)[0][0]

    #  constraint that integral of du/da must equal sigma_ext
    # means that last value of u should match sigma_ext
    
    # sigmacontact is positive compression
    
    (from_displacement,displacement,from_displacement_gradient,displacement_gradient) = sigmacontact_from_displacement(scp,du_da_short,closure_index_for_gradient=closure_index)
    #dfrom_displacement = 0 when displacement < 0; (-3/2)displacement**(1/2)*Lm*ddisplacement otherwise

    #u = np.cumsum(du_da)*scp.dx_fine
    # u nominally on position basis x_fine+dx_fine/2.0
    
    (from_stress,from_stress_gradient) = sigmacontact_from_stress(scp,du_da_short,closure_index_for_gradient=closure_index)
    
    # elements of residual have units of stress^2
    residual = (from_displacement-from_stress)
    dresidual = from_displacement_gradient - from_stress_gradient

    average = (from_displacement+from_stress)/2.0
    daverage = (from_displacement_gradient + from_stress_gradient)/2.0

    # We only worry about residual, negative, and displaced
    # up to the point before the last... why?
    # well the last point corresponds to the crack tip, which
    # CAN hold tension and doesn't have to follow the contact stress
    # law... so all these [:-1]'s represent that a stress concentration
    # at the crack tip is OK for our goal

    #negative = average[:-1][average[:-1] < 0]  # negative sigmacontact means tension on the surfaces, which is not allowed (except at the actual tip)!
    negative_except_at_tip = average < 0
    negative_except_at_tip[-1] = False

    negative = average[negative_except_at_tip]
    
    dnegative = daverage[negative_except_at_tip,:]

    #displaced = average[:-1][displacement[:-1] > 0.0] # should not have stresses with positive displacement
    displaced_except_at_tip = displacement > 0.0
    displaced_except_at_tip[-1] = False

    displaced = average[displaced_except_at_tip]
    
    ddisplaced = daverage[displaced_except_at_tip,:]
    
    goal_function =  1.0*np.sum(residual[:-1]**2.0) + 1.0*np.sum(negative**2.0) + 1.0*np.sum(displaced**2.0) 
    gradient = 1.0*np.sum(2.0*residual[:-1,np.newaxis]*dresidual[:-1,:],axis=0) + 1.0*np.sum(2.0*negative[:,np.newaxis]*dnegative,axis=0) + 1.0*np.sum(2.0*displaced[:,np.newaxis]*ddisplaced,axis=0)

    #print("grad_residual=%s" % (str(np.sum(2.0*residual[:-1,np.newaxis]*dresidual[:-1,:],axis=0))))
    
    #print("from_displacement_grad_residual_component=%s" % (str(np.sum(2.0*residual[:-1,np.newaxis]*from_displacement_gradient[:-1,:],axis=0))))

    #print("from_stress_grad_residual_component=%s" % (str(np.sum(2.0*residual[:-1,np.newaxis]*(-from_stress_gradient[:-1,:]),axis=0))))

    return (goal_function,gradient)


def soft_closure_goal_function_with_gradient_normalized(du_da_shortened_normalized,scp,closure_index,du_da_normalization,goal_function_normalization):
    (goal_function,gradient) = soft_closure_goal_function_with_gradient(du_da_shortened_normalized*du_da_normalization,scp,closure_index)
    goal_function_normalized = goal_function / goal_function_normalization

    # d_gfn/d_dudasn = d_gfn/d_gf * d_gf/d_dudas * d_dudas/d_dudasn  
    #  ... where d_gfn/d_gf = 1/goal_function_normalization
    #  dudasn = dudas/du_da_normalization so d_dudasn/d_dudas = 1/du_da_normalization so d_dudas/d_dudasn = du_da_normalization
    # so d_gfn/d_dudasn = du_da_normalization/goal_function_normalization * d_gf/d_dudas
    gradient_normalized = (du_da_normalization/goal_function_normalization) * gradient
    return (goal_function_normalized,gradient_normalized)

class CalcContactFailure(Exception):
    message = None

    def __init__(self,message):
        self.message=message
        pass

    def __str__(self):
        return "CalcContactFailure: %s" % (self.message)
    pass



def calc_contact(scp,sigma_ext):
    """
    return (du_da, # distributed stress concentration field along crack, in Pa/m, positive tensile 
            contact_stress,  # physical positive-compression contact stress between crack faces, Pa
            displacement)  # Physical displacement between crack surfaces; positive means no contact. 
                           # Negative means Hertzian soft contact stress is present given crack 
                           # parameters (scp) and externally applied load (sigma_ext, positive tensile). 
   
    n.b. displacement here is twice what you would get from solve_normalstress() for hard closure, 
         because solve_normalstress() gives you the displacement of each side whereas calc_contact() 
         gives you the dissplacement between the two sides! 
    """
    
    #iniguess=np.arange(scp.afull_idx+1,dtype='d')/(scp.afull_idx+1) * sigma_ext  # This iniguess was for u
    
    # now we use du_da_shortened
    # which is an initial load value, followed by afull_idx_fine+1 numbers representing
    # the distributed stress concentration
    # ... Note that the distributed stress concentration indices
    # have no effect to the left of the closure point


    if sigma_ext > 0: # Tensile

        # closure_index is used to disallow any stresses or stress concentration
        # added to region that was open prior to applying load
        closure_index = np.where(scp.sigma_closure[:(scp.afull_idx)]!=0.0)[0][0] - 1
        pass
    else:
        closure_index = -1
        pass

    #du_da_shortened_iniguess=np.ones(scp.afull_idx_fine+1,dtype='d')*(1.0/(scp.afull_idx+1)) * sigma_ext/scp.dx_fine  #
    du_da_shortened_first_iniguess=np.ones(scp.afull_idx+2-(closure_index+1),dtype='d')*(1.0/(scp.afull_idx+1))* sigma_ext/scp.dx  #
    du_da_shortened_iniguess = du_da_shortened_first_iniguess
    
    perturbation_amplitude_Pa=100000.0

    # Attempt calc_contact_kernel with different initial guesses until we get convergence
    itercnt=0
    while True:
        try: 
            (du_da,contact_stress_from_displacement,displacement,contact_stress_from_stress,residual) = calc_contact_kernel(scp,sigma_ext,closure_index,du_da_shortened_iniguess)
            return (du_da,contact_stress_from_displacement,displacement,contact_stress_from_stress,residual)
        except CalcContactFailure as Failure:
            
            print("calc_contact: Observed failure %s; retry #%d with different initial conditions" % (str(Failure,itercnt)))
            sys.stdout.flush()
            du_da_shortened_iniguess = du_da_shortened_first_iniguess + (perturbation_amplitude_Pa/scp.dx)*np.random.randn(*du_da_shortened_first_iniguess.shape)
            if itercnt==10: # up to 10 tries
                raise
            pass
        itercnt+=1
        pass


def calc_contact_kernel(scp,sigma_ext,closure_index,du_da_shortened_iniguess):
    """Kernel for calc_contact... can be rerun with a pertrurbed sigma_ext if it fails"""

    
    # sigma_yield = scp.sigma_yield  # not a parameter yet... just use infinity for now
    sigma_yield=np.inf
    
    #crack_model=scp.crack_model  # not a parameter yet...
    #crack_model = ModeI_throughcrack_CODformula(scp.E)

        
    
    # Check gradient

    if sigma_ext != 0.0: # don't verify where the verification would fail with NaN
        grad_eval = soft_closure_goal_function_with_gradient(du_da_shortened_iniguess,scp,closure_index)[1]
        grad_approx = scipy.optimize.approx_fprime(du_da_shortened_iniguess,lambda x: soft_closure_goal_function_with_gradient(x,scp,closure_index)[0],sigma_ext/(scp.dx*scp.afull_idx)/3e5)
        grad_sumsquareddiff = np.sqrt(np.sum((grad_eval-grad_approx)**2.0))
        grad_sumsquared = np.sqrt(np.sum(grad_eval**2.0))
        
        #print("grad_sumsquared=%g; grad_sumsquareddiff=%g" % (grad_sumsquared,grad_sumsquareddiff))
        
        if (grad_sumsquareddiff/grad_sumsquared >= 1e-4):
            scp.save_debug_pickle(sigma_ext,duda__from_duda_shortened(scp,du_da_shortened_iniguess,closure_index),closure_index,du_da_normalization=None,goal_function_normalization=None)
            raise ValueError("Grad error too high: FAIL grad_sumsquared=%g; grad_sumsquareddiff=%g" % (grad_sumsquared,grad_sumsquareddiff))
            
        #assert(grad_sumsquareddiff/grad_sumsquared < 1e-4) # NOTE: In the obscure case where our initial guess is at a relative minimum, this might fail extraneously

    
        # check accelerated gradient
        grad_eval_accel = soft_closure_goal_function_with_gradient_accel(du_da_shortened_iniguess,scp,closure_index)[1]
        grad_sumsquareddiff_accel = np.sqrt(np.sum((grad_eval_accel-grad_approx)**2.0))
        grad_sumsquared_accel = np.sqrt(np.sum(grad_eval_accel**2.0))
    
        assert(grad_sumsquareddiff_accel/grad_sumsquared_accel < 1e-4) # NOTE: In the obscure case where our initial guess is at a relative minimum, this might fail extraneously
        pass


    goal_stress_fit_error_pascals = 150e3 # Amount of stress error to allow in fitting process. If we have more than this we keep trying to minimize
    goal_residual = (goal_stress_fit_error_pascals**2.0)*scp.afull_idx
    goal_residual_str = str(goal_residual) # for gdb
    
    du_da_normalization = max(abs(sigma_ext),10e6)/scp.dx
    goal_function_normalization = goal_residual
    load_constraint_fun_normalization = max(abs(sigma_ext),10e6)

    def load_constraint_fun_normalized(du_da_shortened_normalized):

        du_da_shortened = du_da_shortened_normalized*du_da_normalization
        #du_da_short = np.concatenate(((du_da_shortened[0],),np.zeros(closure_index+1,dtype='d'),du_da_shortened[1:]))        
        du_da_short = duda_short__from_duda_shortened(du_da_shortened,closure_index)
        
        
        
        return ((np.sum(du_da_short)*scp.dx)-sigma_ext)/load_constraint_fun_normalization
        
    load_constraint_normalized = { "type": "eq",
                                   "fun": load_constraint_fun_normalized }


    if sigma_ext > 0: # Tensile

        #nonnegative_constraint = scipy.optimize.NonlinearConstraint(lambda du_da: du_da,0.0,np.inf)
        nonnegative_constraint = { "type": "ineq",
                                   "fun": lambda du_da_shortened: du_da_shortened }



        # Calculate x

        
        # Allow total iterations to be broken into pieces separated by failures with minimize error 9 (Iteration limit exceeded)
        # (for some reason, restarting the minimizer where it left off seems to help get it to the goal)
        total_maxiter=1000000
        niter = 0
        #epsval1 = np.abs(sigma_ext)/scp.a/5000.0
        #epsval2 = np.max(np.abs(scp.sigma_closure))/scp.a/5000.0
        #epsval = max(epsval1,epsval2)
        epsval=1e-5
        epsvalscaled = epsval
        terminate=False
        starting_value=du_da_shortened_iniguess
        status=None
        fun_str=None

        while niter < total_maxiter and not terminate: 
            this_niter=10000
            #print("calling scipy.optimize.minimize; sigma_ext=%g; eps=%g maxiter=%d ftol=%g" % (sigma_ext,epsvalscaled,this_niter,scp.afull_idx_fine*(np.abs(sigma_ext)+20e6)**2.0/1e14))
            res = scipy.optimize.minimize(soft_closure_goal_function_with_gradient_normalized_accel,starting_value/du_da_normalization,args=(scp,closure_index,du_da_normalization,goal_function_normalization),   # was soft_closure_goal_function_accel
                                          constraints = [ load_constraint_normalized ], #[ nonnegative_constraint, load_constraint ],
                                          method="SLSQP",
                                          jac=True,
                                          options={"eps": epsvalscaled,
                                                   "maxiter": this_niter,
                                                   "ftol": 1e-12})#scp.afull_idx*(np.abs(sigma_ext)+20e6)**2.0/1e14})
            #print("res=%s" % (str(res)))
            #print("niter = %d; residual = %g; res.message=%s" % (niter+res.nit,res.fun,res.message)

            res_fun_denormalized = res.fun*goal_function_normalization

            # Variables accessible from gdb)
            status = res.status
            fun_str = str(res_fun_denormalized)

            if res_fun_denormalized <= goal_residual:
                terminate=True
                pass
            
            if res.status != 9 and res.status != 7:  # anything but reached iteration limit or eps increase
                if res_fun_denormalized <= goal_residual or res.nit==0:
                    terminate=True
                    pass 
                else:
                    # ... otherwise keep trying!
                    #if res.nit==1:
                    #    import pdb
                    #    pdb.set_trace()
                    #    pass
                    epsvalscaled = epsval # reset eps to nominal value
                    starting_value = res.x*du_da_normalization # Next iteration starts where this one left off
                    pass
                pass
            elif res.status==7:
                # Rank-deficient equality constraint subproblem HFTI 
                # Generally indicates too fine epsilon...
                if epsvalscaled/epsval < 10:
                    epsvalscaled *= 2 
                    starting_value = res.x*du_da_normalization # Next iteration starts where this one left off
                    pass
                else:
                    print("WARNING: initialize_contact(): repeated rank-deficient equality constraint subproblem HFTI... Terminating!\n")
                    sys.stdout.flush()
                    terminate=True  # Don't allow eps to grow too much
                    pass
                pass
            else:
                epsvalscaled = epsval # reset eps to nominal value
                starting_value = res.x*du_da_normalization # Next iteration starts where this one left off
                pass
            niter += this_niter #res.nit
            pass

        if niter >= total_maxiter and res_fun_denormalized > goal_residual:
            print("soft_closure/calc_contact (tensile): WARNING Maximum number of iterations (%d) reached and residual (%g) exceeds goal (%g); res.status=%d" % (total_maxiter,res_fun_denormalized,goal_residual,res.status))
            sys.stdout.flush()
            scp.save_debug_pickle(sigma_ext,duda__from_duda_shortened(scp,res.x*du_da_normalization,closure_index),closure_index,du_da_normalization,goal_function_normalization,load_constraint_fun_normalization=load_constraint_fun_normalization)

            pass

        if (res.status==6 and res_fun_denormalized > goal_residual) or abs(res.fun) > 1e25: 
            # Characteristic of a divergence failure 
            raise CalcContactFailure("Divergence identified: Maximum number of iterations (%d) reached and residual (%g) exceeds goal (%g)" % (total_maxiter,res_fun_denormalized,goal_residual))
            
        if (res.status==9 or res.status==0) and res_fun_denormalized > goal_residual:
            raise CalcContactFailure("Maximum number of iterations (%d) reached and residual (%g) exceeds goal (%g)" % (total_maxiter,res_fun_denormalized,goal_residual))


        #res = scipy.optimize.minimize(goal_function,du_da_shortened_iniguess,method='nelder-mead',options={"maxfev": 15000})
        if res_fun_denormalized > goal_residual and not res.success: # and res.status != 4:
            # (ignore incompatible constraint, because our constraints are
            # compatible by definition, and scipy 1.2 seems to diagnose
            # this incorrectly... should file bug report)
            print("minimize error %d: %s" % (res.status,res.message))
            sys.stdout.flush()
            scp.save_debug_pickle(sigma_ext,duda__from_duda_shortened(scp,res.x*du_da_normalization,closure_index),closure_index,du_da_normalization,goal_function_normalization,load_constraint_fun_normalization=load_constraint_fun_normalization)
            import pdb
            pdb.set_trace()
            pass

        # Verify proper operation of accelerated code
        (slowcalc,slowcalc_gradient) = soft_closure_goal_function_with_gradient(res.x*du_da_normalization,scp,closure_index)
        (fastcalc,fastcalc_gradient) = soft_closure_goal_function_with_gradient_accel(res.x*du_da_normalization,scp,closure_index)
        if abs((slowcalc-fastcalc)/slowcalc) >= 1e-4 and (slowcalc > goal_residual/100.0 or fastcalc > goal_residual/100.0):
            scp.save_debug_pickle(sigma_ext,duda__from_duda_shortened(scp,res.x*du_da_normalization,closure_index),closure_index,du_da_normalization,goal_function_normalization,load_constraint_fun_normalization=load_constraint_fun_normalization)

            #from VibroSim_Simulator.function_as_script import scriptify
            #(slowcalc2,slowcalc2_grad) = scriptify(soft_closure_goal_function_with_gradient)(res.x,scp,closure_index)
            raise ValueError("Accelerated goal calculation mismatch: %g vs %g" % (slowcalc,fastcalc))

        
        
        du_da_shortened=res.x*du_da_normalization
        #du_da = np.concatenate(((du_da_shortened[0],),np.zeros(closure_index+1,dtype='d'),du_da_shortened[1:],np.zeros(scp.xsteps - scp.afull_idx - 2 ,dtype='d')))        
        du_da = duda__from_duda_shortened(scp,du_da_shortened,closure_index)

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
        
        
        
        constraints = [ load_constraint_normalized ]
        
        #if sigma_ext < 0.0:
        #    # if we are applying compressive external load,
        #    # net shift of concentration should always be compressive
        #    constraints.append(nonpositive_constraint)
        #    pass

        total_maxiter=1000000
        niter = 0
        #epsval1 = np.abs(sigma_ext)/scp.a/5000.0
        #epsval2 = np.max(np.abs(scp.sigma_closure))/scp.a/5000.0
        #epsval = max(epsval1,epsval2)
        epsval=1e-5
        epsvalscaled = epsval
        terminate=False
        starting_value=du_da_shortened_iniguess
        while niter < total_maxiter and not terminate: 
            this_niter=10000
            res = scipy.optimize.minimize(soft_closure_goal_function_with_gradient_normalized_accel,starting_value/du_da_normalization,args=(scp,closure_index,du_da_normalization,goal_function_normalization),   # was soft_closure_goal_function_accel
                                          constraints = constraints,
                                          method="SLSQP",
                                          jac=True,
                                          options={"eps": epsvalscaled,
                                                   "maxiter": this_niter,
                                                   "ftol": 1e-12})  # scp.afull_idx*(np.abs(sigma_ext)+20e6)**2.0/1e14})

            res_fun_denormalized = res.fun*goal_function_normalization

            # Variables accessible from gdb
            status = res.status
            fun_str = str(res_fun_denormalized)

            if res_fun_denormalized <= goal_residual:
                terminate=True
                pass
 
            if res.status != 9 and res.status != 7: # anything but reached iteration limit or eps increase needed
                if res_fun_denormalized <= goal_residual or res.nit==0:
                    terminate=True
                    pass
                else:
                    # ... otherwise keep trying!
                    epsvalscaled = epsval # reset eps to nominal value
                    starting_value = res.x*du_da_normalization # Next iteration starts where this one left off
                    pass
                pass
            elif res.status==7:
                # Rank-deficient equality constraint subproblem HFTI 
                # Generally indicates too fine epsilon...
                if epsvalscaled/epsval < 10:
                    epsvalscaled *= 2 
                    starting_value = res.x*du_da_normalization # Next iteration starts where this one left off
                    pass
                else:
                    print("WARNING: initialize_contact(): repeated rank-deficient equality constraint subproblem HFTI... Terminating!\n")
                    sys.stdout.flush()

                    terminate=True  # Don't allow eps to grow too much
                    pass
                pass
            else:
                epsvalscaled = epsval # reset eps to nominal value
                starting_value = res.x*du_da_normalization # Next iteration starts where this one left off
                pass
            niter += this_niter #res.nit
            pass
            
        if niter >= total_maxiter and res_fun_denormalized > goal_residual:
            print("soft_closure/calc_contact (compressive): WARNING Maximum number of iterations (%d) reached and residual (%g) exceeds goal (%g)" % (total_maxiter,res_fun_denormalized,goal_residual))
            sys.stdout.flush()
            pass

        if res.status==6 and res_fun_denormalized > goal_residual: 
            # Characteristic of a divergence failure 
            raise CalcContactFailure("Divergence Identified: Maximum number of iterations (%d) reached and residual (%g) exceeds goal (%g)" % (total_maxiter,res_fun_denormalized,goal_residual))

        if (res.status==9 or res.status==0) and res_fun_denormalized > goal_residual:
            raise CalcContactFailure("Maximum number of iterations (%d) reached and residual (%g) exceeds goal (%g)" % (total_maxiter,res_fun_denormalized,goal_residual))

        #res = scipy.optimize.minimize(goal_function,du_da_shortened_iniguess,method='nelder-mead',options={"maxfev": 15000})
        if res_fun_denormalized > goal_residual and not res.success: #  and res.status != 4:
            # (ignore incompatible constraint, because our constraints are
            # compatible by definition, and scipy 1.2 seems to diagnose
            # this incorrectly... should file bug report)
            print("minimize error %d: %s" % (res.status,res.message))
            sys.stdout.flush()
            scp.save_debug_pickle(sigma_ext,duda__from_duda_shortened(scp,res.x*du_da_normalization,closure_index),closure_index,du_da_normalization,goal_function_normalization,load_constraint_fun_normalization=load_constraint_fun_normalization)
            import pdb
            pdb.set_trace()
            pass
        
        du_da_shortened=res.x*du_da_normalization

        # Verify proper operation of accelerated code
        (slowcalc,slowcalc_gradient) = soft_closure_goal_function_with_gradient(res.x*du_da_normalization,scp,closure_index)
        (fastcalc,fastcalc_gradient) = soft_closure_goal_function_with_gradient_accel(res.x*du_da_normalization,scp,closure_index)

        if abs((slowcalc-fastcalc)/slowcalc) >= 1e-4 and (slowcalc > goal_residual/100.0 or fastcalc > goal_residual/100.0):
            #from VibroSim_Simulator.function_as_script import scriptify
            #(slowcalc2,slowcalc2_grad) = scriptify(soft_closure_goal_function_with_gradient)(res.x*du_da_normalization,scp,closure_index)
            raise ValueError("Accelerated goal calculation mismatch: %g vs %g" % (slowcalc,fastcalc))


        #du_da = np.concatenate((du_da_shortened,np.zeros(scp.xsteps*scp.fine_refinement - scp.afull_idx_fine - 2 ,dtype='d')))
        assert(closure_index==-1)
        #du_da = np.concatenate(((du_da_shortened[0],),np.zeros(closure_index+1,dtype='d'),du_da_shortened[1:],np.zeros(scp.xsteps - scp.afull_idx - 2 ,dtype='d')))        
        du_da = duda__from_duda_shortened(scp,du_da_shortened,closure_index)

        #(contact_stress,displacement) = sigmacontact_from_displacement(scp,du_da)
        

        

        #sys.modules["__main__"].__dict__.update(globals())
        #sys.modules["__main__"].__dict__.update(locals())
        #raise ValueError("Foo!")
        pass


    
    contact_stress_from_stress = sigmacontact_from_stress(scp,du_da)
    (contact_stress_from_displacement,displacement) = sigmacontact_from_displacement(scp,du_da)

    residual = res_fun_denormalized
    
    
    return (du_da,contact_stress_from_displacement,displacement,contact_stress_from_stress,residual)
    









def soft_closure_plots(scp,du_da,titleprefix=""):
    from matplotlib import pyplot as pl
    #pl.rc('text', usetex=True) # Support greek letters in plot legend
    
    #last_closureidx = np.where(x_bnd >= a)[0][0]

    #  constraint that integral of du/da must equal sigma_ext
    # means that last value of u should match sigma_ext
    
    # sigmacontact is positive compression

    #du_da = calc_du_da(u,scp.dx_fine)


    x_du_da = np.concatenate(((scp.x[0]-scp.dx,),scp.x))
    
    
    (from_displacement,displacement) = sigmacontact_from_displacement(scp,du_da)
    from_stress = sigmacontact_from_stress(scp,du_da)


    
    
    #pl.figure()
    #pl.clf()
    #pl.plot(scp.x_fine,u)
    #pl.title("distributed stress concentration")
    #pl.grid()
    
    sigmacontact_plot=pl.figure()
    pl.clf()
    pl.plot(scp.x*1e3,from_displacement/1e6,'-',
            scp.x*1e3,from_stress/1e6,'-'),
    pl.grid()
    pl.legend(("from displacement","from stress"))
    pl.ylabel("Stress (MPa)")
    pl.xlabel('Position (mm)')
    pl.title(titleprefix+"sigmacontact")

    u = np.cumsum(du_da)*scp.dx
    # u nominally on position basis x_fine+dx_fine/2.0

    duda_plot = pl.figure()
    pl.clf()
    pl.plot(x_du_da*1e3,du_da/1e12,'-')
    pl.grid()
    pl.title(titleprefix+"distributed stress concentration derivative\ntotal load=%f MPa" % (u[scp.afull_idx]/1e6))
    pl.xlabel('Position (mm)')
    pl.ylabel("Distributed stress concentration (TPa/m)")

    displacement_plot=pl.figure()
    pl.clf()
    pl.plot(scp.x*1e3,(scp.crack_initial_opening-(scp.sigma_closure/scp.Lm)**(2.0/3.0))*1e6,'-')
    pl.plot(scp.x*1e3,displacement*1e6,'-')
    pl.grid()
    pl.legend(('Initial displacement','Final displacement'))
    pl.xlabel('Position (mm)')
    pl.ylabel('Displacement (um)')
    pl.title(titleprefix+"displacement")

    return (sigmacontact_plot,duda_plot,displacement_plot)
    


