
import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize



class ModeII_crack_model(object):
    # abstract class
    #
    # Implementations should define:
    #  * methods: 
    #    * eval_tauII_theta0_times_rootr_over_sqrt_a_over_tauext(self,a)
    #    * evaluate_ModeII_CSD_vectorized(self,tau_applied,x,xt)  # should be vectorized over x (not necessarily xt)
    pass


class ModeII_Beta_CSD_Formula(ModeII_crack_model):
    """This represents a crack model where we are given a formula
    for K_II of the form K_II = tau*sqrt(pi*a*beta), and
    CSD is a function ut(object,surface_position,surface_length). 
    (ut for tangent displacement)


    You can add member variables (which will be accessible from 
    the u function) by providing them as keyword arguments to 
    the constructor. 

    At minimum you must provide a function:
       ut(object,surface_position,surface_length)  
       which should be vectorized over surface position, and a function
       beta(object), which return the COD and beta values respectively. 
       (beta is a function, so you can set it up so that the crack model
       will work correctly if its attribute parameters are updated)

"""

    ut=None
    beta=None
    
    def __init__(self,**kwargs):
        if "ut" not in kwargs:
            raise ValueError("Must provide COD function u(object,sigma_applied,surface_position,surface_length)")

        if "beta" not in kwargs:
            raise ValueError("Must provide K coefficient beta(object)")
        

        for kwarg in kwargs:
            setattr(self,kwarg,kwargs[kwarg])
            pass

        pass

    def eval_tauII_theta0_times_rootr_over_sqrt_a_over_tauext(self,a):
        # For a shear crack with the tip at the origin, intact material
        # to the right (x > 0), broken material to the left (x < 0)
        # The shear stress @ theta=0 multiplied by sqrt(x)/(sqrt(a)*tauext)
        # where x ( > 0) is the position where the stress is measured,
        # a is the (half) length of the crack, and tauext
        # is the external shear load

        # Per Suresh (9.47a) and Anderson (table 2.1)
        # Based on K_II=(tau_ext*sqrt(pi*a*beta))
        # instead of  K_II=(tau_ext*sqrt(pi*a))

        tauII_theta0_times_rootr_over_sqrt_a_over_tauext = sqrt(self.beta(self))/sqrt(2.0)  
                
        return tauII_theta0_times_rootr_over_sqrt_a_over_tauext

    def eval_ModeII_CSD_vectorized(self,sigma_applied,x,xt):
        return self.ut(self,sigma_applied,x,xt)
    
    
    pass




def indef_integral_of_simple_squareroot_quotients(a,u):
    (a,u) = np.broadcast_arrays(a,u) # make sure a and u are the same shape
    # From Wolfram Alpha: integral of (sqrt(u)/sqrt(a-u)) du
    #  = a*arctan(sqrt(u)/sqrt(a-u)) - sqrt(u)*sqrt(a-u)
    #if (a==u).any():
    #    raise ValueError("Divide by zero")
    # return a*arctan(sqrt(u)/sqrt(a-u)) - sqrt(u)*sqrt(a-u)

    # Calculate division-by-zero and
    # non division-by-zero regimes separately
    
    # Limiting case as a-u -> 0:
    # Let v = a-u -> u = a-v
    # integral = a*arctan(sqrt(a-v)/sqrt(v)) - sqrt(a-v)*sqrt(v)
    # Per wolfram alpha: limit as v approaches 0 + of a*atan(sqrt(a-v)/sqrt(v)) 
    # = pi*a/2.0
    divzero = (a==u) | ((np.abs(a-u) < 1e-10*a) & (np.abs(a-u) < 1e-10*u))

    integral = np.zeros(a.shape[0],dtype='d')
    integral[~divzero] = a[~divzero]*arctan(sqrt(u[~divzero])/sqrt(a[~divzero]-u[~divzero])) - sqrt(u[~divzero])*sqrt(a[~divzero]-u[~divzero])

    integral[divzero] = np.pi*a[divzero]/2.0
    return integral





def integral_shearstress_growing_effective_crack_length_byxt(x,tauext1,tauext_max,F,xt1,xt2,crack_model):
    """ Evaluate the incremental shear stress field on a shear crack
    that is growing in effective length from xt1 to xt2 due to an external 
    load (previous value tauextp, limiting value tauext_max)
    
    It is assumed that the effective tip moves linearly with 
    applied external shear stress, with rate given by F (Pascals 
    of external shear stress / meters of tip motion ... i.e. 1/(our
    previously defined D)

    The external shear stress is presumed to be bounded by tauext_max
    (which may be np.Inf to leave it unbounded). In such a case
    it will evaluate the incremental shear stress only up to 
    the length that gives tauext_max.

    Returns (use_xt2,tauext2,res) where use_xt2 is the actual 
    upper bound of the integration (as limited by tauext_max), 
    tauext2 is the external shear stress load corresponding 
    to the crack being opened to use_xt2, and which will be 
    <= to tauext_max. 

    Rationale: 

     The mode II normal stress formula is:
      sigma_xy_crack = (K_II / sqrt(2*pi*r))  (Suresh, Eq. 9.47a at theta=0)

    ... we choose to add in the external field not being held by the crack
      sigma_xy_total = (K_II / sqrt(2*pi*r)) + sigma_ext

    The variable tauII_theta0_times_rootr_over_sqrt_a_over_sigmaext
    represents the value of sigmaxy_crack(x,K) with the above formula 
    for K_II (the simple K_II=tau_ext*sqrt(pi*a)) )
    substituted for K, evaluated for horizontal axis beyond the 
    tip (i.e. theta=0) and then multiplied by sqrt(r) (sqrt(position
    beyond the tip) and divided by sqrt(cracklength) and also by 
    tauext. 

    Let tauII(r,K) be the mode II shear stress formula divided by 
    the external load, where the K dependence is presumed to be 
    strictly multiplicative. Let K_over_tauext = K/tauext

    Let xt be the position of the effective tip at external load
    tauext

    The incremental shear stress field would be
    integral_tauext1^tauext2 of 1.0 + tauII(x-xt,K_over_tauext) dtauext
    (note that K and xt are dependent on tauext)
   

    The variable tauII_theta0_times_rootr_over_sqrt_a_over_tauext
    represents the value of tauII(x,K) with tauext*sqrt(pi*a) 
    substituted for K, evaluated for horizontal axis beyond the 
    tip (i.e. theta=0) and then multiplied by sqrt(r) (sqrt(position
    beyond the tip) and divided by sqrt(cracklength) and also by 
    tauext. 

    Then we can rewrite the incremental shear stress as: 
    integral_tauext1^tauext2 of 1.0 + tauII_theta0_times_rootr_over_sqrt_a_over_tauext*sqrt(xt)/sqrt(x-xt) dtauext
    Here, xt is still dependent on tauext... this will give shear stress
    as a function of position (x). 

    We assume xt is linearly dependent on shear stress:
    xt = xtp + (1/F)*(tauext-tauext1)
    where xtp is the final xt from the previous step. 

    tauII_theta0_times_rootr_over_sqrt_a_over_tauext is a constant 

    So our incremental shear is
    integral_tauext1^tauext2 (1.0 + tauII_theta0_times_rootr_over_sqrt_a_over_tauext sqrt(xt)/sqrt(x-xt) dtauext
    where we ignore any contributions corresponding to (x-xt) <= 0

    (the new 1.0 term represents that beyond the effective tip the external 
    load directly increments the stress state, in addition to the stress 
    concentration caused by the presence of the open region)    

    pull out constant term

    (tauext2-tauext1) + integral_tauext1^tauext2 tauII_theta0_times_rootr_over_sqrt_a_over_tauext sqrt(xt)/sqrt(x-xt) dtauext

    Perform change of integration variable tauext -> xt: 
       Derivative of xt:
       dxt = (1/F)*dtauext
       dtauext = F*dxt


    So the  incremental shear stress we are solving for is
    (tauext2-tauext1) + integral_xt1^xt2 tauII_theta0_times_rootr_over_sqrt_a_over_tauext sqrt(xt)*F/sqrt(x-xt)  dxt
    where we ignore any contributions corresponding to (x-xt) <= 0

    and tauext2 = tauext1 + (xt2-xt1)*F 
 
    F is a constant so have 
    F*(xt2-xt1) +  F * tauII_theta0_times_rootr_over_sqrt_a_over_tauext * integral_xt1^xt2 sqrt(xt)/(sqrt(x-xt))  dxt


    This is then the integral of (sqrt(u)/sqrt(a-bu)) du
       where u = xt, 
    with solution given by
       indef_integral_of_simple_squareroot_quotients(x,xt2) - indef_integral_of_simple_squareroot_quotients(x,xt1)

    Well almost. We only consider the region of this integral where 
    x-xt > 0. This can be accomplished by shifting the bounds when 
    needed. 

    x > xt
     =>
    xt2 < x  and xt1 < x  ... xt1 < xt2

    So: Integral = 0 where x < xt1
    Integral upper bound =  x where xt1 < x < xt2
    Integral upper bound = xt2 where x > xt2

       indef_integral_of_simple_squareroot_quotients(x,upper_bound) - indef_integral_of_simple_squareroot_quotients(x,xt1)
    
    So our actual solution putting everything together is:
    0 where x < xt1 
    otherwise: 
    upper_bound = min(x, xt2) 
    F*(upper_bound-xt1) + (tauII_theta0_times_rootr_over_sqrt_a_over_tauext*F)*(indef_integral_of_simple_squareroot_quotients(x,upper_bound) - indef_integral_of_simple_squareroot_quotients(x,xt1))

    """
    
    # For a shear crack with the tip at the origin, intact material
    # to the right (x > 0), broken material to the left (x < 0)
    # The shear stress @ theta=0 multiplied by sqrt(x)/(sqrt(a)*tauext)
    # where x ( > 0) is the position where the stress is measured,
    # a is the (half) length of the crack, and tauext
    # is the external shear load


    tauext2 = tauext1 + (xt2-xt1)*F

    use_xt2 = xt2
    if tauext2 > tauext_max:
        # bound tauext by tauext_max... by limiting xt2
        if F > 0:
            use_xt2 = xt1 + (tauext_max-tauext1)/F
            pass
        if F==0 or use_xt2 > xt2:
            use_xt2 = xt2
            pass
        
        tauext2 = tauext_max
        pass
    


    upper_bound = use_xt2*np.ones(x.shape,dtype='d')
    
    # alternate upper_bound:
    use_alternate = x < upper_bound
    upper_bound[use_alternate] = x[use_alternate]
    
    res=np.zeros(x.shape,dtype='d')

    nonzero = x > xt1

    xtavg=(xt1+use_xt2)/2.0
    
    tauII_theta0_times_rootr_over_sqrt_a_over_tauext = crack_model.eval_tauII_theta0_times_rootr_over_sqrt_a_over_tauext(xtavg) 
    
    res[nonzero] = F*(upper_bound[nonzero]-xt1) + (tauII_theta0_times_rootr_over_sqrt_a_over_tauext*F) * (indef_integral_of_simple_squareroot_quotients(x[nonzero],upper_bound[nonzero]) - indef_integral_of_simple_squareroot_quotients(x[nonzero],xt1))

    
    
    return (use_xt2,tauext2,res)


def solve_incremental_shearstress(x,x_bnd,tau,sigma_closure,shear_displ,xt_idx,dx,tauext,tauext_max,a,mu,crack_model,calculate_displacements=True):
    """The overall frictional slip constraint is that
    F <= mu*N
    For a through-crack of thickness h, short segment of width dx
    tau*h*dx <= mu*h*dx*sigma_closure (where sigma_closure is positive (i.e. compressive)

    or equivalently 
    tau <= mu*sigma_closure (where sigma_closure is positive (i.e. compressive))

    Consider an increment in position dx. 
    Assume from previous steps we have a superimposed 
    tau(x) in the plane of the crack. In this step we are 
    adding an increment to tau_external. 

    The rule is that 
    (the preexisting tau(x) + the increment in tau(x)) <= mu*sigma_closure

    Here, given the preexisting tau(x), sigma_closure(x),  and an increment 
    of opening the  crack by one unit of dx, we are evaluating the 
    increment in tau(x) as well as the increment in tau_external
    
    We can evaluate the increment in tau from: 

    (use_xt2,tauext2,tau_increment)=integral_shearstress_growing_effective_crack_length_byxt(x,tauext,tauext_max,F,x[xt_idx],x[xt_idx+1])

    But to do this we need to solve for F. 

    We do this by setting the shear stress equal to the frictional load
    over the new step (unit of dx).
    """

    next_bound = x_bnd[xt_idx+1]
    if next_bound > a:
        next_bound=a
        pass
    
    def obj_fcn(F):
        (use_xt2,tauext2,tau_increment)=integral_shearstress_growing_effective_crack_length_byxt(x,tauext,np.inf,F,x_bnd[xt_idx],next_bound,crack_model)
        return (tau+tau_increment - mu*sigma_closure)[xt_idx]

    # F measures the closure gradient in (Pascals external shear stress / meters of tip motion)

    if sigma_closure[xt_idx] >= 0.0 and tau[xt_idx] < mu*sigma_closure[xt_idx]:
        # There is a closure stress here but not the full tau it can support

        # Bound it by 0  and the F that will give the maximum
        # contribution of tau_increment: 2.0*(tauext_max-tauext1)/(xt2-xt1)
        Fbnd = 2.0*(tauext_max - tauext)/(next_bound-x_bnd[xt_idx])

        # Increase Fbnd until we get a positive result from obj_fcn
        while Fbnd != 0.0 and obj_fcn(Fbnd) < 0.0:
            Fbnd*=2.0;
            pass
        
        # Condition below should only occur when Fbnd==0.0, i.e. when tauext_max==tauext, or if the objective function is already satisfied
        if Fbnd == 0.0 or obj_fcn(Fbnd) <= 0.0:
            # Maximum value of objective is < 0... This means that
            # with the steepest tau vs. xt slope possible (given
            # the total shear load we are applying) we still
            # can't get tau+tau_increment to match mu*sigma_closure.
            # ... We will have to make do with tau+tau_increment
            #  < sigma_closure
            # So our best result is just Fbnd
            F=Fbnd
            pass
        else:
            # brentq requires function to be different signs
            # at 0.0 (negative) and Fbnd (positive) 
            F = scipy.optimize.brentq(obj_fcn,0.0,Fbnd,disp=True)
            pass
        
        (use_xt2,tauext2,tau_increment)=integral_shearstress_growing_effective_crack_length_byxt(x,tauext,tauext_max,F,x_bnd[xt_idx],next_bound,crack_model)

        # For displacement calculate at x centers... use average of left and right boundaries, except for (perhaps) last point where instead of the right boundary we use the actual tip.
        if calculate_displacements:
            incremental_displacement = np.zeros(x.shape[0],dtype='d')
            xt = (x_bnd[xt_idx]+use_xt2)/2.0
            left_of_effective_tip = (x < xt)
            incremental_displacement[left_of_effective_tip] = shear_displacement(tauext2-tauext,x[left_of_effective_tip],xt,crack_model)
            pass
        pass
    else:
        # No closure stress at this point, or tau is already at the limit
        # of what can be supported here
        
        # ... just open up to the next spot
        use_xt2 = x_bnd[xt_idx+1]

        if use_xt2 > a:
            # Cannot open beyond tips
            use_xt2 = a
            pass

        tauext2 = tauext
        tau_increment = np.zeros(x.shape[0],dtype='d')
        incremental_displacement = np.zeros(x.shape[0],dtype='d')
        pass
    
    if calculate_displacements:
        ret_displ = shear_displ+incremental_displacement
        pass
    else:
        ret_displ=None
        pass


        
    return (use_xt2,tauext2, tau+tau_increment, ret_displ)
    
    


#####SHEAR DISPLACEMENT FUNCTION 
#A function for shear displacement was obtained from Suresh (9.47b)
#This function calculated the displacement in the x direction, ux, 
#in Mode II. This is for a semi-infinite crack in an infinite plate
#of an isotropic and homogeneous solid 
#NOTE: This function will always produce ux=0 for theta equals 0. 
#This is because sin(0)=0 and the function is multiplied completely
#by this. uy would not equal zero, however we are not interested
#in that value because the displacement in the y direction 
#(perpendicular to the crack) is NOT the sliding motion of the
#surfaces creating heating from a shear loading that we are 
#trying to find.
#NOTE: Leaving the np.sqrts because if x-xt<0, then the 
#displacment would have no real meaning really, so nan is 
#sufficient because something is wrong at that point.
# sdh 11/4/18 -- we're really interested here in theta=pi
#                here, I think... (verified)
def shear_displacement(tau_applied,x,xt,crack_model):
    ##plane stress is considered
    #Kappa = (3.0-nu)/(1.0+nu)
    #KII = tau_applied*np.sqrt(np.pi*(xt))
    #theta = np.pi
    #return (KII/(2.0*E))*(np.sqrt((xt-x)/(2.0*np.pi)))*((1.0+nu)* \
    # (((2.0*Kappa+3.0)*(np.sin(theta/2.0)))+(np.sin(3.0*theta/2.0))))
    ut = crack_model.eval_ModeII_CSD_vectorized(tau_applied,x,xt)
    return ut


def solve_shearstress(x,x_bnd,sigma_closure,dx,tauext_max,a,mu,tau_yield,crack_model,verbose=False,calculate_displacements=True):
    #Initialize the external applied dynamic shear stress starting at zero
    
    tauext = 0.0 # External shear load in this step (Pa)
    


    #####MAIN SUPERPOSITION LOOP

    #Initialize shear stress field (function of x)
    tau = np.zeros(x.shape,dtype='d')
    
    #Initialize the Displacement state as zero
    if calculate_displacements:
        shear_displ = np.zeros(x.shape,dtype='d')
        pass
    else:
        shear_displ = None
        pass
    
    #Initialize x step counter
    xt_idx = 0

    use_xt2=0

    # Before initial slip, tau just increases uniformly
    # (Note: stress distribution may not be very accurate if
    # initial slip does not occur @ x=0)
    argmin_sigma_closure = np.argmin(sigma_closure[x < a])
    min_sigma_closure=sigma_closure[x < a][argmin_sigma_closure]
    if min_sigma_closure > 0:
        # We can hold a stress of mu*min_sigma_closure
        # without any slip at all.

        uniform_shear = np.min((mu*min_sigma_closure,tauext_max))
        
        tau += uniform_shear
        tauext += uniform_shear

        # assume anything to the left of the
        # sigma_closure minimum is free to move
        # once we get to this point
        xt_idx=argmin_sigma_closure
        use_xt2=x_bnd[xt_idx]
        pass
    elif min_sigma_closure <= 0:
        # There is an opening point...

        # Find where sigma_closure goes from negative (tensile)
        # to positive (compressive)

        signchange_idxs = np.where((sigma_closure[x < a][:-1] < 0.0) & (sigma_closure[x < a][1:] >= 0.0))[0]

        if signchange_idxs.shape[0] > 0:
            xt_idx=signchange_idxs[0]


            if x_bnd[xt_idx+1] < a:
                closure_slope=(sigma_closure[xt_idx+2]-sigma_closure[xt_idx+1])/dx
                pass
            else:
                closure_slope=(sigma_closure[xt_idx+1]-sigma_closure[xt_idx])/dx
                pass
            
            assert(closure_slope > 0.0)
            
            # Project tip position backwards from x[signchange_idxs+1]
            use_xt2=x[xt_idx+1]-sigma_closure[xt_idx+1]/closure_slope
            pass
        else:
            # No signchange
            
            if sigma_closure[x<a][-1] > 0.0:
                # have compressive (positive) closure stresses, but no signchange
                # ... crack must be fully closed
                xt_idx=0
                use_xt2=0.0
                
                pass
            else:
                # crack must be fully open            
                xt_idx = np.where(x < a)[0][-1] # open all the way to tip
                # if closure stress is tensile everywhere
                use_xt2=a
                pass
            pass

        
        pass
    else:
        assert(0) # shouldn't be possible
        pass
    
    

    
    done=False
    
    #if tauext==tauext_max:
    #    # Used up all of our applied load...  Done!
    #    done=True
    #    pass

    while not done and tauext < tauext_max: 
        
        (use_xt2,tauext, tau, shear_displ) = solve_incremental_shearstress(x,x_bnd,tau,sigma_closure,shear_displ,xt_idx,dx,tauext,tauext_max,a,mu,crack_model,calculate_displacements=calculate_displacements)
        
    
        if use_xt2 < x_bnd[xt_idx+1] or tauext==tauext_max or use_xt2 >= a:
            # Used up all of our applied load or all of our crack... Done!
            done=True
            pass

        if verbose: 
            #Print what is happening in the loop
            print("Step: %d @ x=%f mm: %f MPa of shear held" % (xt_idx,x[xt_idx]*1e3,tauext/1e6))
            if calculate_displacements:
                print("Shear displacement @ x=%f mm: %f nm" % (x[0]*1e3, shear_displ[0]*1e9))
                pass
            pass
        
        xt_idx+=1
        
        
        pass

    if tauext < tauext_max:
        # We opened the crack to the tips without providing
        # the full external load.
        # Now the effective tip is the physical tip (at a)
        #
        # ... Apply the remaining load increment
        assert(use_xt2 == a)
        
        tau_increment = np.zeros(x.shape[0],dtype='d')
        ti_nodivzero_nonegsqrt = x-a > 1e-10*a
        ti_divzero = (x-a >= 0) & ~ti_nodivzero_nonegsqrt
        
        tauII_theta0_times_rootr_over_sqrt_a_over_tauext = crack_model.eval_tauII_theta0_times_rootr_over_sqrt_a_over_tauext(a) 

        #tau_increment = tauII_theta0_times_rootr_over_sqrt_a_over_tauext*(tauext_max-tauext)*sqrt(a)/sqrt(x-a)
        # New (sigmaext_max - sigmaext) term is the incremental external  stress field beyond the tips added in addition to the stress contcentration effect
        tau_increment[ti_nodivzero_nonegsqrt] = (tauext_max-tauext) + tauII_theta0_times_rootr_over_sqrt_a_over_tauext*(tauext_max-tauext)*sqrt(a)/sqrt(x[ti_nodivzero_nonegsqrt]-a)
        tau_increment[ti_divzero]=np.inf

        # Limit shear stresses at physical tip (and elsewhere) to yield
        tau_increment[tau + tau_increment > tau_yield] = tau_yield-tau[tau+tau_increment > tau_yield]
        
        # accumulate stresses onto tau
        tau += tau_increment

        # record increment in displacement
        left_of_effective_tip = x < a
        if calculate_displacements:
            shear_displ[left_of_effective_tip] += shear_displacement(tauext_max-tauext,x[left_of_effective_tip],a,crack_model)
            pass
        # Record increment in tauext
        tauext = tauext_max

        if verbose:
            print("Step: Open to tips @ x=%f mm: %f MPa of shear held" % (a*1e3,tauext/1e6))
            print("Shear displacement @ x=%f mm: %f nm" % (x[0]*1e3, shear_displ[0]*1e9))
            pass
        pass
    
    return (use_xt2, tau, shear_displ)



def ModeII_throughcrack_CSDformula(E,nu):
    def ut(E,nu,tau_applied,x,xt):
        # Non weightfunction method:

        # ***!!! This could probably be improved by using
        # the formula for crack surface displacement along
        # the entire crack, not the near-tip formula
        # used below
        
        #plane stress is considered
        Kappa = (3.0-nu)/(1.0+nu)
        KII = tau_applied*np.sqrt(np.pi*(xt))
        theta = np.pi
        ut = (KII/(2.0*E))*(np.sqrt((xt-x)/(2.0*np.pi)))*((1.0+nu)* (((2.0*Kappa+3.0)*(np.sin(theta/2.0)))+(np.sin(3.0*theta/2.0))))
        
        return ut
    
    
    return ModeII_Beta_CSD_Formula(E=E,
                                   nu=nu,
                                   beta=lambda obj: 1.0,
                                   ut = lambda obj,tau_applied,x,xt: ut(obj.E,obj.nu,tau_applied,x,xt))

    
    

