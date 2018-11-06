# -*- coding: utf-8 -*-
"""
Created on Sat Nov 03 2018

@author: Hank
"""

import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize

if __name__=="__main__":
    from matplotlib import pyplot as pl
    pl.rc('text', usetex=True) # Support greek letters in plot legend
    pass




def indef_integral_of_squareroot_quotients_old(a,b,c,x):
    # From Wolfram Alpha: integral of (sqrt(a+bx)/sqrt(c-bx)) dx
    # OBSOLETE (not used anymore)
    return (-2.0*np.sqrt(a+b*x)*np.sqrt(c-b*x)  +  (1.0j)*(a + c)*log(2.0*sqrt(a+b*x)*sqrt(c-b*x)-(1.0j)*(a + 2*b*x -c)))/(2.0*b)


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

# For a shear crack with the tip at the origin, intact material
# to the right (x > 0), broken material to the left (x < 0)
# The shear stress @ theta=0 multiplied by sqrt(x)/(sqrt(a)*tauext)
# where x ( > 0) is the position where the stress is measured,
# a is the (half) length of the crack, and tauext
# is the external shear load
tauII_theta0_times_rootx_over_sqrt_a_over_tauext = 1.0/sqrt(2.0)  # Per Suresh (9.47a) and Anderson (table 2.1)


def integral_shearstress_growing_effective_crack_length_bytau_old(tauext1,tauext2,xtp,D):
    """ OBSOLETE Evaluate the incremental shear stress field on a shear crack
    that is growing in effective length the external load
    increases from tauext1 to tauext2.
    
    It is assumed that the effective tip moves linearly with 
    applied external shear stress, with rate given by D (meters of
    tip motion / Pascals of external shear stress)

    Let tauII(r,K) be the mode II shear stress formula divided by 
    the external load, where the K dependence is presumed to be 
    strictly multiplicative. Let K_over_tauext = K/tauext

    Let xt be the position of the effective tip at external load
    tauext

    The incremental shear stress field would be
    integral_tauext1^tauext2 of tauII(x-xt,K_over_tauext) dtauext
    (note that K and xt are dependent on tauext)
   

    The variable tauII_theta0_times_rootx_over_sqrt_a_over_tauext
    represents the value of tauII(x,K) with tauext*sqrt(pi*a) 
    substituted for K, evaluated for horizontal axis beyond the 
    tip (i.e. theta=0) and then multiplied by sqrt(x) (sqrt(position
    beyond the tip) and divided by sqrt(cracklength) and also by 
    tauext. 

    Then we can rewrite the incremental shear stress as: 
    integral_tauext1^tauext2 of tauII_theta0_times_rootx_over_sqrt_a_over_tauext*sqrt(xt)/sqrt(x-xt) dtauext
    Here, xt is still dependent on tauext... this will give shear stress
    as a function of position (x). 

    We assume xt is linearly dependent on shear stress:
    xt = xtp + D*(tauext-tauext1)
    where xtp is the final xt from the previous step. 

    tauII_theta0_times_rootx_over_sqrt_a_over_tauext is a constant, 
    so it will just be a leading coefficient that we will ignore from 
    here on (except when multiplying it in at the final step). 

    So our incremental shear is
    integral_tauext1^tauext2 sqrt(xt)/sqrt(x-xt) dtauext
    where we ignore any contributions corresponding to (x-xt) <= 0

    ... Substitute for xt:
    integral_tauext1^tauext2 sqrt(xtp+D*tauext-D*tauext1)/sqrt(x-xtp-D*tauext-D*tauext1) dtauext

    ... Substitute a = xtp-D*tauext1, c=x-xtp-D*tauext1
    integral_tauext1^tauext2 sqrt(a + D*tauext)/sqrt(c - D*tauext) dtauext

    This is then the integral of (sqrt(a+bx)/sqrt(c-bx)) dx
       where x = tauext, and b=D
    with solution given by
       indef_integral_of_squareroot_quotients(a,D,c,tauext2) - indef_integral_of_squareroot_quotients(a,D,c,tauext1)

    Well almost. We only consider the region of this integral where 
    x-xt > 0. This can be accomplished by shifting the bounds when 
    needed. 

    Substitute for xt: 
    x - xtp - D*tauext + D*tauext1 > 0
    x - xtp + D*tauext1 > D*tauext
    D assumed positive
    tauext < (x - xtp)/D + tauext1
    
    So use_tauext2 = min( tauext2, (x - xtp)/D + tauext1 )
    
    
    So our actual solution putting everything together is:
    tauII_theta0_times_rootx_over_sqrt_a_over_tauext*(indef_integral_of_squareroot_quotients(a,D,c,use_tauext2) - indef_integral_of_squareroot_quotients(a,D,c,tauext1))

    """
    

    a = xtp-D*tauext1
    c = x-xtp-D*tauext1
    
    use_tauext2 = tauext2*np.ones(x.shape,dtype='d')

    # alternate tauext2:
    alt_tauext2 = (x - xtp)/D + tauext1

    use_alternate = alt_tauext2 < use_tauext2
    use_tauext2[use_alternate] = alt_tauext2[use_alternate]

    
    return tauII_theta0_times_rootx_over_sqrt_a_over_tauext*(indef_integral_of_squareroot_quotients_old(a,D,c,use_tauext2) - indef_integral_of_squareroot_quotients_old(a,D,c,tauext1))


def integral_shearstress_growing_effective_crack_length_byxt(x,tauext1,tauext_max,F,xt1,xt2):
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

    Let tauII(r,K) be the mode II shear stress formula divided by 
    the external load, where the K dependence is presumed to be 
    strictly multiplicative. Let K_over_tauext = K/tauext

    Let xt be the position of the effective tip at external load
    tauext

    The incremental shear stress field would be
    integral_tauext1^tauext2 of tauII(x-xt,K_over_tauext) dtauext
    (note that K and xt are dependent on tauext)
   

    The variable tauII_theta0_times_rootx_over_sqrt_a_over_tauext
    represents the value of tauII(x,K) with tauext*sqrt(pi*a) 
    substituted for K, evaluated for horizontal axis beyond the 
    tip (i.e. theta=0) and then multiplied by sqrt(x) (sqrt(position
    beyond the tip) and divided by sqrt(cracklength) and also by 
    tauext. 

    Then we can rewrite the incremental shear stress as: 
    integral_tauext1^tauext2 of tauII_theta0_times_rootx_over_sqrt_a_over_tauext*sqrt(xt)/sqrt(x-xt) dtauext
    Here, xt is still dependent on tauext... this will give shear stress
    as a function of position (x). 

    We assume xt is linearly dependent on shear stress:
    xt = xtp + (1/F)*(tauext-tauext1)
    where xtp is the final xt from the previous step. 

    tauII_theta0_times_rootx_over_sqrt_a_over_tauext is a constant, 
    so it will just be a leading coefficient that we will ignore from 
    here on (except when multiplying it in at the final step). 

    So our incremental shear is
    integral_tauext1^tauext2 sqrt(xt)/sqrt(x-xt) dtauext
    where we ignore any contributions corresponding to (x-xt) <= 0

    Perform change of integration variable tauext -> xt: 
       Derivative of xt:
       dxt = (1/F)*dtauext
       dtauext = F*dxt


    So the  incremental shear we are solving for is
    integral_xt1^xt2 sqrt(xt)*F/sqrt(x-xt)  dxt
    where we ignore any contributions corresponding to (x-xt) <= 0

    and tauext2 = tauext1 + (xt2-xt1)*F 
 
    F is a constant so have 
    F * integral_xt1^xt2 sqrt(xt)/(sqrt(x-xt))  dxt


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
    (tauII_theta0_times_rootx_over_sqrt_a_over_tauext*F)*(indef_integral_of_simple_squareroot_quotients(x,upper_bound) - indef_integral_of_simple_squareroot_quotients(x,xt1))

    """
    
    # For a shear crack with the tip at the origin, intact material
    # to the right (x > 0), broken material to the left (x < 0)
    # The shear stress @ theta=0 multiplied by sqrt(x)/(sqrt(a)*tauext)
    # where x ( > 0) is the position where the stress is measured,
    # a is the (half) length of the crack, and tauext
    # is the external shear load
    tauII_theta0_times_rootx_over_sqrt_a_over_tauext = 1.0/sqrt(2.0)  # Per Suresh (9.47a) and Anderson (table 2.1)

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

    
    
    res[nonzero] = (tauII_theta0_times_rootx_over_sqrt_a_over_tauext*F) * (indef_integral_of_simple_squareroot_quotients(x[nonzero],upper_bound[nonzero]) - indef_integral_of_simple_squareroot_quotients(x[nonzero],xt1))

    
    
    return (use_xt2,tauext2,res)


def solve_incremental_shearstress(x,x_bnd,tau,sigma_closure,shear_displ,xt_idx,dx,tauext,tauext_max,a,mu,E,nu):
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
        (use_xt2,tauext2,tau_increment)=integral_shearstress_growing_effective_crack_length_byxt(x,tauext,tauext_max,F,x_bnd[xt_idx],next_bound)
        return (tau+tau_increment - mu*sigma_closure)[xt_idx]

    # F measures the closure gradient in (Pascals external shear stress / meters of tip motion)

    if sigma_closure[xt_idx] >= 0.0 and tau[xt_idx] < mu*sigma_closure[xt_idx]:
        # There is a closure stress here but not the full tau it can support

        # Bound it by 0  and the F that will give the maximum
        # contribution of tau_increment: 2.0*(tauext_max-tauext1)/(xt2-xt1)
        Fbnd = 2.0*(tauext_max - tauext)/(next_bound-x_bnd[xt_idx])

        if obj_fcn(Fbnd) < 0.0:
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
        
        (use_xt2,tauext2,tau_increment)=integral_shearstress_growing_effective_crack_length_byxt(x,tauext,tauext_max,F,x_bnd[xt_idx],next_bound)

        # For displacement calculate at x centers... use average of left and right boundaries, except for (perhaps) last point where instead of the right boundary we use the actual tip.
        incremental_displacement = np.zeros(x.shape[0],dtype='d')
        xt = (x_bnd[xt_idx]+use_xt2)/2.0
        left_of_effective_tip = (x < xt)
        incremental_displacement[left_of_effective_tip] = shear_displacement(tauext2-tauext,x[left_of_effective_tip],xt,E,nu)
        pass
    else:
        # No closure stress at this point, or tau is already at the limit
        # of what can be supported here
        
        use_xt2 = x_bnd[xt_idx+1]
        tauext2 = tauext
        tau_increment = np.zeros(x.shape[0],dtype='d')
        incremental_displacement = np.zeros(x.shape[0],dtype='d')
        
    return (use_xt2,tauext2, tau+tau_increment, shear_displ+incremental_displacement)
    
    
#####INTEGRAL OF SHEAR FUNCTION
#Define a function for the integral of the shear function
#Determined from Wolfram Alpha, 
#(integral   sqrt(a+(b*x))/sqrt(c-(b*x))  dx)
#The previously derived integral had four constants, a,b,c,and q
#It was determined that b and q would always equal so it was re-
#calculated to have a slightly nicer integral
#NOTE: I had to use np.lib.scimath.sqrt in all but one constant
#case to allow complex square roots to be calculated
def tau_integral_old(D,x,xtp,Text,Textp):
    # OBSOLETE
    #D = a constant D that is unknown?
    #x = a constant
    #xtp = the crack length from the previous step
    #Text = the external load and the varible integrated w.r.t.
    #Textp = the previous external load step
    #The first term (1/np.sqrt(2)) is a constant that was pulled
    #before integration 
    i = 1j
    return (1/np.sqrt(2))*(((-2*(np.lib.scimath.sqrt((xtp-D*Textp)+ \
    (D*Text)))*(np.lib.scimath.sqrt((x-(xtp-D*Textp))-(D*Text))))/(2*D))+ \
    ((i*((xtp-D*Textp)+(x-(xtp-D*Textp)))* \
    np.log((2*(np.lib.scimath.sqrt((xtp-D*Textp)+(D*Text)))* \
    (np.lib.scimath.sqrt((x-(xtp-D*Textp))-(D*Text))))- \
    (i*((xtp-D*Text)+(2*D*Text)-(x-(xtp-D*Text))))))/(2*D)))



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
#                here, I think... should verify!!! ***
def shear_displacement(tau_applied,x,xt,E,nu):
    #plane stress is considered
    Kappa = (3.0-nu)/(1.0+nu)
    KII = tau_applied*np.sqrt(np.pi*(xt-x))
    if (xt-x < 0).any():
        #sys.modules["__main__"].__dict__.update(globals())
        #sys.modules["__main__"].__dict__.update(locals())
        raise ValueError("Shear displacement to right of crack tip")
    theta = np.pi
    return (KII/(2.0*E))*(np.sqrt((xt-x)/(2.0*np.pi)))*((1.0+nu)* \
    (((2.0*Kappa+3.0)*(np.sin(theta/2.0)))+(np.sin(3.0*theta/2.0))))


def solve_shearstress(x,x_bnd,sigma_closure,dx,tauext_max,a,mu,E,nu,tau_yield,verbose=False):
    #Initialize the external applied dynamic shear stress starting at zero
    
    tauext = 0.0 # External shear load in this step (Pa)
    


    #####MAIN SUPERPOSITION LOOP

    #Initialize shear stress field (function of x)
    tau = np.zeros(x.shape,dtype='d')
    
    #Initialized the Displacement state as zero
    shear_displ = np.zeros(x.shape,dtype='d')

    #Initialize x step counter
    xt_idx = 0

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
        use_xt2=0
        pass

    

    
    done=False

    if tauext==tauext_max:
        # Used up all of our applied load...  Done!
        done=True
        pass

    while not done: 
        
        (use_xt2,tauext, tau, shear_displ) = solve_incremental_shearstress(x,x_bnd,tau,sigma_closure,shear_displ,xt_idx,dx,tauext,tauext_max,a,mu,E,nu)
        
    
        if use_xt2 < x_bnd[xt_idx+1] or tauext==tauext_max or use_xt2 >= a:
            # Used up all of our applied load or all of our crack... Done!
            done=True
            pass

        if verbose: 
            #Print what is happening in the loop
            print("Step: %d @ x=%f mm: %f MPa of shear held" % (xt_idx,x[xt_idx]*1e3,tauext/1e6))
            print("Shear displacement @ x=%f mm: %f nm" % (x[0]*1e3, shear_displ[0]*1e9))
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
        
        #tau_increment = tauII_theta0_times_rootx_over_sqrt_a_over_tauext*(tauext_max-tauext)*sqrt(a)/sqrt(x-a)
        tau_increment[ti_nodivzero_nonegsqrt] = tauII_theta0_times_rootx_over_sqrt_a_over_tauext*(tauext_max-tauext)*sqrt(a)/sqrt(x[ti_nodivzero_nonegsqrt]-a)
        tau_increment[ti_divzero]=np.inf

        # Limit shear stresses at physical tip (and elsewhere) to yield
        tau_increment[tau + tau_increment > tau_yield] = tau_yield-tau[tau+tau_increment > tau_yield]
        
        # accumulate stresses onto tau
        tau += tau_increment

        # record increment in displacement
        left_of_effective_tip = x < a
        shear_displ[left_of_effective_tip] += shear_displacement(tauext_max-tauext,x[left_of_effective_tip],a,E,nu)
        
        # Record increment in tauext
        tauext = tauext_max

        if verbose:
            print("Step: Open to tips @ x=%f mm: %f MPa of shear held" % (a*1e3,tauext/1e6))
            print("Shear displacement @ x=%f mm: %f nm" % (x[0]*1e3, shear_displ[0]*1e9))
            pass
        pass
    
    return (use_xt2, tau, shear_displ)


# align_yaxis from https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


if __name__=="__main__":
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
    
    # Closure state (function of position; positive compression)
    sigma_closure = 80e6/cos(x/a) -70e6 # Pa
    sigma_closure[x > a]=0.0


    (effective_length, tau, shear_displ) = solve_shearstress(x,x_bnd,sigma_closure,dx,tauext_max,a,mu,E,nu,tau_yield,verbose=True)

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
    pl.show()
    pass

    
    
