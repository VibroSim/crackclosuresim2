import os
import os.path
import csv
import sys
import copy
import re
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize
import scipy as sp
import scipy.interpolate


#if __name__=="__main__":
#    from matplotlib import pyplot as pl
#    pl.rc('text', usetex=True) # Support greek letters in plot legend
#    pass


class ModeI_crack_model(object):
    # abstract class
    #
    # Implementations should define:
    #  * methods: 
    #    * eval_sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext(self,a)
    #    * evaluate_ModeI_COD_vectorized(self,sigma_applied,x,xt)  # should be vectorized over x (not necessarily xt)
    pass

class ModeI_Beta_COD_Formula(ModeI_crack_model):
    """This represents a crack model where we are given a formula
    for K_I of the form K_I = sigma*sqrt(pi*a*beta), and
    COD is a function u(object,surface_position,surface_length).


    You can add member variables (which will be accessible from 
    the u function) by providing them as keyword arguments to 
    the constructor. 

    At minimum you must provide a function:
       u(object,surface_position,surface_length)  
       which should be vectorized over surface position, and a function
       beta(object), which return the COD and beta values respectively. 
       (beta is a function, so you can set it up so that the crack model
       will work correctly if its attribute parameters are updated)

"""

    u=None
    beta=None

    
    
    def __init__(self,**kwargs):
        if "u" not in kwargs:
            raise ValueError("Must provide COD function u(object,sigma_applied,surface_position,surface_length)")

        if "beta" not in kwargs:
            raise ValueError("Must provide K coefficient beta(object)")
        

        for kwarg in kwargs:
            setattr(self,kwarg,kwargs[kwarg])
            pass

        pass

    def eval_sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext(self,a):
        # For a mode I crack with the tip at the origin, intact material
        # to the right (x > 0), broken material to the left (x < 0)
        # The tensile stress @ theta=0 multiplied by sqrt(x)/(sqrt(a)*sigmaext)
        # where x ( > 0) is the position where the stress is measured,
        # a is the (half) length of the crack, and sigmaext
        # is the external tensile load

        # Per Suresh (9.43 and 9.44a) and Anderson (table 2.1)
        # and based on K_I=(sigma_ext*sqrt(pi*a*beta))
        # instead of  K_I=(sigma_ext*sqrt(pi*a))

        sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext = sqrt(self.beta(self))/sqrt(2.0)  
                
        return sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext

    def eval_ModeI_COD_vectorized(self,sigma_applied,x,xt):
        return self.u(self,sigma_applied,x,xt)
        
    
    pass



class ModeI_Beta_WeightFunction(ModeI_crack_model):
    """This represents a crack model where we are given a 
    weight function weightfun_times_sqrt_aminx(object, x,a) 
    representing the weight function m(x,a) multiplied
    by sqrt(a-x)

    You can add member variables (which will be accessible from 
    the u function) by providing them as keyword arguments to 
    the constructor. 

    Does NOT assume the weight function is nondimensionalizable,
    so the weight function can have dimensional parameters

    NOTE: Do not change internal parameters after construction 
          if using the surrogate, because the surrogate won't 
          be updated!

"""
    # Settable parameters
    weightfun_times_sqrt_aminx=None
    epsx=None
    Eeff = None   # = E for plane stress, = E/(1-nu^2) for plane strain
    surrogate_a = None  # Range of crack lengths used for surrogate
    use_surrogate=None
    plot_surrogate=None

    # These are lambdas set by the constructor
    K_I_ov_sigma_ext_vect=None
    K_I_ov_sigma_ext_surrogate=None    
    K_I_ov_sigma_ext_use=None
    
    def __init__(self,**kwargs):
        self.use_surrogate=False
        self.plot_surrogate=False
        
        if "weightfun_times_sqrt_aminx" not in kwargs:
            raise ValueError("Must provide singularity-compensated weight function weightfun_time_sqrt_aminx(object,x,a)")

        if "epsx" not in kwargs:
            raise ValueError("Must provide epsilon_x representing the size of the small analytically integrated region around the tip singularity")
        
        if "Eeff" not in kwargs:
            raise ValueError("Must provide Eeff (effective modulus)")

        
        for kwarg in kwargs:
            setattr(self,kwarg,kwargs[kwarg])
            pass

        # Create K_I_ov_sigma_ext_vec and its surrogate
        K_I_ov_sigma_ext = lambda a : scipy.integrate.quad(lambda u : self.weightfun_times_sqrt_aminx(self,u,a)/np.sqrt(a-u),-a,a-self.epsx)[0] + self.weightfun_times_sqrt_aminx(self,a,a)*2.0*sqrt(self.epsx)
        self.K_I_ov_sigma_ext_vect = np.vectorize(K_I_ov_sigma_ext)

        self.K_I_ov_sigma_ext_use = self.K_I_ov_sigma_ext_vect  # overridden by self.use_surrogate below
        
        if self.use_surrogate or self.plot_surrogate:
            # simple splrep surrogate
            K_I_ov_sigma_ext_eval=self.K_I_ov_sigma_ext_vect(self.surrogate_a)

            
            (t1,c1,k1) = sp.interpolate.splrep(self.surrogate_a,K_I_ov_sigma_ext_eval)
            
            
            self.K_I_ov_sigma_ext_surrogate = lambda a: sp.interpolate.splev(a,(t1,c1,k1),ext=2)
            
            surrogate_a_fine=np.linspace(self.surrogate_a[0],self.surrogate_a[-1],self.surrogate_a.shape[0]*4)
    
            if self.plot_surrogate:
                from matplotlib import pyplot as pl
                pl.figure()
                pl.plot(surrogate_a_fine,self.K_I_ov_sigma_ext_vect(surrogate_a_fine),'-',
                        surrogate_a_fine,self.K_I_ov_sigma_ext_surrogate(surrogate_a_fine),'-')
                pl.title("K$_I$ over sigma$_{ext}$")
                pl.legend(("Direct","Surrogate"))
                pass
            
            if self.use_surrogate:
                self.K_I_ov_sigma_ext_use = self.K_I_ov_sigma_ext_surrogate
                pass
            pass
        

        pass

    def eval_sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext(self,a):


        # sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext = ([ integral_-a^(a-epsilon) M(x,a)/sqrt(a-x) dx + M(a,a)*2sqrt(epsilon) ] / sqrt(2*pi))
        
        sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext = (scipy.integrate.quad(lambda x: self.weightfun_times_sqrt_aminx(self,x,a)/np.sqrt(a-x),-a,a-self.epsx)[0] + self.weightfun_times_sqrt_aminx(self,a,a)*2.0*np.sqrt(weightfun_epsx)) / (np.sqrt(2*pi*a))
        # unit check: (should be unitless)
        # Integral of stress*weight function*dx = SIF (i.e. stress*sqrt(meters))
        # units of weight function = 1/sqrt(meters)
        # units of weightfun_times_sqrt_aminx = unitless
        
        # Units of sigmaI_theta0_times_rootr_over_sqrta_over_sigmaext:
        #   ((1/sqrt(meters))*meters + sqrt(meters) ) / sqrt(meters)
        #   = unitless (check)

                
        return sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext

    def eval_ModeI_COD_vectorized(self,sigma_applied,x,xt):
        # we are using weightfunctions

        # New implementation with weight functions:
        # m = (E'/2K) du/da   (Anderson, Fracture Mechanics, eq. 2.50 and Rice (1972) eq. 13
        # u = integral_x..xt  (2K/E') m(x,a) da 
        # u = integral_x..xt  (2K/E') M(x,a)/sqrt(a-x) da 
        
        # need K... well from above         K_I(a) = sigma_ext * [ integral_0^(a-epsilon) M(u,a)/sqrt(a-u) du + M(a,a)*2sqrt(epsilon) ]
        # u = (2.0/E') * integral_x..xt  K_I(a) M(x,a)/sqrt(a-x) da 
        # u = (2.0/E') * [ integral_x..(x+epsilon)  K_I(a) M(x,a)/sqrt(a-x) da  + integral_(x+epsilon)..xt K_I(a) M(x,a)/sqrt(a-x) da ]
        # u = (2.0/E') * [ K_I(x) M(x,x) integral_x..(x+epsilon) 1.0/sqrt(a-x) da  + integral_(x+epsilon)..xt K_I(a) M(x,a)/sqrt(a-x) da ]
        #
        # as above we can evaluate the left hand integral to be 2*sqrt(epsilon)
        # so 
        # u = (2.0/E') * [ K_I(x) M(x,x) 2*sqrt(epsilon)  + integral_(x+epsilon)..a K_I(a) M(x,a)/sqrt(a-x) da ]
        #
        # NOTE: POSSIBLE PROBLEM... Should be dependent on nu? (Poisson's ratio?) 
        
        right_integral = lambda _x : scipy.integrate.quad(lambda a: self.K_I_ov_sigma_ext_use(a)*self.weightfun_times_sqrt_aminx(self,_x,a)/np.sqrt(a-_x),_x+self.epsx,xt)[0]
        
        right_integral_vect = np.vectorize(right_integral)
        
        u = (2.0*sigma_applied/self.Eeff) * ( self.K_I_ov_sigma_ext_use(x)*self.weightfun_times_sqrt_aminx(self,x,x)*2.0*np.sqrt(self.epsx) + right_integral_vect(x))
        
        return u
    
    pass





def indef_integral_of_simple_squareroot_quotients(a,u):
    """ This routine is no longer used because integrated out 
    to infinity the form of solution kernel that goes with this
    fails load balancing... See 
    indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x,xt) 
    for the replacement"""

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

    #if np.count_nonzero(a < u) > 0:
    #    import pdb
    #    pdb.set_trace()
    #    pass
    
    integral = np.zeros(a.shape[0],dtype='d')
    integral[~divzero] = a[~divzero]*arctan(sqrt(u[~divzero])/sqrt(a[~divzero]-u[~divzero])) - sqrt(u[~divzero])*sqrt(a[~divzero]-u[~divzero])

    integral[divzero] = np.pi*a[divzero]/2.0
    return integral

def indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x,xt):
    """
    This is the indefinite integral of the crack tip stress solution for an 
    open linear elastic crack.
         ___
     / \/x_t    /   r0   \2
     | --===- * |--------|  dx_t
     / \/ r     \(r + r0)/
     
    where r is implicitly defined as x - x_t. 

    The first factor represents the standard sqrt(a) divided by the square 
    root of the radius away from the crack decay that is found in standard 
    crack tip stress solutions e.g. Anderson (2004), and Tada (2000). 
    However, this alone does not accurate account for the load balance in 
    of the load that would have been carried by half of the crack surface 
    and the load that would be added ahead of the crack tip. There is 
    presumed to be another constant term outside this integral matching
    the load at infinity. 
    
    The second factor in the integral represents additional decay of the 
    1/sqrt(r) singularity which, combined with the outside constant term)
    enforces the load balance of the stress state as r is integrated to 
    infinity. 
    
    This function of r0 complicates the integral because not only is 
    r = x - x_t a function of x_t (the variable of integration), r0 is also a 
    function of x_t (r0 is presumed to have the form constant*x_t, where 
     this constant will 
    be refered to as b=r0_over_a). 
         
    The resulting integral is:
            ___
     /    \/x_t      /       b*x_t       \2
     | --=======- *  |-------------------|  dx_t
     / \/x - x_t     \((x - x_t) + b*x_t)/
    
    The function inputs are:
    
        crack_model - contains the values describing the particular 1/sqrt(r)
              LEFM tip model desired, including a function returning 
              the r0_over_a value needed for the integral. The assumption
              is that r0_over_a, even though it is given parameters including
              x_t, is not actually dependent on x_t. If there is dependence on
              x_t then this solution is not correct (but may be close enough
              for practical purposes). 

        x  -  the x value or range along the crack that the evaluated 
              integral is being 
              calculated over, not the variable of integration
        
        xt -  the value or range of half crack length that the indefinite
              integral is being evaluated at
        
    
    This function then returns the indefinite integral evaluated at
    (x,x_t)
        
    """
    (x,xt) = np.broadcast_arrays(x,xt) # make sure x and xt are the same shape
    #From Wolfram Alpha: 
    #   integrate ((sqrt(u))/(sqrt(a-u)))*((b*u)/((a-u)+b*u))^2 du =
    #Plain-Text Wolfram Alpha output
    #   (b^2 (-(((-1 + b) Sqrt[a - u] Sqrt[u] (a (1 + b) + (-1 + b) b u))/(b 
    #   (a + (-1 + b) u))) + a (-5 + b) ArcTan[Sqrt[u]/Sqrt[a - u]] + (a (-1 + 
    #   5 b) ArcTan[(Sqrt[b] Sqrt[u])/Sqrt[a - u]])/b^(3/2)))/(-1 + b)^3
    
    #where b*u = r0 --> b = r0_over_a, u = xt, and a = x

    # Calculate division-by-zero and
    # non division-by-zero regimes separately
    
    # Limiting case as x-xt -> 0:
    # Let r = x-xt -> xt = x-r
    #
    # The limit approaches ((b**2)/(b-1)**3)*(pi/2.0)*((x*(5*b-1)/(b**(3./2.)))
    #                               +(x*(b-5))) as r->0
    
    divzero = (x==xt) | ((np.abs(x-xt) < 1e-10*x) & (np.abs(x-xt) < 1e-10*xt))
    
    #if np.count_nonzero(x < xt) > 0:
    #    import pdb
    #    pdb.set_trace()
    #    pass
    
    b = crack_model.r0_over_a(xt)
    
    f1=sqrt(xt[~divzero])
    f2=sqrt(x[~divzero]-xt[~divzero])

    A=((b**2)/(b-1)**3)
    B=((x[~divzero]*(5*b-1)*arctan((sqrt(b)*f1)/(f2)))/(b**(3./2.)))
    C=((b-1)*(f1)*(f2)*(x[~divzero]*(b+1)+(b-1)*b*xt[~divzero]))
    D=(b*(x[~divzero]+(b-1)*xt[~divzero]))
    E=(x[~divzero]*(b-5)*arctan(f1/f2))

    integral = np.zeros(x.shape[0],dtype='d')
    integral[~divzero] =A*(B-(C/D)+E)
    
    integral[divzero] = ((b**2)/(b-1)**3)*(pi/2.0)*x[divzero]*(((5*b-1)/(b**(3./2.)))+(b-5))
    return integral



def integral_tensilestress_growing_effective_crack_length_byxt(x,sigmaext1,sigmaext_max,F,xt1,xt2,crack_model):
    """ Evaluate the incremental normal stress field on a mode I crack
    that is growing in effective length from xt1 to xt2 due to an external 
    load (previous value sigmaext1, limiting value sigmaext_max)
    
    It is assumed that the effective tip moves linearly with 
    applied external normal stress, with rate given by F (Pascals 
    of external normal stress / meters of tip motion 

    The external normal stress is presumed to be bounded by sigmaext_max
    (which may be np.Inf to leave it unbounded). In such a case
    it will evaluate the incremental normal stress only up to 
    the length that gives sigmaext_max.

    Returns (use_xt2,sigmaext2,res) where use_xt2 is the actual 
    upper bound of the integration (as limited by sigmaext_max), 
    sigmaext2 is the external tensile stress load corresponding 
    to the crack being opened to use_xt2, and which will be 
    <= to sigmaext_max. 
    
    Rationale: 
     The mode I normal stress formula is:
      sigma_yy_crack = (K_I / sqrt(2*pi*r))  (Suresh, Eq. 9.44a at theta=0)

    ... we choose to add in the external field not being held by the crack
      sigma_yy_total = (K_I / sqrt(2*pi*r)) + sigma_ext

     
    In the region where the stress accumulates, to the right of the tip, 
    the origin of the K_I is almost irrelevant. So we can use this
    formula even for different geometries/loading conditions. 

    Using the weight function to find K, 
       K_I = integral_0^a (sigma_ext(x) m(x,a) dx)
    For uniform loading, sigma_ext = independent of x, therefore
       K_I = sigma_ext * integral_0^a m(x,a) dx
    ... 
    now m(x,a) has the form M(x,a)/sqrt(a-x) 
    
    M(x,a) is weightfun_times_sqrt_aminx

    Break the integral into two pieces: 

       K_I = sigma_ext * [ integral_0^(a-epsilon) M(x,a)/sqrt(a-x) dx + integral_(a-epsilon)^a M(x,a)/sqrt(a-x) dx ]
    Evaluate the left hand integral with quadrature integration. 
    Evaluate the right hand integral analytically: 

    integral_(a-epsilon)^a M(x,a)/sqrt(a-x) dx

    Treat M(x,a) as constant M(a,a) over the small region
     = M(a,a) * integral_(a-epsilon)^a 1/sqrt(a-x) dx

     Let u = a-x : du=-dx
     = M(a,a) * -integral_epsilon^0 u^(-1/2) du
     = M(a,a) * -2u^(1/2) |_epsilon^0
     = M(a,a) * 2epsilon^(1/2) = 2 sqrt(epsilon)

    So
      K_I = sigma_ext * [ integral_0^(a-epsilon) M(x,a)/sqrt(a-x) dx + M(a,a)*2sqrt(epsilon) ]

    Now K_I/sigmaext = [ integral_0^(a-epsilon) M(x,a)/sqrt(a-x) dx + M(a,a)*2sqrt(epsilon) ]

    Let K_over_sigmaext = K_I/sigmaext

 
    From above.  sigma_yy_crack = (K_I / sqrt(2*pi*r)) ... Call sigma_yy_crack/sigmaext now sigmaI and make it a function of r=x-xt and K_I/sigmaext


    So sigmaI(r,K_I/sigmaext) = K_over_sigmaext / sqrt(2*pi*(x-xt))

      
    Let xt be the position of the effective tip at external load
    sigmaext

    The incremental normal stress field would be
    integral_sigmaext1^sigmaext2 of 1.0 + sigmaI(x-xt,K_over_sigmaext) dsigmaext
    (note that K and xt are dependent on sigmaext)
   

    The variable sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext
    represents the value of sigmayy_crack(x,K) with the above formula for K_I 
    (either from the weight function or the simple K_I=sigma_ext*sqrt(pi*a))
    substituted for K, evaluated for horizontal axis beyond the 
    tip (i.e. theta=0) and then multiplied by sqrt(r) (sqrt(position
    beyond the tip) and divided by sqrt(cracklength) and also by 
    sigmaext. 

    So sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext = ([ integral_0^(a-epsilon) M(x,a)/sqrt(a-x) dx + M(a,a)*2sqrt(epsilon) ] / (sqrt(2*pi*a)))

    Then we can rewrite the incremental normal stress as: 
    integral_sigmaext1^sigmaext2 of sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*sqrt(xt)/sqrt(x-xt) dsigmaext
    if the entire stress field is presumed to come from the crack tip
    singularity. 
    Here, xt is still dependent on sigmaext... sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext also has 
    some dependence on a (i.e. xt), but as xt isn't moving much this is presumed to be small 

    if we assume that a long way from the effective tip the stress is 
    unform, we add 1.0 into the integral
    
    integral_sigmaext1^sigmaext2 of 1.0 + sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*sqrt(xt)/sqrt(x-xt) dsigmaext

    From force balancing (see total_load_matching.pdf) but using a 
    (r0^2/(r+r0)^2) decay factor (see total_load_matching_crossterm_r2_work.pdf)
    we can apply that decay factor to the singular term, 

    integral_sigmaext1^sigmaext2 of 1.0 + (r0^2/(r+r0)^2)*sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*sqrt(xt)/sqrt(x-xt) dsigmaext

    Here r is defined as sqrt(x-xt), and r0 -- evaluated per the above 
    .pdfs -- is 8*xt/(pi^2*beta) (through crack, beta typically 1) or 
    (2^(1/3))*xt/(pi^(2/3)*beta^(1/3)) (half penny surface crack, 
    beta typically 4/(pi^2).  ... because r and r0 are functions of 
    xt, they are implicitly dependent on xt and need to be 
    considered in the integration. 
    
    this will give normal stress
    as a function of position (x). 

    We assume xt is linearly dependent on normal stress:
    xt = xtp + (1/F)*(sigmaext-sigmaext1)
    where xtp is the final xt from the previous step. 
    
    sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext is treated as constant, 
    between xt1 and xt2
    
    So our incremental tension is
    integral_sigmaext1^sigmaext2 (1.0 + sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext r0^2/(r+r0)^2 * sqrt(xt)/sqrt(x-xt)) dsigmaext
    where we ignore any contributions corresponding to (x-xt) <= 0

    (the 1.0 term represents that beyond the effective tip the external 
    load directly increments the stress state, in addition to the stress 
    concentration caused by the presence of the open region; the 
    r0^2/(r+r0)^2 actor makes the stress concentration integrate to the 
    right load -- with r0 (proportional to xt) selected on that basis
    as discussed above and in the previously mentioned .pdfs.  

    representing r0 as r0_over_xt*xt, and r by x-xt, the r0^2/(r+r0)^2 factor
    becomes r0_over_xt^2*xt^2/(x-xt+r0_over_xt*xt)^2
    
    pull out constant term

    (sigmaext2-sigmaext1) + integral_sigmaext1^sigmaext2 sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext r0_over_xt^2*xt^2/(x-xt+r0_over_xt*xt)^2 * sqrt(xt)/sqrt(x-xt) dsigmaext


    Perform change of integration variable sigmaext -> xt: 
       Derivative of xt:
       dxt = (1/F)*dsigmaext
       dsigmaext = F*dxt


    So the  incremental normal stress we are solving for is
    (sigmaext2-sigmaext1) + integral_xt1^xt2 sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext r0_over_xt^2*xt^2/(x-xt+r0_over_xt*xt)^2   sqrt(xt)*F/sqrt(x-xt)  dxt
    where we ignore any contributions corresponding to (x-xt) <= 0

    and sigmaext2 = sigmaext1 + (xt2-xt1)*F 
 
    F is a constant and sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext is treated as constant so have 
    F*(xt2-xt1)  +  F * sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext * integral_xt1^xt2 r0_over_xt^2*xt^2/(x-xt+r0_over_xt*xt)^2   sqrt(xt)/(sqrt(x-xt))  dxt

    The right hand term is then the integral of 
      r0_over_xt^2*xt^2/(x-xt+r0_over_xt*xt)^2  * (sqrt(xt)/sqrt(x-xt)) dxt

      with solution to the indefinite integral given by 
       indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x,xt)

      so the definite integral is given by 
        indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x,xt2) - indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x,xt1)

    Well almost. We only consider the region of this integral where 
    x-xt > 0. This can be accomplished by shifting the bounds when 
    needed. 

    x > xt
     =>
    xt2 < x  and xt1 < x  ... xt1 < xt2

    So: Integral = 0 where x < xt1
    Integral upper bound =  x where xt1 < x < xt2
    Integral upper bound = xt2 where x > xt2

       indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x,upper_bound) - indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x,xt1)
    
    So our actual solution putting everything together is:
    0 where x < xt1 
    otherwise: 
    upper_bound = min(x, xt2) 
    F*(upper_bound-xt1) + (sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*F)*(indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x,upper_bound) - indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x,xt1))

    """
    
    # For a mode I tension crack with the tip at the origin, intact material
    # to the right (x > 0), broken material to the left (x < 0)
    # The tensile stress @ theta=0 multiplied by sqrt(x)/(sqrt(a)*sigmaext)
    # where x ( > 0) is the position where the stress is measured,
    # a is the (half) length of the crack, and sigmaext
    # is the external tensile load

    sigmaext2 = sigmaext1 + (xt2-xt1)*F

    #print("sigmaext1 = %g; sigmaext2=%g; sigmaext_max=%g; xt1=%g; xt2=%g; F=%g" % (sigmaext1,sigmaext2,sigmaext_max,xt1,xt2,F))

    
    use_xt2 = xt2
    if sigmaext2 > sigmaext_max:
        # bound sigmaext by sigmaext_max... by limiting xt2
        if F > 0:
            use_xt2 = xt1 + (sigmaext_max-sigmaext1)/F
            pass
        if F==0 or use_xt2 > xt2:
            use_xt2 = xt2
            pass
        
        sigmaext2 = sigmaext_max
        pass
    
    
    #print("use_xt2 = %g" % (use_xt2))
    upper_bound = use_xt2*np.ones(x.shape,dtype='d')
    
    # alternate upper_bound:
    use_alternate = x < upper_bound
    upper_bound[use_alternate] = x[use_alternate]
    
    res=np.zeros(x.shape,dtype='d')

    nonzero = x > xt1

    xtavg = (xt1+use_xt2)/2.0


    sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext = crack_model.eval_sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext(xtavg)
    
    
    # KI/(sigma_ext*sqrt(a))    
    # evaluated from basic formula: 
    #  KI = sigma_ext * sqrt(pi*a)   from Suresh
    #  KI/(sigma_ext*sqrt(a)) = sqrt(pi) = 1.77

    # KI/(sigma_ext*sqrt(a))    
    # evaluated via basic weightfunction:
    # KI/(sigma_ext*sqrt(a)) = (integral of the weightfunction from -a..a)/(sigma_ext*sqrt(a))    
    #
    #  For basic weightfunction from Fett and Munz: 
    #  m = sqrt(1/(pi*a)) * sqrt(a+x)/sqrt(a-x)
    # This would be: (1/(a*sqrt(pi))) integral_-a^a sqrt(a+x)/sqrt(a-x) dx
    # let u = x/a; du = dx/a -> dx=a*du
    #   This would be: (1/(a*sqrt(pi))) integral_-1^1 sqrt(a+au)/sqrt(a-au) a*du
    #   This would be: (1/(sqrt(pi))) integral_-1^1 sqrt(1+u)/sqrt(1-u) du
    #  ... = (1/sqrt(pi)) * pi = sqrt(pi) by wolfram alpha ... CHECK!
 

    # This is the (integral of the weightfunction from 0..a)/sqrt(2*pi*a)
    # For basic weightfunction: sqrt(1/(pi*a)) * sqrt(a+x)/sqrt(a-x)
    #   This would be: (1/(a*pi*sqrt(2))) integral_0^a sqrt(a+x)/sqrt(a-x) dx
    # let u = x/a; du = dx/a -> dx=a*du
    #   This would be: (1/(a*pi*sqrt(2))) integral_0^1 sqrt(a+au)/sqrt(a-au) a*du
    #   This would be: (1/(pi*sqrt(2))) integral_0^1 sqrt(1+u)/sqrt(1-u) du
    #  ... = (2+pi)/(2*pi*sqrt(2))... = .578 by wolfram alpha
    #  ... or a factor of (2+pi)/(2*pi)=.818 smaller than the .707
    # of the simple formula

    # old version that fails load balance 
    #res[nonzero] = F*(upper_bound[nonzero]-xt1)  +  (sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*F) * (indef_integral_of_simple_squareroot_quotients(x[nonzero],upper_bound[nonzero]) - indef_integral_of_simple_squareroot_quotients(x[nonzero],xt1))

    res[nonzero] = F*(upper_bound[nonzero]-xt1)  +  (sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*F) * (indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x[nonzero],upper_bound[nonzero]) - indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x[nonzero],xt1))

    

    return (use_xt2,sigmaext2,res)






def integral_compressivestress_shrinking_effective_crack_length_byxt(x,sigmaext1,sigmaext_max,F,xt1,xt2,crack_model):
    """ Like integral_tensilestress_growing_effective_crack_length_byxt()
but for compression. """

    # sigmaext1 is starting load (negative for compression)
    # sigmaext_max is limit load (negative for compression)

    # sigmaext2 is the ending load (negative for compression,
    # more negative than sigmaext1
    
    # For a mode I tension crack with the tip at the origin, intact material
    # to the right (x > 0), broken material to the left (x < 0)
    # The tensile stress @ theta=0 multiplied by sqrt(x)/(sqrt(a)*sigmaext)
    # where x ( > 0) is the position where the stress is measured,
    # a is the (half) length of the crack, and sigmaext
    # is the external tensile load

    # ... F is positive
    
    sigmaext2 = sigmaext1 - (xt2-xt1)*F

    #print("sigmaext1 = %g; sigmaext2=%g; sigmaext_max=%g; xt1=%g; xt2=%g; F=%g" % (sigmaext1,sigmaext2,sigmaext_max,xt1,xt2,F))

    
    use_xt1 = xt1
    if sigmaext2 < sigmaext_max:
        # bound sigmaext by sigmaext_max... by limiting xt1
        if F > 0:
            use_xt1 = xt2 + (sigmaext_max-sigmaext1)/F
            pass
        if F==0 or use_xt1 < xt1:
            use_xt1 = xt1
            pass
        
        sigmaext2 = sigmaext_max
        pass
    
    
    #print("use_xt1 = %g" % (use_xt1))
    upper_bound = xt2*np.ones(x.shape,dtype='d')
    
    nonzero = x > use_xt1
    # alternate upper_bound:
    use_alternate = x < upper_bound
    upper_bound[use_alternate] = x[use_alternate]
    
    res=np.zeros(x.shape,dtype='d')


    xtavg = (use_xt1+xt2)/2.0


    sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext = crack_model.eval_sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext(xtavg)
    
    # KI/(sigma_ext*sqrt(a))    
    # evaluated from basic formula: 
    #  KI = sigma_ext * sqrt(pi*a)   from Suresh
    #  KI/(sigma_ext*sqrt(a)) = sqrt(pi) = 1.77

    # KI/(sigma_ext*sqrt(a))    
    # evaluated via basic weightfunction:
    # KI/(sigma_ext*sqrt(a)) = (integral of the weightfunction from -a..a)/(sigma_ext*sqrt(a))    
    #
    #  For basic weightfunction from Fett and Munz: 
    #  m = sqrt(1/(pi*a)) * sqrt(a+x)/sqrt(a-x)
    # This would be: (1/(a*sqrt(pi))) integral_-a^a sqrt(a+x)/sqrt(a-x) dx
    # let u = x/a; du = dx/a -> dx=a*du
    #   This would be: (1/(a*sqrt(pi))) integral_-1^1 sqrt(a+au)/sqrt(a-au) a*du
    #   This would be: (1/(sqrt(pi))) integral_-1^1 sqrt(1+u)/sqrt(1-u) du
    #  ... = (1/sqrt(pi)) * pi = sqrt(pi) by wolfram alpha ... CHECK!
 

    # This is the (integral of the weightfunction from 0..a)/sqrt(2*pi*a)
    # For basic weightfunction: sqrt(1/(pi*a)) * sqrt(a+x)/sqrt(a-x)
    #   This would be: (1/(a*pi*sqrt(2))) integral_0^a sqrt(a+x)/sqrt(a-x) dx
    # let u = x/a; du = dx/a -> dx=a*du
    #   This would be: (1/(a*pi*sqrt(2))) integral_0^1 sqrt(a+au)/sqrt(a-au) a*du
    #   This would be: (1/(pi*sqrt(2))) integral_0^1 sqrt(1+u)/sqrt(1-u) du
    #  ... = (2+pi)/(2*pi*sqrt(2))... = .578 by wolfram alpha
    #  ... or a factor of (2+pi)/(2*pi)=.818 smaller than the .707
    # of the simple formula

    # old version that fails load balance 
    #res[nonzero] = -F*(upper_bound[nonzero]-use_xt1)  -  (sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*F) * (indef_integral_of_simple_squareroot_quotients(x[nonzero],upper_bound[nonzero]) - indef_integral_of_simple_squareroot_quotients(x[nonzero],use_xt1))

    res[nonzero] = -F*(upper_bound[nonzero]-use_xt1)  -  (sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*F) * (indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x[nonzero],upper_bound[nonzero]) - indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay(crack_model,x[nonzero],use_xt1))


    
    

    return (use_xt1,sigmaext2,res)




def solve_incremental_tensilestress(x,x_bnd,sigma,sigma_closure,tensile_displ,xt_idx,dx,sigmaext,sigmaext_max,a,crack_model,calculate_displacements=True):
    """The overall crack opening constraint is that
    (tensile load on crack surface) > 0 to open
    For a through-crack of thickness h, short segment of width dx
    sigma*h*dx - sigma_closure*h*dx > 0 (where sigma represents the total local stress field increment due to the external load and
                                          sigma_closure represents the closure stresses prior to such increment (positive for compression))

    or equivalently 
    sigma > sigma_closure (where sigma_closure is positive (i.e. compressive))

    Consider an increment in position dx. 
    Assume from previous steps we have a superimposed 
    sigma(x) in the plane of the crack. In this step we are 
    adding an increment to sigma_external. 

    Stresses accumulate strictly to the right of the effective tip. 

    The rule for the crack to remain closed is that 
    (the preexisting sigma(x) + the increment in sigma(x)) <= sigma_closure

    Here, given the preexisting sigma(x), sigma_closure(x),  and an increment 
    of opening the  crack by one unit of dx, we are evaluating the 
    increment in sigma(x) as well as the increment in sigma_external
    
    We can evaluate the increment in sigma from: 

    (use_xt2,sigmaext2,sigma_increment)=integral_tensilestress_growing_effective_crack_length_byxt(x,sigmaext,sigmaext_max,F,x[xt_idx],x[xt_idx+1],weightfun_times_sqrt_aminx,weightfun_epsx)

    But to do this we need to solve for F. 

    We do this by setting the tensile normal stress equal to the closure stress
    over the new step (unit of dx).
    """

    next_bound = x_bnd[xt_idx+1]
    if next_bound > a:
        next_bound=a
        pass
    
    def obj_fcn(F):
        (use_xt2,sigmaext2,sigma_increment)=integral_tensilestress_growing_effective_crack_length_byxt(x,sigmaext,np.inf,F,x_bnd[xt_idx],next_bound,crack_model)
        #print("obj_fcn return %g" % ((sigma+sigma_increment - sigma_closure)[xt_idx]))
        return (sigma+sigma_increment - sigma_closure)[xt_idx]
    
    # F measures the closure gradient in (Pascals external tensile stress / meters of tip motion)
    
    if sigma_closure[xt_idx] >= 0.0 and sigma[xt_idx] < sigma_closure[xt_idx]:
        # There is a closure stress here but not yet the full external tensile load to counterbalance it

        # Bound it by 0  and the F that will give the maximum
        # contribution of sigma_increment: 2.0*(sigmaext_max-sigmaext1)/(xt2-xt1)
        if np.isinf(sigmaext_max):
            if sigmaext != 0.0:
                Fbnd = 2.0*(sigmaext)/(next_bound-x_bnd[xt_idx])
                pass
            else:
                Fbnd = 2.0*(20e6)/(next_bound-x_bnd[xt_idx])
                pass
            
            pass
        else:
            Fbnd = 2.0*(sigmaext_max - sigmaext)/(next_bound-x_bnd[xt_idx])
            pass
        # Increase Fbnd until we get a positive result from obj_fcn
        while Fbnd != 0.0 and obj_fcn(Fbnd) < 0.0:
            Fbnd*=2.0;
            pass

        # Condition below should only occur when Fbnd==0.0, i.e. when sigmaext_max==sigmaext, or if the objective is already satisfied
        if Fbnd == 0.0 or obj_fcn(Fbnd) <= 0.0:
            # Maximum value of objective is < 0... This means that
            # with the steepest sigma vs. xt slope possible (given
            # the total tensile load we are applying) we still
            # can't get sigma+sigma_increment to match sigma_closure.
            # ... We will have to make do with sigma+sigma_increment
            #  < sigma_closure
            # So our best result is just Fbnd
            F=Fbnd

            #print("obj_fcn(Fbnd) returns %g; obj_fcn(200*Fbnd) returns %g" % (obj_fcn(Fbnd),obj_fcn(200*Fbnd)))
                  
            pass
        else:
            # brentq requires function to be different signs
            # at 0.0 (negative) and Fbnd (positive) 
            F = scipy.optimize.brentq(obj_fcn,0.0,Fbnd,disp=True)
            pass
        
        (use_xt2,sigmaext2,sigma_increment)=integral_tensilestress_growing_effective_crack_length_byxt(x,sigmaext,sigmaext_max,F,x_bnd[xt_idx],next_bound,crack_model)
        #print("F=%g" % (F))
        #print("use_xt2=%f" % (use_xt2))
        assert(use_xt2 <= a)

        
        # For displacement calculate at x centers... use average of left and right boundaries, except for (perhaps) last point where instead of the right boundary we use the actual tip.
        if calculate_displacements:
            incremental_displacement = np.zeros(x.shape[0],dtype='d')
            xt = (x_bnd[xt_idx]+use_xt2)/2.0
            left_of_effective_tip = (x < xt)
            incremental_displacement[left_of_effective_tip] = tensile_displacement(sigmaext2-sigmaext,x[left_of_effective_tip],xt,crack_model)
            pass
        
        pass
    else:
        # No closure stress at this point, or sigma is already at the limit
        # of what can be supported here

        # ... just open up to the next spot
        use_xt2 = x_bnd[xt_idx+1]

        if use_xt2 > a:
            # Cannot open beyond tips
            use_xt2 = a
            pass
            
        sigmaext2 = sigmaext
        sigma_increment = np.zeros(x.shape[0],dtype='d')
        incremental_displacement = np.zeros(x.shape[0],dtype='d')
        pass

    if calculate_displacements:
        ret_displ = tensile_displ+incremental_displacement
        pass
    else:
        ret_displ=None
        pass

    dsigmaext_dxt = (sigmaext2-sigmaext)/(x_bnd[xt_idx+1]-x_bnd[xt_idx])
    
    #print("dsigmaext_dxt=%g" % (dsigmaext_dxt))
    
    return (use_xt2,sigmaext2, sigma+sigma_increment, ret_displ, dsigmaext_dxt) 
   



def solve_incremental_compressivestress(x,x_bnd,sigma,sigma_closure,tensile_displ,use_xt2,xt_idx,dx,sigmaext,sigmaext_max,a,sigma_yield,crack_model,calculate_displacements=True):
    """Like solve_incremental_tensilestress but for negative sigmaext and sigmaext_max    """

    next_bound = x_bnd[xt_idx]
    if next_bound < 0.0:
        next_bound=0.0
        pass
    
    def obj_fcn(F):
        (use_xt1,sigmaext2,sigma_increment)=integral_compressivestress_shrinking_effective_crack_length_byxt(x,sigmaext,-np.inf,F,next_bound,use_xt2,crack_model)
        #print("obj_fcn return %g" % ((sigma+sigma_increment - sigma_closure)[xt_idx]))
        return (sigma+sigma_increment - sigma_closure)[xt_idx]

    #if use_xt2 >= 1.43216e-3 and use_xt2 <= 1.4572e-3:
    #    print("Problem spot!")
    #    sys.modules["__main__"].__dict__.update(globals())
    #    sys.modules["__main__"].__dict__.update(locals())
    #    raise ValueError("Problem")
    #    pass
    
    # F measures the closure gradient in (Pascals external tensile stress / meters of tip motion)
    
    if sigma_closure[xt_idx] <= 0.0 and sigma[xt_idx] > sigma_closure[xt_idx]:
        # There is not the full external compressive load to close the crack here...

        # Bound it by 0  and the F that will give the maximum
        # contribution of sigma_increment: 2.0*(sigmaext_max-sigmaext1)/(xt2-xt1)
        # (F is positive, in general... next_bound is smaller than use_xt2)
        if np.isinf(sigmaext_max): # sigmaext_max is -inf when we are closing the crack all the way to find out opening displacement
            Fbnd = 2.0*(-sigma_yield)/(next_bound-use_xt2)
            pass
        else:
            Fbnd = 2.0*(sigmaext_max - sigmaext)/(next_bound-use_xt2)
            pass
        
        # Increase Fbnd until we get a negative result from obj_fcn
        while Fbnd != 0.0 and obj_fcn(Fbnd) > 0.0:
            Fbnd*=2.0;
            pass

        # Condition below should only occur when Fbnd==0.0, i.e. when sigmaext_max==sigmaext, or if the objective is already satisfied
        if Fbnd ==0.0 or obj_fcn(Fbnd) >= 0.0:
            # Maximum value of objective is < 0... This means that
            # with the steepest sigma vs. xt slope possible (given
            # the total tensile load we are applying) we still
            # can't get sigma+sigma_increment to match sigma_closure.
            # ... We will have to make do with sigma+sigma_increment
            #  < sigma_closure
            # So our best result is just Fbnd
            F=Fbnd

            #print("obj_fcn(Fbnd) returns %g; obj_fcn(200*Fbnd) returns %g" % (obj_fcn(Fbnd),obj_fcn(200*Fbnd)))
                  
            pass
        else:
            # brentq requires function to be different signs
            # at 0.0 (negative) and Fbnd (positive) 
            F = scipy.optimize.brentq(obj_fcn,0.0,Fbnd,disp=True)
            pass
        
        (use_xt1,sigmaext2,sigma_increment)=integral_compressivestress_shrinking_effective_crack_length_byxt(x,sigmaext,sigmaext_max,F,next_bound,use_xt2,crack_model)
        #print("use_xt1=%f" % (use_xt1))
        assert(use_xt1 >= 0.0)

        
        # For displacement calculate at x centers... use average of left and right boundaries, except for (perhaps) last point where instead of the right boundary we use the actual tip.
        if calculate_displacements:
            incremental_displacement = np.zeros(x.shape[0],dtype='d')
            xt = (use_xt1+use_xt2)/2.0
            left_of_effective_tip = (x < xt)
            incremental_displacement[left_of_effective_tip] = tensile_displacement(sigmaext2-sigmaext,x[left_of_effective_tip],xt,crack_model)
            pass
        
        pass
    else:
        # This region has enough stress to close
        # ... just close it up to the next spot
        use_xt1 = next_bound

        if use_xt1 < 0.0:
            # Cannot close beyond the center
            use_xt1 = 0.0
            pass
        
        sigmaext2 = sigmaext
        sigma_increment = np.zeros(x.shape[0],dtype='d')
        incremental_displacement = np.zeros(x.shape[0],dtype='d')
        pass

    if calculate_displacements:
        ret_displ = tensile_displ+incremental_displacement
        pass
    else:
        ret_displ=None
        pass

    
    # Limit compressive stresses at physical tip (and elsewhere) to yield
    sigma_increment[sigma + sigma_increment < -sigma_yield] = -sigma_yield-sigma[sigma+sigma_increment < -sigma_yield]


    #assert((sigma+sigma_increment <= 0.0).all())

    dsigmaext_dxt = (sigmaext2-sigmaext)/(x_bnd[xt_idx+1]-x_bnd[xt_idx])

    
    return (use_xt1,sigmaext2, sigma+sigma_increment, ret_displ,dsigmaext_dxt) 
   



#####TENSILE DISPLACEMENT FUNCTION

def tensile_displacement(sigma_applied,x,xt,crack_model):
    ##plane stress is considered

    u = crack_model.eval_ModeI_COD_vectorized(sigma_applied,x,xt)
    #if (xt > 1e-3):
    #    sys.modules["__main__"].__dict__.update(globals())
    #    sys.modules["__main__"].__dict__.update(locals())
    #    raise ValueError("xt exceeds 1mm")
    
    return u




def solve_normalstress_tensile(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,verbose=False, diag_plots=False,calculate_displacements=True):
    #Initialize the external applied tensile stress starting at zero
    
    sigmaext = 0.0 # External tensile load in this step (Pa)

    #if sigmaext_max==0.0:
    #    sys.modules["__main__"].__dict__.update(globals())
    #    sys.modules["__main__"].__dict__.update(locals())
    #    raise ValueError("Zero external load")
        

    #####MAIN SUPERPOSITION LOOP

    #Initialize tensile stress field (function of x)
    sigma = np.zeros(x.shape,dtype='d')
    
    #Initialized the Displacement state as zero
    if calculate_displacements:
        tensile_displ = np.zeros(x.shape,dtype='d')
        pass
    else:
        tensile_displ = None
        pass
    
    #Initialize x step counter
    xt_idx = 0
    
    use_xt2=0.0
    
    # Before opening, sigma just increases uniformly
    # (Note: stress distribution may not be very accurate if
    # initial opening does not occur @ x=0)
    argmin_sigma_closure = np.argmin(sigma_closure[x < a])
    min_sigma_closure=sigma_closure[x < a][argmin_sigma_closure]
    if min_sigma_closure > 0:
        # We can hold a compressive stress of min_sigma_closure
        # without any opening at all.

        uniform_tension = np.min((min_sigma_closure,sigmaext_max))
        
        sigma += uniform_tension
        sigmaext += uniform_tension

        # assume anything to the left of the
        # sigma_closure minimum is open
        # once we get to this point
        #xt_idx=argmin_sigma_closure
        #assert(xt_idx==0) # for now, do not yet handle cases where crack starts peeling open anywhere but the center
        xt_idx=0
        

        use_xt_start=x_bnd[xt_idx]        
        use_xt2 = use_xt_start
        
        pass
    elif min_sigma_closure <= 0:
        # There is an opening point...

        # Find where sigma_closure goes from negative (tensile)
        # to positive (compressive)
        
        signchange_idxs = np.where((sigma_closure[x < a][:-1] <= 0.0) & (sigma_closure[x < a][1:] > 0.0))[0]

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
            use_xt_start=x[xt_idx+1]-sigma_closure[xt_idx+1]/closure_slope        
            use_xt2 = use_xt_start
            
            
            pass
        else:
            # No signchange
            
            if sigma_closure[x<a][-1] > 0.0:
                # have compressive (positive) closure stresses, but no signchange
                # ... crack must be fully closed
                xt_idx=0
                use_xt_start=0.0
                use_xt2=0.0
                pass
            else:
                # crack must be fully open

                xt_idx = np.where(x < a)[0][-1] # open all the way to tip
                # if closure stress is tensile everywhere
                use_xt_start = a
                use_xt2 = a
                pass
            pass
        

        
        ## Use the rightmost opening point (closest to physical tip)
        #xt_idx=xt_idxs[-1]
        #
        #if (sigma_closure[:xt_idx] > 0.0).any():
        #    sys.modules["__main__"].__dict__.update(globals())
        #    sys.modules["__main__"].__dict__.update(locals())
        #    raise ValueError("foo!")
        #    sys.stderr.write("crackclosuresim2.crackclosure.solve_normalstress_tensile(): WARNING: Multiple opening points!\n")
        #    pass
        
        pass
    else:
        assert(0) # shouldn't be possible
        pass
    
    done=False

    
    dsigmaext_dxt = np.ones(x.shape,dtype='d')*np.nan  # dsigmaext_dxt is a measure of the distributed stress concentration
    
    while not done and sigmaext < sigmaext_max: 
        
        (use_xt2,sigmaext, sigma, tensile_displ, dsigmaext_dxt[xt_idx]) = solve_incremental_tensilestress(x,x_bnd,sigma,sigma_closure,tensile_displ,xt_idx,dx,sigmaext,sigmaext_max,a,crack_model,calculate_displacements=calculate_displacements)
        
        
        if use_xt2 < x_bnd[xt_idx+1] or sigmaext==sigmaext_max or use_xt2 >= a:
            # Used up all of our applied load or all of our crack... Done!
            done=True
            pass
        
        if verbose: 
            #Print what is happening in the loop
            print("Step: %d @ x=%f mm: %f MPa of tension held" % (xt_idx,x[xt_idx]*1e3,sigmaext/1e6))
            if calculate_displacements:
                print("Tensile displacement @ x=%f mm: %f nm" % (x[0]*1e3, tensile_displ[0]*1e9))
                pass
            pass

        if not done:
            # loop back
            xt_idx+=1
            pass
        
        
        pass

    if sigmaext < sigmaext_max and not np.isinf(sigmaext_max):
        # We opened the crack to the tips without providing
        # the full external load.
        # Now the effective tip is the physical tip (at a)
        #
        # ... Apply the remaining load increment
        assert(use_xt2 == a)
        
        sigma_increment = np.zeros(x.shape[0],dtype='d')
        si_nodivzero_nonegsqrt = x-a > 1e-10*a
        si_divzero = (x-a >= 0) & ~si_nodivzero_nonegsqrt
        
        #sigma_increment = sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*(sigmaext_max-sigmaext)*sqrt(a)/sqrt(x-a)

        sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext = crack_model.eval_sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext(a)

        # New (sigmaext_max - sigmaext) term is the incremental external  stress field beyond the tips added in addition to the stress contcentration effect
        sigma_increment[si_nodivzero_nonegsqrt] = (sigmaext_max - sigmaext) + sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*(sigmaext_max-sigmaext)*sqrt(a)/sqrt(x[si_nodivzero_nonegsqrt]-a)
        sigma_increment[si_divzero]=np.inf
        
        # Limit tensile stresses at physical tip (and elsewhere) to yield
        sigma_increment[sigma + sigma_increment > sigma_yield] = sigma_yield-sigma[sigma+sigma_increment > sigma_yield]
        
        # accumulate stresses onto sigma
        sigma += sigma_increment

        # record increment in displacement
        left_of_effective_tip = x < a
        if calculate_displacements:
            tensile_displ[left_of_effective_tip] += tensile_displacement(sigmaext_max-sigmaext,x[left_of_effective_tip],a,crack_model)
            pass
        
        # Record increment in sigmaext
        sigmaext = sigmaext_max
        
        if verbose:
            print("Step: Open to tips @ x=%f mm: %f MPa of tension held" % (a*1e3,sigmaext/1e6))
            if calculate_displacements:
                print("Tensile displacement @ x=%f mm: %f nm" % (x[0]*1e3, tensile_displ[0]*1e9))
                pass
            
            pass
        pass
    
    sigma_with_sigma_closure=sigma-sigma_closure*(x > use_xt_start)*(x <= a)  # sigma_closure only contributes after where we started peeling it open


    
    return (use_xt2, sigma_with_sigma_closure, tensile_displ, dsigmaext_dxt)


def initialize_normalstress_compressive(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,calculate_displacements):
    #Initialize the external applied compressive stress (sigmaext_max negative) starting at zero
    
    sigmaext = 0.0 # External tensile load in this step (Pa)

    

    #Initialize tensile stress field (function of x)
    sigma = np.zeros(x.shape,dtype='d')
    
    #Initialized the Displacement state as zero
    if calculate_displacements:
        tensile_displ = np.zeros(x.shape,dtype='d')
        pass
    else:
        tensile_displ = None
        pass
    
    #Initialize x step counter
    xt_idx = np.where(x<a)[0][-1]

    use_xt2=a
    use_xt1=a #x_bnd[xt_idx]
    
    # Before closing, situation acts just like crack of length a
    # (Note: stress distribution may not be very accurate if
    # closure maximum does not occur @ x=a)
    argmax_sigma_closure = np.argmax(sigma_closure[x < a])  # most compressive
    max_sigma_closure=sigma_closure[x < a][argmax_sigma_closure]
    if max_sigma_closure < 0:
        # All closure stresses negative... crack is entirely open
        # We can hold a stress of max_sigma_closure (negative, i.e. compressive)
        # without any opening at all.
        
        fullyopen_compression = np.max((max_sigma_closure,sigmaext_max))


        sigma_increment = np.zeros(x.shape[0],dtype='d')
        si_nodivzero_nonegsqrt = x-a > 1e-10*a
        si_divzero = (x-a >= 0) & ~si_nodivzero_nonegsqrt
        

        sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext = crack_model.eval_sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext(a)

        # New (sigmaext_max - sigmaext) term is the incremental external  stress field beyond the tips added in addition to the stress contcentration effect
        sigma_increment[si_nodivzero_nonegsqrt] = (fullyopen_compression) + sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*(fullyopen_compression)*sqrt(a)/sqrt(x[si_nodivzero_nonegsqrt]-a)
        sigma_increment[si_divzero]=-np.inf
        
        # Limit compressive stresses at physical tip (and elsewhere) to yield
        sigma_increment[sigma + sigma_increment < -sigma_yield] = -sigma_yield-sigma[sigma+sigma_increment < -sigma_yield]
        
        # accumulate stresses onto sigma
        sigma += sigma_increment

        # record increment in displacement
        left_of_effective_tip = x < a
        if calculate_displacements:
            tensile_displ[left_of_effective_tip] += tensile_displacement(fullyopen_compression,x[left_of_effective_tip],a,crack_model)
            pass
        
        # Record increment in sigmaext
        sigmaext = fullyopen_compression
        
        if verbose:
            print("Step: Crack open to tip: %f MPa of compression held with no closure" % (-fullyopen_compression/1e6))
            if calculate_displacements:
                print("Tensile displacement @ x=%f mm: %f nm" % (x[0]*1e3, tensile_displ[0]*1e9))
                pass
            
            pass
        
        pass
    elif max_sigma_closure >= 0:
        # There are closure stresses (positive compression) somewhere
        
        # Find where sigma_closure goes from negative (tensile)
        # to positive (compressive)


        signchange_idxs = np.where((sigma_closure[x < a][:-1] <= 0.0) & (sigma_closure[x < a][1:] > 0.0))[0]

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
            use_xt1 = use_xt2

            pass
        else:
            # No signchange
            
            if sigma_closure[x<a][-1] > 0.0:
                # have compressive (positive) closure stresses, but no signchange
                # ... crack must be fully closed
                xt_idx=0
                use_xt2=0.0
                use_xt1=0.0
                pass
            else:
                # crack must be fully open
                xt_idx = np.where(x < a)[0][-1] # open all the way to tip
                # if closure stress is tensile everywhere
                
                use_xt2=a
                use_xt1=a #x_bnd[xt_idx]
                pass
            pass

        
        pass
    else:
        assert(0) # Shouldn't be possible
        pass



    return (sigmaext,sigma,tensile_displ,xt_idx,use_xt2,use_xt1)


def crackopening_from_tensile_closure(x,x_bnd,sigma_closure,dx,a,sigma_yield,crack_model):
    """Based on the assumed closure model, whereby we give a 
meaning to "tensile" closure stresses -- based on the compressive 
loading required to close the crack to that point -- we can determine
an opening profile for the unloaded crack. This function calculates
that crack opening"""


    sigmaext_max=-np.inf # as much external compression as we need
    
    (sigmaext,sigma,tensile_displ,xt_idx,use_xt2,use_xt1) = initialize_normalstress_compressive(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,calculate_displacements=True)
    
    done=False

    #if sigmaext==sigmaext_max:
    #    # Used up all of our applied load...  Done!
    #    done=True
    #    pass
    
    while not done: 
        
        (use_xt1,sigmaext, sigma, tensile_displ, dsigmaext_dxt) = solve_incremental_compressivestress(x,x_bnd,sigma,sigma_closure,tensile_displ,use_xt2,xt_idx,dx,sigmaext,sigmaext_max,a,sigma_yield,crack_model,calculate_displacements=True)
        
        
        if use_xt1 <= 0.0:
            # Used up  all of our crack... Done!
            done=True
            pass
        
        #if verbose: 
        #    #Print what is happening in the loop
        #    print("Step: %d @ x=%f mm: %f MPa of compression held" % (xt_idx,x[xt_idx]*1e3,-sigmaext/1e6))
        #    if calculate_displacements:
        #        print("Tensile displacement @ x=%f mm: %f nm" % (x[0]*1e3, tensile_displ[0]*1e9))
        #        pass
        #    pass
        
        xt_idx-=1
        use_xt2=use_xt1
        if not done:
            assert(x_bnd[xt_idx+1]==use_xt2)
            pass
        pass

    return -tensile_displ


def solve_normalstress_compressive(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,verbose=False, diag_plots=False,calculate_displacements=True):
    
    #####MAIN SUPERPOSITION LOOP


    (sigmaext,sigma,tensile_displ,xt_idx,use_xt2,use_xt1) = initialize_normalstress_compressive(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,calculate_displacements)

    #assert((sigma <= 0.0).all())

    done=False

    #if sigmaext==sigmaext_max:
    #    # Used up all of our applied load...  Done!
    #    done=True
    #    pass
    
    dsigmaext_dxt = np.ones(x.shape,dtype='d')*np.nan  # dsigmaext_dxt is a measure of the distributed stress concentration

    
    while not done and sigmaext > sigmaext_max: 
        
        (use_xt1,sigmaext, sigma, tensile_displ, dsigmaext_dxt[xt_idx]) = solve_incremental_compressivestress(x,x_bnd,sigma,sigma_closure,tensile_displ,use_xt2,xt_idx,dx,sigmaext,sigmaext_max,a,sigma_yield,crack_model,calculate_displacements=calculate_displacements)
        
        
        if use_xt1 > x_bnd[xt_idx] or sigmaext==sigmaext_max or use_xt1 <= 0.0:
            # Used up all of our applied load or all of our crack... Done!
            done=True
            pass
        
        if verbose: 
            #Print what is happening in the loop
            print("Step: %d @ x=%f mm: %f MPa of compression held" % (xt_idx,x[xt_idx]*1e3,-sigmaext/1e6))
            if calculate_displacements:
                print("Tensile displacement @ x=%f mm: %f nm" % (x[0]*1e3, tensile_displ[0]*1e9))
                pass
            pass
        
        if not done:
            # loop back
            #assert(x_bnd[xt_idx+1]==use_xt2)

            xt_idx-=1
            use_xt2=use_xt1
            
            pass
        
        pass

    if use_xt1 <= 0.0: 
        # We closed the crack fully without accepting 
        # the full external load.

        # ... Apply the remaining load increment,
        # which will be held uniformly across the sample

        if not np.isinf(sigmaext_max):
            uniform_tension = (sigmaext_max-sigmaext)  # negative number
        
            sigma += uniform_tension
            sigmaext += uniform_tension
            
            #assert((sigma <= 0.0).all())
            
            if verbose:
                print("Step: Closed to center: %f MPa of compression held" % (sigmaext/1e6))
                if calculate_displacements:
                    print("Tensile displacement @ x=%f mm: %f nm" % (x[0]*1e3, tensile_displ[0]*1e9))
                    pass
                
                pass
            pass
        pass

    #assert((sigma <= 0.0).all())

    #sigma_with_sigma_closure=sigma-sigma_closure*(x > use_xt1)*(x <= a)  # sigma_closure only contributes after the effective tip

    sigma_with_sigma_closure=sigma.copy()
    # sigma_closure should be superimposed with the
    # external load effect sigma over the entire region
    # where the crack is closed.
    # this is presumbly true everywhere beyond the current segment up to the crack length a WE SHOULD PROBABLY DEAL BETTER WITH THE LAST SEGMENT AT THE TIP!
    sigma_with_sigma_closure[(xt_idx+1):][x[(xt_idx+1):] <= a] -= sigma_closure[xt_idx+1:][x[(xt_idx+1):] <= a]

    # The current segment (indexd by xt_idx) may be partial,
    # so weight it according to the portion that is actually closed
    
    sigma_with_sigma_closure[xt_idx] -= sigma_closure[xt_idx]*(use_xt2-use_xt1)/dx;

    # correct any small residual tension
    if sigma_with_sigma_closure[xt_idx] > 0.0:
        sigma_with_sigma_closure[xt_idx]=0.0
        pass
    
        
    
    #assert((sigma_with_sigma_closure <= 0.0).all())

    return (use_xt1, sigma_with_sigma_closure, tensile_displ, dsigmaext_dxt)

def solve_normalstress(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,verbose=False, diag_plots=False,calculate_displacements=True):
    """NOTE: sigma_closure is positive compressive; pretty much everything else is positive tensile.

    NOTE: Modified to return a sigma that is a physical contact stress, 
    INCLUDING the effect of sigma_closure
"""
    if sigmaext_max >= 0.0:
        return solve_normalstress_tensile(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,verbose=verbose,diag_plots=diag_plots,calculate_displacements=calculate_displacements)
    else:
        return solve_normalstress_compressive(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,verbose=verbose,diag_plots=diag_plots,calculate_displacements=calculate_displacements)
    pass

def inverse_closure(reff,seff,x,x_bnd,dx,xt,sigma_yield,crack_model,verbose=False):
    """ Given effective crack lengths reff at externally applied loads seff,
    calculate a closure stress field that produces such a field.
    reff,seff presumed to be ordered from most compressive to 
    most tensile. 

    
    seff is positive for tensile opening loads

    returns sigma_closure that is positive for compressive
    closure stresses. 

"""

    sigma_closure = np.zeros(x.shape,dtype='d')
    
    last_closure = seff[0] # If everything is closed, then closure stresses and external loads match. 
    last_reff = reff[0]

    firstiteration=True

    # NOTE: Always iterate starting at step 1, then back to 0 
    # then back to 1, then increasing. 
    # The reason for this is to get a reasonable extrapolation 
    # to the left of the given data. 
    #
    # Basically, the first iteration at step 1 uses 
    # seff[0] as an approximation for the closure state
    # from the (non-executed) step 0. 
    # Then execute step 0 based on that initial result to get 
    # a slope we can extrapolate (don't allow the slope to be negative...
    # if it would be, just make it zero -- straight horizontal)
    # all the way to the crack center. 
    # Once this is done, rerun step 1 based on the step zero
    # result and continue from there. 

    for lcnt in [1,0]+list(range(1,reff.shape[0])):
        # In each step, we solve for a new linear segment
        # of the sigma_closure distribution.
        # we assume last_closure is the end closure stress
        # from the previous step, corresponding to a
        # position of reff[lcnt-1], with an
        # external load of seff[lcnt-1] opening the
        # crack to this point
        #print("lcnt=%d" % (lcnt))
        #if lcnt==7:
        #    import pdb
        #    pdb.set_trace()
        #    pass

        # So at this step, if the new closure stress is
        # new_closure, then
        # in between we have a line:
        # sigma_closure(x) = last_closure + (new_closure-last_closure)*(x-reff[lcnt-1])
        # So we need to iteratively solve for a new_closure that satisfies
        
        # (reff[lcnt], sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(x,x_bnd,sigma_closure,dx,seff[lcnt],a,sigma_yield,crack_model)
        # For the given reff[lcnt], seff[lcnt]

        #if reff[lcnt] < last_reff+dx/10.0:
        #    # if the opening step is too small, skip this iteration
        #    # to avoid dividing by zero
        #    continue
            
        if lcnt==1 and firstiteration:
            # first iteration: extrapolate back to crack center
            # if necessary  ***!!!! May want to fix this
            #new_zone = (x_bnd[:-1] <= reff[lcnt])

            # providing closure stresses to right of closure point is useful
            # because intermediate calculations may examine more open cases,
            # and having nice behavior here improves convergence
            new_zone=np.ones(x.shape,dtype=np.bool)
            #new_zone = (x_bnd[1:] >= reff[lcnt])
            #open_zone = (x_bnd[1:] < reff[lcnt])
            pass
        elif lcnt==0: 
            # Iteration at position zero after initialization at lcnt==1
            #new_zone = (x_bnd[1:] < reff[lcnt])
            new_zone=np.ones(x.shape,dtype=np.bool)
            if np.count_nonzero(new_zone)==0:
                # nothing to do 
                continue
            zone_following = (x_bnd[1:] >= reff[lcnt+1])
            zone_following_start = np.where(zone_following)[0][0]
            pass
        else:
            #new_zone = (x_bnd[1:] >= reff[lcnt-1]) & (x_bnd[:-1] <= reff[lcnt])
            #new_zone = (x_bnd[1:] >= reff[lcnt-1])
            new_zone = (x_bnd[1:] >= last_reff)
            pass

        if lcnt != 0:
            # (these values are not used for lcnt==0)
            zone_start = np.where(x_bnd[1:] >= last_reff)[0][0]
            zone_end = np.where(x_bnd[:-1] <= reff[lcnt])[0][-1]
            zone_prev = max(0,zone_start-1)
            pass

        #print("np.where(new_zone) = %s" % (str(np.where(new_zone))))
        
        def goal(new_closure):
            new_closure_field = copy.copy(sigma_closure)

            
            # WARNING: All changes that affect new_closure_field in goal()
            # MUST ALSO BE APPLIED to generating sigma_closure after the optimization !!!
            if lcnt==1 and firstiteration:
                new_closure_field[new_zone] = seff[0] + (new_closure-seff[0]) * (x[new_zone]-x[zone_prev])/(x[zone_end]-x[zone_prev])
                pass
            elif lcnt==0:
                if new_closure >= sigma_closure[zone_following_start]:
                    # use straight horizontal
                    new_closure_field[new_zone] = new_closure
                    pass
                else: 
                    # Connect with slope
                    new_closure_field[new_zone] = sigma_closure[zone_following_start] + (new_closure-sigma_closure[zone_following_start]) * (x[new_zone]-x[zone_following_start])/(0.0-x[zone_following_start])
                    pass
                pass
            else:
                # draw straight line between zone_prev and new_zone. Use np.max() to prevent new_closure_field from being negative. 
                new_closure_field[new_zone] = np.max((np.zeros(np.count_nonzero(new_zone),dtype='d'),sigma_closure[zone_prev] + (new_closure-sigma_closure[zone_prev]) * (x[new_zone]-x[zone_prev])/(x[zone_end]-x[zone_prev])),axis=0)
                pass

            (gotreff, sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(x,x_bnd,new_closure_field,dx,seff[lcnt],xt,sigma_yield,crack_model,calculate_displacements=False,verbose=verbose)

            return gotreff-reff[lcnt]

        if lcnt==0:
            # For position 0, resulting solution 
            # may be less than seff[0] or even negative
            max_sol_attempts=300
            first_initialization_factor=0.03
            initialization_scale_factor=-1.05
            seed=max(abs(seff[-1]-seff[0]),seff[0],np.mean(seff))
            pass
        else:
            max_sol_attempts=20
            first_initialization_factor=1.0
            initialization_scale_factor=1.1
            seed = seff[lcnt]
            pass

        solvecnt=0


        if lcnt > 1 and not firstiteration and goal(-np.inf) < 0.0:
            # Applying 
            # no increment isn't enough to get
            # it to open this far under this load...
            # Use the most extreme value possible
            new_closure = -np.inf
            pass
        else:
            inifactor=first_initialization_factor
            while solvecnt < max_sol_attempts:
                (new_closure,infodict,ier,mesg) = scipy.optimize.fsolve(goal,seed*inifactor,full_output=True)
                #print("ier=%d; new_closure=%g" % (ier,new_closure[0]))
                if ier==1:
                    break
                inifactor*=initialization_scale_factor
                solvecnt+=1
                pass
            if ier != 1:

                if lcnt > 0 and last_reff >= xt-dx/4.0:
                    # if we don't converge and previous step was within
                    # a quarter-step of the end, claim we are good
                    # and quit.
                    break

            
                sys.modules["__main__"].__dict__.update(globals())
                sys.modules["__main__"].__dict__.update(locals())
                raise ValueError("Error in inverse_closure fsolve: %s" % str(mesg))
            pass
        

        
        #closure_gradient = (sigma_closure[zone_end]-sigma_closure[zone_start])/(x[zone_end]-x[zone_start])
        

        # WARNING: All changes that affect sigma_closure here
        # MUST ALSO BE APPLIED to generating new_closure_field in goal()!!!
        if lcnt==1 and firstiteration:
            sigma_closure[new_zone] = seff[0] + (new_closure-seff[0]) * (x[new_zone]-x[zone_prev])/(x[zone_end]-x[zone_prev])
            pass
        elif lcnt==0:
            if new_closure >= sigma_closure[zone_following_start]:
                # use straight horizontal
                sigma_closure[new_zone] = new_closure
                pass
            else: 
                # Connect with slope
                sigma_closure[new_zone] = sigma_closure[zone_following_start] + (new_closure-sigma_closure[zone_following_start]) * (x[new_zone]-x[zone_following_start])/(0.0-x[zone_following_start])
                pass
            pass
        else:
            # draw straight line between zone_prev and new_zone. Use np.max() to prevent sigma_closure from being negative. 
            sigma_closure[new_zone] = np.max((np.zeros(np.count_nonzero(new_zone),dtype='d'),sigma_closure[zone_prev] + (new_closure-sigma_closure[zone_prev]) * (x[new_zone]-x[zone_prev])/(x[zone_end]-x[zone_prev])),axis=0)
            pass

        #print("sigma_closures %s" % (str(sigma_closure)))
        #print("new_zone closures: %s" % (str(sigma_closure[new_zone][:2])))
        
        firstiteration=False
        last_closure = new_closure
        last_reff = reff[lcnt]
        pass

    #if reff[lcnt] < xt:
    #    # don't have data out to the tips
    #    # extrapolate last closure gradient to the tips
    #    
    #    new_zone = (x_bnd > reff[lcnt])
    #    zone_start = np.where(new_zone)[0]
    #    zone_end = np.where
    #    new_closure=last_closure + closure_gradient * (xt-x[zone_start])
    #    
    #    sigma_closure[new_zone] = last_closure + (new_closure-last_closure) * (x[new_zone]-reff[lcnt])/(xt-reff[lcnt])
    #    
    #    pass

    sigma_closure[x > xt] = 0.0
    
    ## Extrapolate first to points back to the origin (open zone
    #first_closure_index = np.where(~open_zone)[0][0]
    
    #initial_slope = (sigma_closure[first_closure_index+1]-sigma_closure[first_closure_index])/dx
    #sigma_closure[open_zone] = sigma_closure[first_closure_index] + (x[open_zone]-x[first_closure_index])*initial_slope
    
    return sigma_closure



def inverse_closure_backwards_broken(reff,seff,x,x_bnd,dx,xt,sigma_yield,crack_model,verbose=False):
    """ Given effective crack lengths reff at externally applied loads seff,
    calculate a closure stress field that produces such a field.
    reff,seff presumed to be ordered from most compressive to 
    most tensile. 

    
    seff is positive for tensile opening loads

    returns sigma_closure that is positive for compressive
    closure stresses. 

    if seff[0] is > 0, and reff[0] > 0 then sigma_closure corresponding
    to reff[0] is assumed to match seff[0]. 


    NOTE: This version implements it backwards (tip to center) 
    and is broken
"""

    assert((np.diff(seff) > 0).all())


    # for a crack with no closure stresses, even an epsilon
    # tensile opening causes the crack to open to the tips.

    # We work back from the tips.
    # at the last, radius, stress combination
    # there is no load on the crack surface except past
    # that last radius.
    #
    # sigma_closure is positive compression. 
    
    sigma_closure = np.zeros(x.shape,dtype='d')

    last_r = xt
    last_closure = None
    
    for lcnt in range(reff.shape[0]-1,-1,-1):
        # In each step, we solve for a new linear segment
        # of the sigma_closure distribution.
        # we assume last_closure is the end closure stress
        # from the previous step, corresponding to a
        # position of reff[lcnt+1], with an
        # external load of seff[lcnt+1] opening the
        # crack to this point

        # So at this step, if the new closure stress is
        # new_closure, then
        # in between we have a line:
        # sigma_closure(x) = last_closure + (new_closure-last_closure)*(x-reff[lcnt+1])
        # So we need to iteratively solve for a new_closure that satisfies
        
        # (reff[lcnt], sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(x,x_bnd,sigma_closure,dx,seff[lcnt],a,sigma_yield,crack_model)
        # For the given reff[lcnt], seff[lcnt]
        

        if lcnt==reff.shape[0]-1:
            # first iteration: Don't know stress to open to tip
            # (unless this is tip!)
            
            #new_zone = (x >= reff[lcnt]) & (x <= xt)
            new_zone = (x_bnd[1:] >= reff[lcnt]) & (x_bnd[:-1] <= xt)
            pass
        else:
            new_zone = (x_bnd[1:] >= reff[lcnt]) & (x_bnd[:-1] <= reff[lcnt+1])
            #new_zone = (x >= reff[lcnt]) & (x < reff[lcnt+1])
            pass
        


        def goal(new_closure):
            new_closure_field = copy.copy(sigma_closure)
            
            if last_closure is not None:
                new_closure_field[new_zone] = last_closure + (new_closure-last_closure) * (reff[lcnt+1]-x[new_zone])/(reff[lcnt+1]-reff[lcnt])   # slope
  # slope
                pass

            else:
                new_closure_field[new_zone] = new_closure # Horizontal line
                pass
            
            (gotreff, sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(x,x_bnd,new_closure_field,dx,seff[lcnt],xt,sigma_yield,crack_model,calculate_displacements=False,verbose=verbose)

            return gotreff-reff[lcnt]

        
        if reff[lcnt] < last_r and np.count_nonzero(new_zone) > 0:
            (new_closure,infodict,ier,mesg) = scipy.optimize.fsolve(goal,seff[lcnt]*3.4,full_output=True)
        
            if ier != 1:
                sys.modules["__main__"].__dict__.update(globals())
                sys.modules["__main__"].__dict__.update(locals())
                raise ValueError("Error in inverse_closure fsolve: %s" % str(mesg))


            
            if last_closure is not None:
                sigma_closure[new_zone] = last_closure + (new_closure-last_closure) * (reff[lcnt+1]-x[new_zone])/(reff[lcnt+1]-reff[lcnt])   # slope
                pass

            else:
                sigma_closure[new_zone] = new_closure # Horizontal line
                pass
                
            last_closure = new_closure
            last_r = reff[lcnt]
            pass
        
        pass
    
    #if reff[lcnt] < xt:
    #    # don't have data out to the tips
    #    # extrapolate last closure gradient to the tips
    #    
    #    new_zone = (x >= reff[lcnt])
    #    new_closure=last_closure + closure_gradient * (xt-reff[lcnt])
    #    
    #    sigma_closure[new_zone] = last_closure + (new_closure-last_closure) * (x[new_zone]-reff[lcnt])/(xt-reff[lcnt])
    #    
    #    pass

    sigma_closure[x > xt] = 0.0
    
    return sigma_closure




class Glinka_ModeI_ThroughCrack(ModeI_Beta_WeightFunction):
    """Create and return ModeI_crack_model corresponding
    to the Through crack weight function from the Glinka paper"""

    def __init__(self,Eeff,x,width,epsx):

        
        def weightfun_through_times_sqrt_aminx(x, a, w):
            # Weight function (stress intensity factor resulting from point load on crack surface) for a through crack or tunnel crack 
            # reference: Glinka, G. "Development of weight functions and computer integration procedures for calculating stress intensity factors around cracks subjected to complex stress fields." Stress and Fatigue-Fracture Design, Petersburg Ontario, Canada, Progress Report 1.1 (1996): 1.
            # x=position, 
            # a=half-crack length (not always physical total length, but may be an effective length for partially closed crack) 
            # w=half-width of geometry  (same axis as x and a)
            M1 = 0.06987 + 0.40117*(a/w) - 5.5407*(a/w)**2.0 + 50.0886*(a/w)**3.0 - 200.699*(a/w)**4.0 + 395.552*(a/w)**5.0 - 377.939*(a/w)**6.0 + 140.218*(a/w)**7.0
            M2 = 0.09049 - 2.14886*(a/w) + 22.5325*(a/w)**2.0 - 89.6553*(a/w)**3.0 + 210.599*(a/w)**4.0 - 239.445*(a/w)**5.0 + 111.128*(a/w)**6.0
            M3 = 0.427216 + 2.56001*(a/w) - 29.6349*(a/w)**2.0 + 138.40*(a/w)**3.0 - 347.255*(a/w)**4.0 + 457.128*(a/w)**5.0 - 295.882*(a/w)**6.0 + 68.1575*(a/w)**7.0
            
            return (2.0/np.sqrt(2*np.pi))*(1.0+M1*scipy.sqrt(1.0-x/a)+M2*(1.0-x/a)+M3*(1.0-x/a)**1.5)
        
        super(Glinka_ModeI_throughcrack, self).__init__(
            weightfun_times_sqrt_aminx=lambda obj,x,a: weightfun_through_times_sqrt_aminx(x,a,width),
            epsx=epsx,
            Eeff=Eeff,
            surrogate_a=x,
            use_surrogate=True)
        pass
    pass



def ModeI_throughcrack_weightfun(Eeff,x,epsx):

    ## Basic weight function from Fett & Munz Stress Intensity Factors and Weight Functions eq. 1.2.5
    # Equivalent to
    # Basic weight function from Anderson, Fracture Mechanics, Example 2.6 (page 57)
    # Weight function given as h(x) = \pm (1/sqrt(pi*a)) * sqrt(x/(2a-x))
    # ... But the coordinate frame is weird. From Fig 2.27 on page 58,
    # x=0 is at the other end of the crack (!) versus our origin is
    # the center of the crack. Let x' = x-a -> x=x'+a
    # Now h(x') = \pm (1/sqrt(pi*a)) * sqrt((x'+a)/(2a-x'-a))
    #     h(x') = \pm (1/sqrt(pi*a)) * sqrt((x'+a)/(a-x'))
    #
    #def weightfun_basic_times_sqrt_aminx(x,a):
    #    return np.sqrt(1.0/(np.pi*a))*np.sqrt((a+x))
    
    # Corrected for origin being at center of crack, not tip.
    # See corrected_tunnelcrack_weightfun.pdf
    def weightfun_basic_times_sqrt_aminx(x,a):
        return (1.0/np.sqrt(np.pi))*np.sqrt(a)/np.sqrt(a+x)
    
    return ModeI_Beta_WeightFunction(weightfun_times_sqrt_aminx=lambda obj,x,a: weightfun_basic_times_sqrt_aminx(x,a),
                                     epsx=epsx,
                                     Eeff=Eeff,
                                     surrogate_a=x,
                                     use_surrogate=True)

class ModeI_throughcrack_CODformula(ModeI_Beta_COD_Formula):
    def r0_over_a(self,xt):
        """
        The baseline approximation of the stress field beyond the crack
        tip is K/sqrt(2*pi*r), but this is only valid within perhaps a/10
        of the tip. 

        Initially we approximated the stress field as 
        K/sqrt(2*pi*r) + sigma_infty so that the stress would approach
        the correct value as r -> infty. Unfortunately this doesn't 
        satisfy force balance (in fact if you integrate it, it fails
        to converge!). 

        So our fix is to approximate the stress field as 
        (K/sqrt(2*pi*r))*(r0^2/(r+r0)^2) + sigma_infty, where 
        r0 is selected to satisfy force balance between the load 
        not held over the cracked region and the stress concentration
        beyond the tip. 


        r0 is the characteristic radius for the 1/r^2 decay 
        of the 1/sqrt(r) term

        Assuming K has the form sigma*sqrt(pi*a*beta) for a through
        crack in a thin plate, then per 
            total_load_matching_crossterm_r2_work.pdf

        r0 = 8a/(pi^2*beta) 
        """
        return 8.0/((np.pi**2.0)*self.beta(self))
    
        
    def __init__(self,Eeff):
        def u(Eeff,sigma_applied,x,xt):
            # Non weightfunction method:

            # Old method: Based on Suresh eq. 9.45.
            # The problem with the old method is it is based
            # on a near-tip approximation
            #Kappa = (3.0-nu)/(1.0+nu)
            #
            #KI = sigma_applied*np.sqrt(np.pi*(xt))
            #theta = np.pi
            #u = (KI/(2.0*E))*(np.sqrt((xt-x)/(2.0*np.pi)))*((1.0+nu)* 
            #                                                (((2.0*Kappa+1.0)*(np.sin(theta/2.0)))-np.sin(3.0*theta/2.0)))
            
            # New Method: Based on Anderson, eq. A2.43
            # uy = 2(sigma/Eeff)*sqrt(a^2-x^2)
            # uy = 2(sigma/Eeff)*sqrt((a+x)(a-x))
            
            #Eeff = E
            u = (2*sigma_applied/Eeff)*np.sqrt((xt+x)*(xt-x))
            return u
    
        super(ModeI_throughcrack_CODformula, self).__init__(Eeff=Eeff,
                                                            beta=lambda obj: 1.0,
                                                            u = lambda obj,sigma_applied,x,xt: u(obj.Eeff,sigma_applied,x,xt))
        pass
    pass



class Tada_ModeI_CircularCrack_along_midline(ModeI_Beta_COD_Formula):
    def r0_over_a(self,xt):
        """Based on calculation given in total_load_matching_crossterm_r2_work.pdf
"""
        return (2.0**(1.0/3.0))/((np.pi**(2.0/3.0))*(self.beta(self)**(1.0/3.0)))
    
    def __init__(self,E,nu):
        def u(E,nu,sigma_applied,x,xt):
            # For a circular crack in an infinite space,
            # loaded in mode I.
            # We will be evaluating along a line through the crack center
            # Based on Tada, H., Paris, P., & Irwin, G. (2000). The stress analysis of cracks handbook / Hiroshi Tada, Paul C. Paris, George R. Irwin. (3rd ed.). New York: ASME Press.
        
            u = (4.0*(1-nu**2.0)/(np.pi*E)) * sigma_applied * np.sqrt(xt**2.0 - x**2.0)
            return u
    
        super(Tada_ModeI_CircularCrack_along_midline, self).__init__(E=E,
                                                                     nu=nu,
                                                                     beta=lambda obj: 4.0/(np.pi**2.0),
                                                                     u = lambda obj,sigma_applied,x,xt: u(obj.E,obj.nu,sigma_applied,x,xt))
        pass
    pass




def perform_inverse_closure(inputfilename,E,nu,sigma_yield,CrackCenterX,dx,specimen_id,hascrackside1=True,hascrackside2=True):
    from matplotlib import pyplot as pl
    import pandas as pd

    #tau_yield = sigma_yield/2.0 # limits stress concentration around singularity


    # read closure profile

    cpdata = pd.read_csv(inputfilename,index_col="Opening load (Pa)")
    
    loads = np.array(cpdata.index)

    tippos_side1 = None
    tippos_side2 = None

    if "xt (side 1, m)" in cpdata:
        tippos_side1 = np.array(cpdata["xt (side 1, m)"])
        assert(hascrackside1)
        pass
    
    if "xt (side 2, m)" in cpdata:
        tippos_side2 = np.array(cpdata["xt (side 2, m)"])
        assert(hascrackside2)
        pass
    #cpdata = np.loadtxt(inputfilename,skiprows=1,delimiter=',')
    #assert(cpdata.shape[1]==3)
    
    #loads = cpdata[:,0]
    #tippos_side1 = cpdata[:,1]
    #tippos_side2 = cpdata[:,2]
    
    sigmaext_max=np.max(loads)
    
    a_side1=0.0
    a_side2=0.0

    # side 1 (left side)
    if tippos_side1 is not None:
        observed_reff_side1 = CrackCenterX - tippos_side1
        observed_seff_side1 = loads
        
        a_side1=np.max(observed_reff_side1)
        pass

    # side 2 (right side)
    if tippos_side2 is not None:
        observed_reff_side2 = tippos_side2 - CrackCenterX
        observed_seff_side2 = loads
        
        a_side2=np.max(observed_reff_side2)
        pass


    # here, x really measures radius past crack center
    xmax_approx = 2.0*max(a_side1,a_side2)  # x array goes past tip position (twice half-length)
    #dx = 25e-6
    xsteps = int(xmax_approx//dx)
    xmax = dx*xsteps

    x_bnd=np.arange(xsteps,dtype='d')*dx
    x = (x_bnd[1:]+x_bnd[:-1])/2.0

    weightfun_epsx = dx/8.0
    crack_model = Tada_ModeI_CircularCrack_along_midline(E,nu)


    sigma_closure_side1 = None
    sigma_closure_side2 = None

    
    if tippos_side1 is not None:
        sigma_closure_side1 = inverse_closure(observed_reff_side1[observed_reff_side1 >= 0.0],
                                              observed_seff_side1[observed_reff_side1 >= 0.0],
                                              x,x_bnd,dx,a_side1,sigma_yield,
                                              crack_model)
        pass



    
    if tippos_side2 is not None:
        sigma_closure_side2 = inverse_closure(observed_reff_side2[observed_reff_side2 >= 0.0],
                                              observed_seff_side2[observed_reff_side2 >= 0.0],
                                              x,x_bnd,dx,a_side2,sigma_yield,
                                              crack_model)
        pass

    side1fig = None
    side2fig = None

    
    # Forward cross-check of closure
    if tippos_side1 is not None:
        side1fig=pl.figure()
        pl.plot(x[x < a_side1]*1e6,sigma_closure_side1[x < a_side1]/1e6,'-',
                observed_reff_side1*1e6,observed_seff_side1/1e6,'x')
        for observcnt in range(len(observed_reff_side1)):        
            (effective_length, sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(x,x_bnd,sigma_closure_side1,dx,observed_seff_side1[observcnt],a_side1,sigma_yield,crack_model)
            pl.plot(effective_length*1e6,observed_seff_side1[observcnt]/1e6,'.')
            #pl.plot(x*1e3,tensile_displ*1e15,'-')
            pass
        pl.grid()
        pl.legend(('Closure stress field','Observed crack tip posn','Recon. crack tip posn'),loc="best")
        pl.xlabel('Radius from crack center (um)')
        pl.ylabel('Stress (MPa)')
        if specimen_id is not None:
            pl.title('%s: Side 1 (left)' % (specimen_id))
            pass
        else:
            pl.title('Side 1 (left)')        
            pass
        
        side2fig=pl.figure()
        pl.plot(x[x < a_side2]*1e6,sigma_closure_side2[x < a_side2]/1e6,'-',
                observed_reff_side2*1e6,observed_seff_side2/1e6,'x')
        for observcnt in range(len(observed_reff_side2)):        
            (effective_length, sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(x,x_bnd,sigma_closure_side2,dx,observed_seff_side2[observcnt],a_side2,sigma_yield,crack_model)
            pl.plot(effective_length*1e6,observed_seff_side2[observcnt]/1e6,'.')
            #pl.plot(x*1e3,tensile_displ*1e15,'-')
            pass
        pl.grid()
        pl.legend(('Closure stress field','Observed crack tip posn','Recon. crack tip posn'),loc="best")
        if specimen_id is not None:
            pl.title('%s: Side 2 (right)' % (specimen_id))
            pass
        else:
            pl.title('Side 2 (right)')        
            pass
        pl.xlabel('Radius from crack center (um)')
        pl.ylabel('Stress (MPa)')
        pass
    
    return (x,x_bnd,a_side1,a_side2,sigma_closure_side1,sigma_closure_side2,side1fig,side2fig)

def save_closurestress(filename,x,sigma_closure,a,crackopening=None):
    import pandas as pd

    nrows = np.count_nonzero(x <= a)+1
    
    out_frame = pd.DataFrame(index=pd.Float64Index(data=x[:nrows],dtype='d',name="Crack radius (m) compared to crack (half) length a=%.8g m" % (a)))
    
    out_frame.insert(len(out_frame.columns),"Closure stress (Pa)", sigma_closure[:nrows])
    if crackopening is not None:
        out_frame.insert(len(out_frame.columns),"Crack opening (m)", crackopening[:nrows])
        pass
    out_frame.to_csv(filename)

    #
    #with open(filename,"wb") as csvfile:
    #    cpwriter = csv.writer(csvfile)
    #    columntitles = ["Crack radius (m)","Closure stress (Pa)"]
    #    if crackopening is not None:
    #        columntitles.append("Crack opening (m)")
    #        pass
    #    cpwriter.writerow(columntitles)
    #
    #    if crackopening is None:
    #        for poscnt in range(np.count_nonzero(x <= a)):
    #            cpwriter.writerow([ x[poscnt], sigma_closure[poscnt]])
    #            pass
    #        pass
    #    else:
    #        for poscnt in range(np.count_nonzero(x <= a)):
    #            cpwriter.writerow([ x[poscnt], sigma_closure[poscnt], crackopening[poscnt]])
    #            pass
    #        pass
    #    pass
    pass



def load_closurestress(filename):
    import pandas as pd
    
    closurestress_dataframe = pd.read_csv(filename,index_col=0)
    
    # determine crack length a from index title if possible
    if closurestress_dataframe.index.name=="Crack radius (m)":
        a = None # Not encoded
    else: 
        matchobj = re.match(r"""Crack radius \(m\) compared to crack \(half\) length a=([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?) m""",closurestress_dataframe.index.name)
        if matchobj is None: 
            raise ValueError("Failed to parse crack length from index title \"%s\"" % (closurestress_dataframe.index.name))
        
        a = float(matchobj.group(1))
        pass
    
    x = np.array(closurestress_dataframe.index)
    xstep = x[1]-x[0]

    if a is None: 
        # Old version of save_closurestress that
        # didn't include crack length in header didn't go 
        # at all beyond crack length... but we need 
        # at least one sample beyond
        x = np.concatenate((x,(x[-1]+xstep,)))
        pass
    x_bnd = np.concatenate(((x[0]-xstep/2.0,),x+xstep/2.0))
    if x_bnd[0] < 0.0:
        x_bnd[0]=0.0
        pass
        
    sigma_closure = np.array(closurestress_dataframe["Closure stress (Pa)"])
    if a is None: 
        # expand out sigma_closure by one sample
        sigma_closure = np.concatenate((sigma_closure,(0.0,)))
        pass

    if "Crack opening (m)" in closurestress_dataframe.keys():
        crack_opening = np.array(closurestress_dataframe["Crack opening (m)"])
        if a is None: 
            # expand out crack_opening by one sample
            crack_opening = np.concatenate((crack_opening,(0.0,)))
            pass
        pass
    else:
        crack_opening = None
        pass



    return (x,x_bnd,xstep,a,sigma_closure,crack_opening) 
