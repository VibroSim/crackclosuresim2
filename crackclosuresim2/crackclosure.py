
import sys
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
    integral_sigmaext1^sigmaext2 of 1.0 + sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*sqrt(xt)/sqrt(x-xt) dsigmaext
    Here, xt is still dependent on sigmaext... sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext also has 
    some dependence on a (i.e. xt), but as xt isn't moving much this is presumed to be small 

    this will give normal stress
    as a function of position (x). 

    We assume xt is linearly dependent on normal stress:
    xt = xtp + (1/F)*(sigmaext-sigmaext1)
    where xtp is the final xt from the previous step. 
    
    sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext is treated as constant, 
    between xt1 and xt2
    
    So our incremental tension is
    integral_sigmaext1^sigmaext2 (1.0 + sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext sqrt(xt)/sqrt(x-xt)) dsigmaext
    where we ignore any contributions corresponding to (x-xt) <= 0

    (the new 1.0 term represents that beyond the effective tip the external 
    load directly increments the stress state, in addition to the stress 
    concentration caused by the presence of the open region)    

    pull out constant term

    (sigmaext2-sigmaext1) + integral_sigmaext1^sigmaext2 sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext sqrt(xt)/sqrt(x-xt) dsigmaext

    Perform change of integration variable sigmaext -> xt: 
       Derivative of xt:
       dxt = (1/F)*dsigmaext
       dsigmaext = F*dxt


    So the  incremental normal stress we are solving for is
    (sigmaext2-sigmaext1) + integral_xt1^xt2 sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext sqrt(xt)*F/sqrt(x-xt)  dxt
    where we ignore any contributions corresponding to (x-xt) <= 0

    and sigmaext2 = sigmaext1 + (xt2-xt1)*F 
 
    F is a constant and sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext is treated as constant so have 
    F*(xt2-xt1)  +  F * sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext * integral_xt1^xt2 sqrt(xt)/(sqrt(x-xt))  dxt


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
    F*(upper_bound-xt1) + (sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*F)*(indef_integral_of_simple_squareroot_quotients(x,upper_bound) - indef_integral_of_simple_squareroot_quotients(x,xt1))

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
    res[nonzero] = F*(upper_bound[nonzero]-xt1)  +  (sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*F) * (indef_integral_of_simple_squareroot_quotients(x[nonzero],upper_bound[nonzero]) - indef_integral_of_simple_squareroot_quotients(x[nonzero],xt1))
    
    

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
    res[nonzero] = -F*(upper_bound[nonzero]-use_xt1)  -  (sigmaI_theta0_times_rootr_over_sqrt_a_over_sigmaext*F) * (indef_integral_of_simple_squareroot_quotients(x[nonzero],upper_bound[nonzero]) - indef_integral_of_simple_squareroot_quotients(x[nonzero],use_xt1))
    
    

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
        Fbnd = 2.0*(sigmaext_max - sigmaext)/(next_bound-x_bnd[xt_idx])
        # Increase Fbnd until we get a positive result from obj_fcn
        while obj_fcn(Fbnd) <= 0.0:
            Fbnd*=2.0;
            pass

        # Condition below should no longer be needed
        if obj_fcn(Fbnd) < 0.0:
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

    
    return (use_xt2,sigmaext2, sigma+sigma_increment, ret_displ) 
   



def solve_incremental_compressivestress(x,x_bnd,sigma,sigma_closure,tensile_displ,use_xt2,xt_idx,dx,sigmaext,sigmaext_max,a,crack_model,calculate_displacements=True):
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
        Fbnd = 2.0*(sigmaext_max - sigmaext)/(next_bound-use_xt2)
        # Increase Fbnd until we get a negative result from obj_fcn
        while obj_fcn(Fbnd) >= 0.0:
            Fbnd*=2.0;
            pass

        # Condition below should no longer be needed
        if obj_fcn(Fbnd) > 0.0:
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

    
    return (use_xt1,sigmaext2, sigma+sigma_increment, ret_displ) 
   



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
        # We can hold a stress of min_sigma_closure
        # without any opening at all.

        uniform_tension = np.min((min_sigma_closure,sigmaext_max))
        
        sigma += uniform_tension
        sigmaext += uniform_tension

        # assume anything to the left of the
        # sigma_closure minimum is open
        # once we get to this point
        xt_idx=argmin_sigma_closure
        use_xt2=0
        pass

    
    done=False

    if sigmaext==sigmaext_max:
        # Used up all of our applied load...  Done!
        done=True
        pass
    
    while not done: 
        
        (use_xt2,sigmaext, sigma, tensile_displ) = solve_incremental_tensilestress(x,x_bnd,sigma,sigma_closure,tensile_displ,xt_idx,dx,sigmaext,sigmaext_max,a,crack_model,calculate_displacements=calculate_displacements)
        
        
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
        
        xt_idx+=1
        
        
        pass

    if sigmaext < sigmaext_max:
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
    
    return (use_xt2, sigma, tensile_displ)

def solve_normalstress_compressive(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,verbose=False, diag_plots=False,calculate_displacements=True):
    #Initialize the external applied compressive stress (sigmaext_max negative) starting at zero
    
    sigmaext = 0.0 # External tensile load in this step (Pa)

    

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
    xt_idx = np.where(x<a)[0][-1]

    use_xt2=a
    use_xt1=x_bnd[xt_idx]
    
    # Before closing, situation acts just like crack of length a
    # (Note: stress distribution may not be very accurate if
    # closure maximum does not occur @ x=a)
    argmax_sigma_closure = np.argmax(sigma_closure[x < a])
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
        
        # Limit tensile stresses at physical tip (and elsewhere) to yield
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

    
    done=False

    if sigmaext==sigmaext_max:
        # Used up all of our applied load...  Done!
        done=True
        pass
    
    while not done: 
        
        (use_xt1,sigmaext, sigma, tensile_displ) = solve_incremental_compressivestress(x,x_bnd,sigma,sigma_closure,tensile_displ,use_xt2,xt_idx,dx,sigmaext,sigmaext_max,a,crack_model,calculate_displacements=calculate_displacements)
        
        
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
        
        xt_idx-=1
        use_xt2=use_xt1
        if not done:
            assert(x_bnd[xt_idx+1]==use_xt2)
            pass
        pass

    if use_xt1 <= 0.0: 
        # We closed the crack fully without accepting 
        # the full external load.

        # ... Apply the remaining load increment,
        # which will be held uniformly across the sample
        
        uniform_tension = (sigmaext_max-sigmaext)  # negative number
        
        sigma += uniform_tension
        sigmaext += uniform_tension

        
        if verbose:
            print("Step: Closed to center: %f MPa of compression held" % (sigmaext/1e6))
            if calculate_displacements:
                print("Tensile displacement @ x=%f mm: %f nm" % (x[0]*1e3, tensile_displ[0]*1e9))
                pass
            
            pass
        pass
    
    return (use_xt1, sigma, tensile_displ)

def solve_normalstress(x,x_bnd,sigma_closure,dx,sigmaext_max,a,sigma_yield,crack_model,verbose=False, diag_plots=False,calculate_displacements=True):
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

    if seff[0] is > 0, and reff[0] > 0 then sigma_closure corresponding
    to reff[0] is assumed to match seff[0]. 


"""

    sigma_closure = np.ones(x.shape,dtype='d')*np.inf
    
    last_closure = seff[0] # If everything is closed, then closure stresses and external loads match. 
    
    for lcnt in range(1,reff.shape[0]):
        # In each step, we solve for a new linear segment
        # of the sigma_closure distribution.
        # we assume last_closure is the end closure stress
        # from the previous step, corresponding to a
        # position of reff[lcnt-1], with an
        # external load of seff[lcnt-1] opening the
        # crack to this point

        # So at this step, if the new closure stress is
        # new_closure, then
        # in between we have a line:
        # sigma_closure(x) = last_closure + (new_closure-last_closure)*(x-reff[lcnt-1])
        # So we need to iteratively solve for a new_closure that satisfies
        
        # (reff[lcnt], sigma, tensile_displ) = solve_normalstress(x,x_bnd,sigma_closure,dx,seff[lcnt],a,sigma_yield,crack_model)
        # For the given reff[lcnt], seff[lcnt]

        def goal(new_closure):
            new_closure_field = sigma_closure

            if lcnt==1:
                # first iteration: extrapolate back to crack center
                # if necessary
                #new_zone = (x <= reff[lcnt])

                # extrapolate out beyond reff[lcnt] to help with
                # convergence

                new_zone = np.ones(x.shape,dtype=np.bool)
                pass
            else:
                #new_zone = (x >= reff[lcnt-1]) & (x <= reff[lcnt])
                # extrapolate out beyond reff[lcnt] to help with
                # convergence
                new_zone = (x >= reff[lcnt-1]) 
                pass
            
            new_closure_field[new_zone] = last_closure + (new_closure-last_closure) * (x[new_zone]-reff[lcnt-1])/(reff[lcnt]-reff[lcnt-1])

            (gotreff, sigma, tensile_displ) = solve_normalstress(x,x_bnd,new_closure_field,dx,seff[lcnt],xt,sigma_yield,crack_model,calculate_displacements=False,verbose=verbose)

            return reff[lcnt]-gotreff

        
        
        (new_closure,infodict,ier,mesg) = scipy.optimize.fsolve(goal,seff[lcnt],full_output=True)
        
        if ier != 1:
            sys.modules["__main__"].__dict__.update(globals())
            sys.modules["__main__"].__dict__.update(locals())
            raise ValueError("Error in inverse_closure fsolve: %s" % str(mesg))
        
        if lcnt==1:
            # first iteration: extrapolate back to crack center
            # if necessary
            new_zone = (x <= reff[lcnt])
            pass
        else:
            new_zone = (x >= reff[lcnt-1]) & (x <= reff[lcnt])
            pass
        
        closure_gradient = (new_closure-last_closure)/(reff[lcnt]-reff[lcnt-1])
        
        sigma_closure[new_zone] = last_closure + (new_closure-last_closure) * (x[new_zone]-reff[lcnt-1])/(reff[lcnt]-reff[lcnt-1])
        
        last_closure = new_closure
        pass

    if reff[lcnt] < xt:
        # don't have data out to the tips
        # extrapolate last closure gradient to the tips
        
        new_zone = (x >= reff[lcnt])
        new_closure=last_closure + closure_gradient * (xt-reff[lcnt])
        
        sigma_closure[new_zone] = last_closure + (new_closure-last_closure) * (x[new_zone]-reff[lcnt])/(xt-reff[lcnt])
        
        pass

    sigma_closure[x > xt] = 0.0
    
    return sigma_closure



def Glinka_ModeI_ThroughCrack(Eeff,x,width,epsx):
    """Create and return ModeI_crack_model corresponding
    to the Through crack weight function from the Glinka paper"""
    
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


    return ModeI_Beta_WeightFunction(weightfun_times_sqrt_aminx=lambda obj,x,a: weightfun_through_times_sqrt_aminx(x,a,width),
                                     epsx=epsx,
                                     Eeff=Eeff,
                                     surrogate_a=x,
                                     use_surrogate=True)




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

def ModeI_throughcrack_CODformula(Eeff):

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
    
    
    return ModeI_Beta_COD_Formula(Eeff=Eeff,
                                  beta=lambda obj: 1.0,
                                  u = lambda obj,sigma_applied,x,xt: u(obj.Eeff,sigma_applied,x,xt))


def Tada_ModeI_CircularCrack_along_midline(E,nu):

    def u(E,nu,sigma_applied,x,xt):
        # For a circular crack in an infinite space,
        # loaded in mode I.
        # We will be evaluating along a line through the crack center
        # Based on Tada, H., Paris, P., & Irwin, G. (2000). The stress analysis of cracks handbook / Hiroshi Tada, Paul C. Paris, George R. Irwin. (3rd ed.). New York: ASME Press.
        
        u = (4.0*(1-nu**2.0)/(np.pi*E)) * sigma_applied * np.sqrt(xt**2.0 - x**2.0)
        return u
    
    
    return ModeI_Beta_COD_Formula(E=E,
                                  nu=nu,
                                  beta=lambda obj: 4.0/(np.pi**2.0),
                                  u = lambda obj,sigma_applied,x,xt: u(obj.E,obj.nu,sigma_applied,x,xt))



