
import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize

if __name__=="__main__":
    from matplotlib import pyplot as pl
    pl.rc('text', usetex=True) # Support greek letters in plot legend
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

# For a mode I crack with the tip at the origin, intact material
# to the right (x > 0), broken material to the left (x < 0)
# The tensile stress @ theta=0 multiplied by sqrt(x)/(sqrt(a)*sigmaext)
# where x ( > 0) is the position where the stress is measured,
# a is the (half) length of the crack, and sigmaext
# is the external tensile load
sigmaI_theta0_times_rootx_over_sqrt_a_over_sigmaext = 1.0/sqrt(2.0)  # Per Suresh (9.43 and 9.44a) and Anderson (table 2.1)



def integral_tensilestress_growing_effective_crack_length_byxt(x,sigmaext1,sigmaext_max,F,xt1,xt2):
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

    Let sigmaI(r,K) be the mode I normal stress formula divided by 
    the external load, where the K dependence is presumed to be 
    strictly multiplicative. Let K_over_sigmaext = K/sigmaext

    Let xt be the position of the effective tip at external load
    sigmaext

    The incremental normal stress field would be
    integral_sigmaext1^sigmaext2 of sigmaI(x-xt,K_over_sigmaext) dsigmaext
    (note that K and xt are dependent on sigmaext)
   

    The variable sigmaI_theta0_times_rootx_over_sqrt_a_over_sigmaext
    represents the value of sigmaI(x,K) with sigmaext*sqrt(pi*a) 
    substituted for K, evaluated for horizontal axis beyond the 
    tip (i.e. theta=0) and then multiplied by sqrt(x) (sqrt(position
    beyond the tip) and divided by sqrt(cracklength) and also by 
    sigmaext. 

    Then we can rewrite the incremental normal stress as: 
    integral_sigmaext1^sigmaext2 of sigmaI_theta0_times_rootx_over_sqrt_a_over_sigmaext*sqrt(xt)/sqrt(x-xt) dsigmaext
    Here, xt is still dependent on sigmaext... this will give normal stress
    as a function of position (x). 

    We assume xt is linearly dependent on normal stress:
    xt = xtp + (1/F)*(sigmaext-sigmaext1)
    where xtp is the final xt from the previous step. 
    
    sigmaI_theta0_times_rootx_over_sqrt_a_over_sigmaext is a constant, 
    so it will just be a leading coefficient that we will ignore from 
    here on (except when multiplying it in at the final step). 
    
    So our incremental tension is
    integral_sigmaext1^sigmaext2 (1.0 + sqrt(xt)/sqrt(x-xt)) dsigmaext  (the new 1.0 term represents that beyond the effective tip the external load directly increments the stress state, in addition to the stress concentration caused by the presence of the open region)
    where we ignore any contributions corresponding to (x-xt) <= 0
    
    Perform change of integration variable sigmaext -> xt: 
       Derivative of xt:
       dxt = (1/F)*dsigmaext
       dsigmaext = F*dxt


    So the  incremental normal stress we are solving for is
    (sigmaext2-sigmaext1) + integral_xt1^xt2 sqrt(xt)*F/sqrt(x-xt)  dxt
    where we ignore any contributions corresponding to (x-xt) <= 0

    and sigmaext2 = sigmaext1 + (xt2-xt1)*F 
 
    F is a constant so have 
    F * (xt2-xt1)  + F * integral_xt1^xt2 sqrt(xt)/(sqrt(x-xt))  dxt


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
    F*(upper_bound-xt1) + (sigmaI_theta0_times_rootx_over_sqrt_a_over_sigmaext*F)*(indef_integral_of_simple_squareroot_quotients(x,upper_bound) - indef_integral_of_simple_squareroot_quotients(x,xt1))

    """
    
    # For a mode I tension crack with the tip at the origin, intact material
    # to the right (x > 0), broken material to the left (x < 0)
    # The tensile stress @ theta=0 multiplied by sqrt(x)/(sqrt(a)*sigmaext)
    # where x ( > 0) is the position where the stress is measured,
    # a is the (half) length of the crack, and sigmaext
    # is the external tensile load
    # sigmaI_theta0_times_rootx_over_sqrt_a_over_sigmaext = 1.0/sqrt(2.0)  # Per Suresh (9.43 and 9.44a) and Anderson (table 2.1) (now a global)

    sigmaext2 = sigmaext1 + (xt2-xt1)*F

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
    
    
    
    upper_bound = use_xt2*np.ones(x.shape,dtype='d')
    
    # alternate upper_bound:
    use_alternate = x < upper_bound
    upper_bound[use_alternate] = x[use_alternate]
    
    res=np.zeros(x.shape,dtype='d')

    nonzero = x > xt1
    
    
    
    res[nonzero] = F*(upper_bound[nonzero]-xt1) + (sigmaI_theta0_times_rootx_over_sqrt_a_over_sigmaext*F) * (indef_integral_of_simple_squareroot_quotients(x[nonzero],upper_bound[nonzero]) - indef_integral_of_simple_squareroot_quotients(x[nonzero],xt1))

    

    return (use_xt2,sigmaext2,res)


def solve_incremental_tensilestress(x,x_bnd,sigma,sigma_closure,tensile_displ,xt_idx,dx,sigmaext,sigmaext_max,a,E,nu):
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

    (use_xt2,sigmaext2,sigma_increment)=integral_tensilestress_growing_effective_crack_length_byxt(x,sigmaext,sigmaext_max,F,x[xt_idx],x[xt_idx+1])

    But to do this we need to solve for F. 

    We do this by setting the tensile normal stress equal to the closure stress
    over the new step (unit of dx).
    """

    next_bound = x_bnd[xt_idx+1]
    if next_bound > a:
        next_bound=a
        pass
    
    def obj_fcn(F):
        (use_xt2,sigmaext2,sigma_increment)=integral_tensilestress_growing_effective_crack_length_byxt(x,sigmaext,sigmaext_max,F,x_bnd[xt_idx],next_bound)
        return (sigma+sigma_increment - sigma_closure)[xt_idx]
    
    # F measures the closure gradient in (Pascals external tensile stress / meters of tip motion)
    
    if sigma_closure[xt_idx] >= 0.0 and sigma[xt_idx] < sigma_closure[xt_idx]:
        # There is a closure stress here but not yet the full external tensile load to counterbalance it

        # Bound it by 0  and the F that will give the maximum
        # contribution of sigma_increment: 2.0*(sigmaext_max-sigmaext1)/(xt2-xt1)
        Fbnd = 2.0*(sigmaext_max - sigmaext)/(next_bound-x_bnd[xt_idx])
        
        if obj_fcn(Fbnd) < 0.0:
            # Maximum value of objective is < 0... This means that
            # with the steepest sigma vs. xt slope possible (given
            # the total tensile load we are applying) we still
            # can't get sigma+sigma_increment to match sigma_closure.
            # ... We will have to make do with sigma+sigma_increment
            #  < sigma_closure
            # So our best result is just Fbnd
            F=Fbnd
            pass
        else:
            # brentq requires function to be different signs
            # at 0.0 (negative) and Fbnd (positive) 
            F = scipy.optimize.brentq(obj_fcn,0.0,Fbnd,disp=True)
            pass
        
        (use_xt2,sigmaext2,sigma_increment)=integral_tensilestress_growing_effective_crack_length_byxt(x,sigmaext,sigmaext_max,F,x_bnd[xt_idx],next_bound)
        
        # For displacement calculate at x centers... use average of left and right boundaries, except for (perhaps) last point where instead of the right boundary we use the actual tip.
        incremental_displacement = np.zeros(x.shape[0],dtype='d')
        xt = (x_bnd[xt_idx]+use_xt2)/2.0
        left_of_effective_tip = (x < xt)
        incremental_displacement[left_of_effective_tip] = tensile_displacement(sigmaext2-sigmaext,x[left_of_effective_tip],xt,E,nu)
        pass
    else:
        # No closure stress at this point, or sigma is already at the limit
        # of what can be supported here
        
        use_xt2 = x_bnd[xt_idx+1]
        sigmaext2 = sigmaext
        sigma_increment = np.zeros(x.shape[0],dtype='d')
        incremental_displacement = np.zeros(x.shape[0],dtype='d')
        pass
    return (use_xt2,sigmaext2, sigma+sigma_increment, tensile_displ+incremental_displacement) 
   
    


#####TENSILE DISPLACEMENT FUNCTION 
#A function for tensile displacement was obtained from Suresh (9.44b)
#This function calculated the displacement in the y direction, uy, 
#in Mode I. This is for a semi-infinite crack in an infinite plate
#of an isotropic and homogeneous solid 
#NOTE: This function will always produce uy=0 for theta equals 0. 
#This is because sin(0)=0 and the function is multiplied completely
#by this. 
def tensile_displacement(sigma_applied,x,xt,E,nu):
    #plane stress is considered
    Kappa = (3.0-nu)/(1.0+nu)
    KI = sigma_applied*np.sqrt(np.pi*(xt))
    if (xt-x < 0).any():
        #sys.modules["__main__"].__dict__.update(globals())
        #sys.modules["__main__"].__dict__.update(locals())
        raise ValueError("tensile displacement to right of crack tip")
    theta = np.pi
    return (KI/(2.0*E))*(np.sqrt((xt-x)/(2.0*np.pi)))*((1.0+nu)* 
    (((2.0*Kappa+1.0)*(np.sin(theta/2.0)))-np.sin(3.0*theta/2.0)))


def solve_normalstress(x,x_bnd,sigma_closure,dx,sigmaext_max,a,E,nu,sigma_yield,verbose=False):
    #Initialize the external applied tensile stress starting at zero
    
    sigmaext = 0.0 # External tensile load in this step (Pa)
    
    

    #####MAIN SUPERPOSITION LOOP

    #Initialize tensile stress field (function of x)
    sigma = np.zeros(x.shape,dtype='d')
    
    #Initialized the Displacement state as zero
    tensile_displ = np.zeros(x.shape,dtype='d')
    
    #Initialize x step counter
    xt_idx = 0
    
    use_xt2=0
    
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
        
        (use_xt2,sigmaext, sigma, tensile_displ) = solve_incremental_tensilestress(x,x_bnd,sigma,sigma_closure,tensile_displ,xt_idx,dx,sigmaext,sigmaext_max,a,E,nu)
        
        
        if use_xt2 < x_bnd[xt_idx+1] or sigmaext==sigmaext_max or use_xt2 >= a:
            # Used up all of our applied load or all of our crack... Done!
            done=True
            pass
        
        if verbose: 
            #Print what is happening in the loop
            print("Step: %d @ x=%f mm: %f MPa of tension held" % (xt_idx,x[xt_idx]*1e3,sigmaext/1e6))
            print("Tensile displacement @ x=%f mm: %f nm" % (x[0]*1e3, tensile_displ[0]*1e9))
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
        
        #sigma_increment = sigmaI_theta0_times_rootx_over_sqrt_a_over_sigmaext*(sigmaext_max-sigmaext)*sqrt(a)/sqrt(x-a)
        sigma_increment[si_nodivzero_nonegsqrt] = sigmaI_theta0_times_rootx_over_sqrt_a_over_sigmaext*(sigmaext_max-sigmaext)*sqrt(a)/sqrt(x[si_nodivzero_nonegsqrt]-a)
        sigma_increment[si_divzero]=np.inf
        
        # Limit tensile stresses at physical tip (and elsewhere) to yield
        sigma_increment[sigma + sigma_increment > sigma_yield] = sigma_yield-sigma[sigma+sigma_increment > sigma_yield]
        
        # accumulate stresses onto sigma
        sigma += sigma_increment

        # record increment in displacement
        left_of_effective_tip = x < a
        tensile_displ[left_of_effective_tip] += tensile_displacement(sigmaext_max-sigmaext,x[left_of_effective_tip],a,E,nu)
        
        # Record increment in sigmaext
        sigmaext = sigmaext_max
        
        if verbose:
            print("Step: Open to tips @ x=%f mm: %f MPa of tension held" % (a*1e3,sigmaext/1e6))
            print("Tensile displacement @ x=%f mm: %f nm" % (x[0]*1e3, tensile_displ[0]*1e9))
            pass
        pass
    
    return (use_xt2, sigma, tensile_displ)


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
    
    # Closure state (function of position; positive compression)
    sigma_closure = 80e6/cos(x/a) -70e6 # Pa
    sigma_closure[x > a]=0.0
    
    use_crackclosuresim=False
    if use_crackclosuresim:
        from scipy.interpolate import splrep
        import crackclosuresim.crack_utils_1D as cu1
        specimen_width=25.4e-3
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
    
    (effective_length, sigma, tensile_displ) = solve_normalstress(x,x_bnd,sigma_closure,dx,sigmaext_max,a,E,nu,sigma_yield,verbose=True)
    
    (fig,ax1) = pl.subplots()
    legax=[]
    legstr=[]
    (pl1,pl2,pl3)=ax1.plot(x*1e3,sigma_closure/1e6,'-',
                               x*1e3,sigma/1e6,'-',
                               x*1e3,(sigma-(sigma_closure*(sigma_closure > 0)))/1e6,'-')
    legax.extend([pl1,pl2,pl3])
    legstr.extend(['Closure stress','Tensile stress','$ \\sigma - \\sigma_{\\mbox{\\tiny closure}}$'])
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

    
    (effective_length2, sigma2, tensile_displ2) = solve_normalstress(x,x_bnd,sigma_closure2,dx,sigmaext_max,a,E,nu,sigma_yield,verbose=True)

    (fig2,ax21) = pl.subplots()
    legax=[]
    legstr=[]
    (pl21,pl22,pl23)=ax21.plot(x*1e3,sigma_closure2/1e6,'-',
                              x*1e3,sigma2/1e6,'-',
                              x*1e3,(sigma2-(sigma_closure2*(sigma_closure2 > 0)))/1e6,'-')
    legax.extend([pl21,pl22,pl23])
    legstr.extend(['Closure stress','Tensile stress','$ \\sigma - \\sigma_{\\mbox{\\tiny closure}}$'])
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
    sigma_closure3 = 80e6/cos(x/a) -90e6 # Pa
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
    
    
    (effective_length3, sigma3, tensile_displ3) = solve_normalstress(x,x_bnd,sigma_closure3,dx,sigmaext_max,a,E,nu,sigma_yield,verbose=True)

    (fig3,ax31) = pl.subplots()
    legax=[]
    legstr=[]
    (pl31,pl32,pl33)=ax31.plot(x*1e3,sigma_closure3/1e6,'-',
                               x*1e3,sigma3/1e6,'-',
                               x*1e3,(sigma3-(sigma_closure3*(sigma_closure3 > 0)))/1e6,'-')
    legax.extend([pl31,pl32,pl33])
    legstr.extend(['Closure stress','Tensile stress','$ \\sigma - \\sigma_{\\mbox{\\tiny closure}}$'])
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
    pass

    
    
