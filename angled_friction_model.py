from scipy.interpolate import splrep, splev
import scipy 
import numpy as np
from matplotlib import pyplot as pl
from scipy.integrate import quad
from scipy.optimize import newton
from numpy.random import rand

import crackclosuresim.crack_utils_1D as cu1

i=(0+1j) # imaginary number

# pdf (need not be normalized) of surface orientation beta
# domain: -pi (backwards trajectory, infinitesimally downward)
# to pi (backwards trajectory, infinitesimally upward)

# note: our beta is 90deg - evans and hutchinson beta
beta_components = ( (2.0,0.0,np.pi/4.0), )  # Each entry: component magnitude, gaussian mean, Gaussian sigma

friction_coefficient=0.3

vibration_frequency=20e3  # (Hz)

static_load=60e6  # tensile static load of 60MPa
vib_normal_stress_ampl = 40e6  # vibrational normal stress amplitude. 
vib_shear_stress_ampl = 20e6  # Assume shear amplitude peaks simultaneously with
# normal stress.
# assume also that there is no synergy between heating from different modes. 

# x is position along crack (currently no x dependence to beta 
beta_unnorm_pdf = lambda beta,x:  np.array([ (magnitude/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(beta - mean)**2/(2.0*sigma**2.0)) for (magnitude,mean,sigma) in beta_components ],dtype='d').sum(0)

# crackclosuresim parameters
# Low K
#E = 109e9
nu = 0.33
E = 207.83e9 # Plane stress
#E = 207.83e9/(1.0-nu**2.0) # Plane strain
width=25.4e-3


# units of meters? half-crack lengths for a surface crack  
reff_rightside=np.array([ .5e-3, .7e-3, .9e-3, 1.05e-3, 1.2e-3, 1.33e-3, 1.45e-3, 1.56e-3, 1.66e-3],dtype='d')

# opening stresses, units of Pa
seff_rightside=np.array([ 0.0, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')

# units of meters? half-crack lengths for a surface crack  
reff_leftside=np.array([ .5e-3, .7e-3, .9e-3, 1.05e-3, 1.2e-3, 1.33e-3, 1.45e-3, 1.56e-3, 1.66e-3],dtype='d')

# opening stresses, units of Pa
seff_leftside=np.array([ 0.0, 50e6, 100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6],dtype='d')


stress_field_spl_leftside=cu1.inverse_closure(reff_leftside,seff_leftside,cu1.weightfun_through,(width,))
stress_field_spl_rightside=cu1.inverse_closure(reff_rightside,seff_rightside,cu1.weightfun_through,(width,))






aleft=-np.max(reff_leftside)
aright=np.max(reff_rightside)

approximate_xstep=25e-6 # 25um
numsteps=int((aright-aleft)//approximate_xstep)
xstep = (aright-aleft)/(numsteps-1.0)
numdraws=20 # draws per step

xrange = aleft + xstep*np.arange(numsteps)
power_per_m2 = np.zeros(xrange.shape[0],dtype='d')
vibration_ampl = np.zeros(xrange.shape[0],dtype='d')

for xcnt in range(xrange.shape[0]):
    x=xrange[xcnt]

    # determine normalization factor for pdf at this x position
    beta_unnorm_int=quad(lambda beta: beta_unnorm_pdf(beta,x),-np.pi,np.pi)[0]

    # normalized pdf
    beta_pdf = lambda beta: beta_unnorm_pdf(beta,x)/beta_unnorm_int    

    # cdf 
    beta_cdf = lambda beta: quad(beta_pdf,-np.pi,beta)[0]
    
    # inverse of cdf   # CDF(beta) = prob -> 
    beta_cdf_inverse = lambda prob: newton(lambda beta: beta_cdf(beta)-prob,0.0)
    
    beta_draws = np.vectorize(beta_cdf_inverse)(rand(numdraws))


    if x < 0.0:
        stress_field_spl=stress_field_spl_leftside
        r=-x
        reff=reff_leftside
        pass
    else: 
        stress_field_spl=stress_field_spl_rightside
        r=x
        reff=reff_rightside
        pass

    closure_state_x = splev(r,stress_field_spl,ext=1) 
    
    # Evaluate closure state on both sides of static load
    closure_point_sub=cu1.find_length(static_load-vib_normal_stress_ampl,stress_field_spl,np.max(reff),cu1.weightfun_through,(width,))
    
    (closure_state_sub_a,closure_state_sub)=cu1.effective_stresses_full(reff,np.max(reff),static_load-vib_normal_stress_ampl,stress_field_spl,cu1.weightfun_through,(width,))

    closure_point_add=cu1.find_length(static_load+vib_normal_stress_ampl,stress_field_spl,np.max(reff),cu1.weightfun_through,(width,))
    (closure_state_add_a,closure_state_add)=cu1.effective_stresses_full(reff,np.max(reff),static_load+vib_normal_stress_ampl,stress_field_spl,cu1.weightfun_through,(width,))

    # ***!!!! bug: closure state from effective_stresses_full appears to drop to zero at physical crack tip.


    # Evaluate at x
    if r <= closure_state_sub_a[0]:
        closure_state_sub_x=0.0
        pass
    else:
        closure_state_sub_x=scipy.interpolate.interp1d(closure_state_sub_a,closure_state_sub,fill_value="extrapolate")(r)
        pass

    if r <= closure_state_add_a[0]:
        closure_state_add_x=0.0
        pass
    else:
        closure_state_add_x=scipy.interpolate.interp1d(closure_state_add_a,closure_state_add,fill_value="extrapolate")(r)
        pass


    # uyy is double calculated value because each side moves by this much
    uyy_add=2.0*cu1.uyy(r,np.max(reff),static_load+vib_normal_stress_ampl,stress_field_spl,cu1.weightfun_through,(width,),E,nu,configuration="PLANE_STRESS")

    uyy_sub=2.0*cu1.uyy(r,np.max(reff),static_load-vib_normal_stress_ampl,stress_field_spl,cu1.weightfun_through,(width,),E,nu,configuration="PLANE_STRESS")

    # Got two closure stress values at this point:
    # closure_state_add_x and closure_state_sub_x

    # and corresponding displacements uyy_add and uyy_sub

    # For the moment, treat global shear stress and
    # global shear displacement as zero. 


    # for each beta draw, evaluate local stress field
    # on the angled asperity

    # Model: normal stress sinusodially varies with closure_ampl around closure_state_ref
    # shear stress sinusoidally varies also in phase
    # (vib_shear_stress_ampl)

    # now consider our draws...
    # We can consider each corresponds to xstep/numdraws units of horizontal
    # distance (but we don't anymore) 

    # represent stress field along crack as a complex number,
    # real part is shear, imaginary part is normal stress (following
    # Evans and Hutchinson Q+iP) except in our case these represent
    # vibration amplitudes as vertical and horizontal forces on a crack
    # segment 
    #
    # Throughout each vibration cycle,
    # stresses cycle between  closure_state_add and closure_state_sub

    # So based on temporary limitation above
    # (no external shear loading applied), Q=0
    # P and Q are FORCES... treat them as acting on quarter-annulus

    # Nominal P and Q are the external loading on this xstep unit
    # that we are trying to match
    # ... We match P (normal)  and don't worry too much about Q (shear)
    Q_static_nominal=0.0
    P_static_nominal=closure_state_x * xstep * np.pi*r/2.0 # A force, total for all of the little contributions from each of the draws

    # For each draw, transform nominal Q and P to 
    # Normal and shear forces on the sliding point.
    N_static_nominal=P_static_nominal*np.cos(beta_draws)+Q_static_nominal*np.sin(beta_draws)
    # T_static_nominal = -P_static_nominal*np.sin(beta_draws) + Q_static_nominal*np.cos(beta_draws)
    # Set T_static_nominal to zero because we don't expect any long-term
    # static shear on the asperity contacts
    T_static_nominal=0.0

    # Transform back to P and Q... sum contributions from all draws
    P_contributions=np.sum(N_static_nominal*np.cos(beta_draws) - T_static_nominal*np.sin(beta_draws))
    #Q_contributions=np.sum(N_static_nominal*np.sin(beta_draws) + T_static_nominal*np.cos(beta_draws))

    # Determine scaling factor for all draws to sum to desired value
    normal_force_factor=P_static_nominal/P_contributions

    # Evaluate P,Q, N, & T assuming this scaling factor
    #P_static = P_static_nominal*normal_force_factor
    #Q_static = Q_static_nominal*normal_force_factor

    #N_static = P_static*np.cos(beta_draws)+Q_static*np.sin(beta_draws)
    #T_static = -P_static*np.sin(beta_draws)+Q_static*np.cos(beta_draws)
    N_static = N_static_nominal*normal_force_factor
    T_static = T_static_nominal*normal_force_factor  # 0

    P_static = N_static*np.cos(beta_draws) - T_static*np.sin(beta_draws)
    Q_static = N_static*np.sin(beta_draws) + T_static*np.cos(beta_draws)

    # P_dynamic per draw
    P_dynamic = (closure_state_add_x - closure_state_sub_x)* (xstep * np.pi*r/2.0)/(2.0*numdraws) # dynamic stress amplitude
    Q_dynamic = 0.0 # Should be shear vibration amplitude

    N_dynamic = P_dynamic*np.cos(beta_draws)+Q_dynamic*np.sin(beta_draws)
    T_dynamic = -P_dynamic*np.sin(beta_draws)+Q_dynamic*np.cos(beta_draws)

    
    # exp( iBeta) = cos(beta)+i*sin(Beta)
    # (Q+iP)*exp(iBeta) = Qcos(beta)-Psin(Beta) + i(Qsin(beta) + Pcos(beta))
    # N =Im( (Q+iP)exp(iBeta) )
    # T =Re( (Q+iP)exp(iBeta) )
    # omega=(np.pi/2-beta_draws)-atan(friction_coefficient)

    # Locking condition: T < mu*N...  here T=T_dynamic, N=N_static-N-dynamic
    ## i.e. Re( (Q+iP)*exp(iBeta)) < mu*Im( (Q+iP)*exp(iBeta))
    ## i.e. Re( (Q+iP)*exp(iBeta))/Im( (Q+iP)*exp(iBeta)) < mu
    ## i.e. cot(angle((Q+iP)*exp(iBeta))) < mu
    ## i.e. acot(cot(angle((Q+iP)*exp(iBeta)))) > acot(mu)
    ## OR acot(cot(angle((Q+iP)*exp(iBeta)))) > acot(mu)-pi
    ## i.e. angle((Q+iP)*exp(iBeta)) > acot(mu) > 0
    ##  OR 0 > angle((Q+iP)*exp(iBeta)) > acot(mu)-pi

    #ang=np.angle((Q + i*P)*np.exp(i*beta_draws))
    #slip = (((ang > 0) & (ang  > np.arctan(1/friction_coefficient))) |
    #        ((ang < 0) & (ang > np.arctan(1/friction_coefficient)-np.pi)))
    slip=np.abs(T_dynamic) >=  -friction_coefficient*(N_static-np.abs(N_dynamic))

    
    PP_vibration_y=uyy_add-uyy_sub
    vibration_ampl[xcnt]=PP_vibration_y/2.0
    tangential_vibration_ampl=np.abs(vibration_ampl[xcnt] * np.sin(beta_draws))*slip
    tangential_vibration_velocity_ampl = 2*np.pi*vibration_frequency*tangential_vibration_ampl

    # Power = (1/2)Fampl*vampl
    # where Fampl = mu*Normal force
    # Q: Are force and velocity always in-phase (probably not)
    # ... What happens when they are not?
    # Expect dynamic stresses and displacements to be in phase, which
    # would put dynamic stresses out of phase with velocity....
    # What effect would that have on power? Should the N_dynamic term
    # drop out?

    # Problem: Experimental observation suggests heating
    # is between linear and quadratic in vibration amplitude.
    # Quadratic would require N to have vibration dependence.
    # P=uNv = u(Nstatic+Ndynamic)v=u(Cn1 + Cn2v)v

    # (Note: N_static term was missing from original calculation)
    Power = 0.5 * (friction_coefficient*(np.abs(N_static)+np.abs(N_dynamic)))*tangential_vibration_velocity_ampl
    TotPower = np.sum(Power)

    power_per_m2[xcnt] = TotPower/(xstep*np.pi*r/2.0)
    

    
    pass

betarange=np.linspace(-np.pi,np.pi,800)
pl.figure(1)
pl.clf()
pl.plot(betarange*180.0/np.pi,beta_pdf(betarange),'-')
pl.xlabel('Facet orientation (degrees from flat)')
pl.ylabel('Probability density (/rad)')
pl.savefig('/tmp/facet_pdf.png',dpi=300)


pl.figure(2)
pl.clf()
pl.plot(xrange*1e3,power_per_m2/1.e3)
pl.xlabel('Position (mm)')
pl.ylabel('Heating power (kW/m^2)')
pl.savefig('/tmp/frictional_heating.png',dpi=300)
pl.show()
