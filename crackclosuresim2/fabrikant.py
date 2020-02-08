#Penny-shaped crack revisited: Closed-form solutions; V. I. Fabrikant
#https://www.tandfonline.com/loi/tpha20

import sys

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splrep,splev

from .shear_stickslip import ModeII_Beta_CSD_Formula


def complex_quad(func,a,b,*args):
    i=0+1j
    real_part = quad(lambda f: func(f,*args).real,a,b)[0]
    imag_part = quad(lambda f: func(f,*args).imag,a,b)[0]
    return real_part + i*imag_part

    


def u_nondim(rho,phi,nu):
    a=1.0
    tauext=1.0
    E=1.0
    # Assuming this written like a weight function, so the
    # tau is in fact the engineering stress at the crack location

    #H=(1-nu**2.0)/(np.pi*E)
    mu = E/(2.0*(1+nu)) # Note mu is shear modulus not friction coefficient here
    G1 = (2.0-nu)/(2*np.pi*mu)  
    G2 = nu/(2*np.pi*mu)
    i=0+1j



    # w.o.l.o.g let tau_xz = tauext
    # let tau_yz = 0
    # We cut space along the x-z plane @ y=0
    # crack goes along the x axis.
    # We are interested in the surface.
    
    # e_rho = [ cos phi  sin phi     0 ][ xhat ]
    # e_phi = [ -sin(phi)  cos phi   0 ][ yhat ]
    # e_z   = [    0          0      1 ][ e_z  ]
    
    # [ sigma_rho  tau_rhophi  tau_rhoz ]
    # [ tau_rhophi  sigma_phi  tau_phiz ]
    # [ tau_rhoz    tau_phiz   sigma_z  ]

    tau_xyz = np.array( ((0,0,tauext),
                         (0,0,0),
                         (tauext,0,0)),dtype='d')
    xform = np.array( ((np.cos(phi),np.sin(phi),0.0),
                       (-np.sin(phi),np.cos(phi),0.0),
                       (0.0,0.0,1.0)),dtype='d');

    tau_rhophiz = np.matmul(xform,np.matmul(tau_xyz,xform.T))

    tau = tau_rhophiz[2,0] + i*tau_rhophiz[1,0]
    
    R = lambda rho0,phi0: (rho**2.0 + rho0**2.0 - 2.0*rho*rho0*np.cos(phi-phi0))**0.5

    eta = lambda rho0,x: ((x**2.0 - rho**2.0)**0.5) * ((x**2.0-rho0**2.0)**0.5)/x
    xi = lambda rho0,phi0: (rho*np.exp(i*phi)-rho0*np.exp(i*phi0))/(rho*np.exp(-i*phi) - rho0*np.exp(-i*phi0))

    t = lambda rho0, phi0: ((rho*rho0)/(a**2.0))*np.exp(i*(phi-phi0))
    
    
    inner_integrand1 = lambda rho0,phi0: ( (1.0/R(rho0,phi0))*np.arctan(eta(rho0,a)/R(rho0,phi0)) - ((G2**2.0)/(G1**2.0))*(3.0-np.conj(t(rho0,phi0)))*eta(rho0,a)/( (a**2.0)*(1.0-np.conj(t(rho0,phi0)))**2.0 ) )*tau*rho0

    inner_integrand2 = lambda rho0,phi0: ((xi(rho0,phi0)/R(rho0,phi0))*np.arctan(eta(rho0,a)/R(rho0,phi0)) + eta(rho0,a)*(xi(rho0,phi0)-t(rho0,phi0)*np.exp(2.0*i*phi0))/( (a**2.0)*(1-t(rho0,phi0))*(1-np.conj(t(rho0,phi0)))))*np.conj(tau)*rho0

    outer_integrand1 = lambda phi0: complex_quad(inner_integrand1,0,a,phi0)
    outer_integrand2 = lambda phi0: complex_quad(inner_integrand2,0,a,phi0)
    
    print("u complex_quad(outer_integrand1,0,2pi")
    integral1 = complex_quad(outer_integrand1,0,2.0*np.pi)
    print("u complex_quad(outer_integrand2,0,2pi")
    integral2 = complex_quad(outer_integrand2,0,2.0*np.pi)

    u = (G1/np.pi)*integral1 + (G2/np.pi)*integral2

    u_rhophiz = np.array((u.real,u.imag,0.0),dtype='d')
    u_xyz = np.matmul(xform.T,u_rhophiz[:,np.newaxis])

    # we are interested in u_x
    return u_xyz[0,0]


# Surrogate index: phi, nu
# Surrogate value: splrep (t,c,k)
# u_surrogate filled by putting different values
# of nu into the fabrikant_example.py,
# and pasting in the generated output
u_surrogate = {
(0.0,0.32): (np.array([0.                 ,0.                 ,0.                 ,
 0.                 ,0.06896551724137931,0.10344827586206896,
 0.13793103448275862,0.1724137931034483 ,0.20689655172413793,
 0.24137931034482757,0.27586206896551724,0.3103448275862069 ,
 0.3448275862068966 ,0.3793103448275862 ,0.41379310344827586,
 0.4482758620689655 ,0.48275862068965514,0.5172413793103449 ,
 0.5517241379310345 ,0.5862068965517241 ,0.6206896551724138 ,
 0.6551724137931034 ,0.6896551724137931 ,0.7241379310344828 ,
 0.7586206896551724 ,0.7931034482758621 ,0.8275862068965517 ,
 0.8620689655172413 ,0.896551724137931  ,0.9310344827586207 ,
 1.                 ,1.                 ,1.                 ,
 1.                 ],dtype=np.dtype('float64')),np.array([1.3605473992312882 ,1.3605467022441233 ,1.3589319856729614 ,
 1.3535217779918054 ,1.3478204787872716 ,1.3404546880386035 ,
 1.3313968037504786 ,1.3206120774339523 ,1.3080578754752972 ,
 1.293682786259433  ,1.2774254679808084 ,1.2592131835950464 ,
 1.2389599333003354 ,1.2165640575335757 ,1.1919051350157    ,
 1.164839916755667  ,1.1351969376825624 ,1.1027691809429085 ,
 1.0673041540989188 ,1.0284891279927673 ,0.9859324466198902 ,
 0.939124958082455  ,0.8874190282299359 ,0.8298443369075583 ,
 0.7653536699466088 ,0.6911079539250612 ,0.6072934234489481 ,
 0.4583099568128398 ,0.35512588869423806,0.                 ,
 0.                 ,0.                 ,0.                 ,
 0.                 ],dtype=np.dtype('float64')),3),
(0.0,0.33): (np.array([0.                 ,0.                 ,0.                 ,
 0.                 ,0.06896551724137931,0.10344827586206896,
 0.13793103448275862,0.1724137931034483 ,0.20689655172413793,
 0.24137931034482757,0.27586206896551724,0.3103448275862069 ,
 0.3448275862068966 ,0.3793103448275862 ,0.41379310344827586,
 0.4482758620689655 ,0.48275862068965514,0.5172413793103449 ,
 0.5517241379310345 ,0.5862068965517241 ,0.6206896551724138 ,
 0.6551724137931034 ,0.6896551724137931 ,0.7241379310344828 ,
 0.7586206896551724 ,0.7931034482758621 ,0.8275862068965517 ,
 0.8620689655172413 ,0.896551724137931  ,0.9310344827586207 ,
 1.                 ,1.                 ,1.                 ,
 1.                 ],dtype=np.dtype('float64')),np.array([1.3587829440880288,1.35878224800477  ,1.3571696255135128,
 1.3517664341764775,1.3460725288244524,1.338716290559762 ,
 1.3296701531836246,1.3188994132713705,1.306361492499003 ,
 1.2920050459271246,1.2757688113008094,1.2575801459086227,
 1.237353161475815 ,1.2149863302825281,1.1903593872181157,
 1.1633292690683246,1.133724733129758 ,1.1013390310238826,
 1.065919997761918 ,1.0271553097575779,0.9846538189292913,
 0.9379070336991843,0.8862681597575713,0.8287681354469675,
 0.7643611045844153,0.6902116757185301,0.6065058419172994,
 0.4577155876926318,0.3546653360526447,0.                ,
 0.                ,0.                ,0.                ,
 0.                ],dtype=np.dtype('float64')),3),
(0.0,0.342): (np.array([0.                 ,0.                 ,0.                 ,
 0.                 ,0.06896551724137931,0.10344827586206896,
 0.13793103448275862,0.1724137931034483 ,0.20689655172413793,
 0.24137931034482757,0.27586206896551724,0.3103448275862069 ,
 0.3448275862068966 ,0.3793103448275862 ,0.41379310344827586,
 0.4482758620689655 ,0.48275862068965514,0.5172413793103449 ,
 0.5517241379310345 ,0.5862068965517241 ,0.6206896551724138 ,
 0.6551724137931034 ,0.6896551724137931 ,0.7241379310344828 ,
 0.7586206896551724 ,0.7931034482758621 ,0.8275862068965517 ,
 0.8620689655172413 ,0.896551724137931  ,0.9310344827586207 ,
 1.                 ,1.                 ,1.                 ,
 1.                 ],dtype=np.dtype('float64')),np.array([1.3562320321167172 ,1.3562313373402777 ,1.3546217423066627 ,
 1.349228694654256  ,1.3435454787589198 ,1.336203050732343  ,
 1.3271738961276869 ,1.3164233766705706 ,1.3039089939707074 ,
 1.2895794994830823 ,1.2733737459613184 ,1.2552192270701654 ,
 1.2350302157796056 ,1.2127053749696386 ,1.1881246653119226 ,
 1.1611452921704783 ,1.131596334325048  ,1.0992714315366818 ,
 1.0639188922176375 ,1.0252269791235693 ,0.9828052784945346 ,
 0.9361462533695286 ,0.8846043236988804 ,0.82721224709275   ,
 0.7629261308084664 ,0.6889159064171179 ,0.6053672178129533 ,
 0.45685629506087994,0.35399950487338433,0.                 ,
 0.                 ,0.                 ,0.                 ,
 0.                 ],dtype=np.dtype('float64')),3),
(0.0,0.3): (np.array([0.                 ,0.                 ,0.                 ,
 0.                 ,0.06896551724137931,0.10344827586206896,
 0.13793103448275862,0.1724137931034483 ,0.20689655172413793,
 0.24137931034482757,0.27586206896551724,0.3103448275862069 ,
 0.3448275862068966 ,0.3793103448275862 ,0.41379310344827586,
 0.4482758620689655 ,0.48275862068965514,0.5172413793103449 ,
 0.5517241379310345 ,0.5862068965517241 ,0.6206896551724138 ,
 0.6551724137931034 ,0.6896551724137931 ,0.7241379310344828 ,
 0.7586206896551724 ,0.7931034482758621 ,0.8275862068965517 ,
 0.8620689655172413 ,0.896551724137931  ,0.9310344827586207 ,
 1.                 ,1.                 ,1.                 ,
 1.                 ],dtype=np.dtype('float64')),np.array([1.3631152773047044 ,1.3631145790018118 ,1.361496814837971  ,
 1.3560763960061708 ,1.3503643362495183 ,1.3429846434108577 ,
 1.3339096633989767 ,1.3231045821411345 ,1.3105266855584936 ,
 1.2961244650003398 ,1.2798364628871033 ,1.2615898048926906 ,
 1.241298328886432  ,1.2188601835067914 ,1.1941547200352132 ,
 1.16703841927511   ,1.137339492455854  ,1.1048505320235305 ,
 1.0693185689855091 ,1.0304302839442416 ,0.9877932816852867 ,
 0.9408974493603797 ,0.8890939304252008 ,0.8314105734744383 ,
 0.7667981877236193 ,0.692412341381753  ,0.6084396205358166 ,
 0.4591749645956898 ,0.35579614831403217,0.                 ,
 0.                 ,0.                 ,0.                 ,
 0.                 ],dtype=np.dtype('float64')),3),
(0.0,0.292): (np.array([0.                 ,0.                 ,0.                 ,
 0.                 ,0.06896551724137931,0.10344827586206896,
 0.13793103448275862,0.1724137931034483 ,0.20689655172413793,
 0.24137931034482757,0.27586206896551724,0.3103448275862069 ,
 0.3448275862068966 ,0.3793103448275862 ,0.41379310344827586,
 0.4482758620689655 ,0.48275862068965514,0.5172413793103449 ,
 0.5517241379310345 ,0.5862068965517241 ,0.6206896551724138 ,
 0.6551724137931034 ,0.6896551724137931 ,0.7241379310344828 ,
 0.7586206896551724 ,0.7931034482758621 ,0.8275862068965517 ,
 0.8620689655172413 ,0.896551724137931  ,0.9310344827586207 ,
 1.                 ,1.                 ,1.                 ,
 1.                 ],dtype=np.dtype('float64')),np.array([1.3637916255185762,1.3637909268711426,1.362172360002695 ,
 1.3567492516805597,1.351034357722867 ,1.343651003241228 ,
 1.3345715204211344,1.3237610779164406,1.3111769404548457,
 1.2967675738269024,1.2804714899618184,1.2622157783672416,
 1.2419142341866263,1.2194649554296124,1.1947472337177374,
 1.1676174783331907,1.1379038158949426,1.105398734804254 ,
 1.069849141669016 ,1.0309415610802368,0.9882834032727281,
 0.9413643022519675,0.8895350795355438,0.8318231013550474,
 0.7671786563408485,0.6927559013564892,0.6087415150691826,
 0.4594027972467066,0.3559726866393318,0.                ,
 0.                ,0.                ,0.                ,
 0.                ],dtype=np.dtype('float64')),3),
(0.0,0.31): (np.array([0.                 ,0.                 ,0.                 ,
 0.                 ,0.06896551724137931,0.10344827586206896,
 0.13793103448275862,0.1724137931034483 ,0.20689655172413793,
 0.24137931034482757,0.27586206896551724,0.3103448275862069 ,
 0.3448275862068966 ,0.3793103448275862 ,0.41379310344827586,
 0.4482758620689655 ,0.48275862068965514,0.5172413793103449 ,
 0.5517241379310345 ,0.5862068965517241 ,0.6206896551724138 ,
 0.6551724137931034 ,0.6896551724137931 ,0.7241379310344828 ,
 0.7586206896551724 ,0.7931034482758621 ,0.8275862068965517 ,
 0.8620689655172413 ,0.896551724137931  ,0.9310344827586207 ,
 1.                 ,1.                 ,1.                 ,
 1.                 ],dtype=np.dtype('float64')),np.array([1.3619896147764663,1.361988917050418 ,1.3603724888380129,
 1.354956546196161 ,1.3492492034664503,1.3418756047887783,
 1.3328081189074477,1.3220119605041403,1.3094444507630072,
 1.2950541235799442,1.2787795721222999,1.2605479822415921,
 1.2402732629769853,1.2178536470008052,1.1931685853850171,
 1.1660746772914363,1.1364002758914313,1.1039381448880197,
 1.0684355241954209,1.0295793531551247,0.9869775605930041,
 0.9401204549133851,0.8883597153957132,0.8307239934498657,
 0.7661649646917695,0.691840546289151 ,0.6079371702350308,
 0.4587957772581772,0.3555023313540416,0.                ,
 0.                ,0.                ,0.                ,
 0.                ],dtype=np.dtype('float64')),3),
}

def array_repr(array):
    return "np.array(%s,dtype=np.%s)" % (np.array2string(array,separator=',',suppress_small=False,threshold=np.inf,floatmode='unique'),repr(array.dtype))

def u(rho,phi,a,tauext,E,nu,use_surrogate=True):
    # NOTE: returns shear displacement of each crack flank
    # relative displacement between both flanks is double this

    
    if use_surrogate and (phi,nu) in u_surrogate:
        (t,c,k) = u_surrogate[(phi,nu)]
        return splev(rho/a,(t,c,k))*a*tauext/E

    if use_surrogate:
        raise ValueError("crackclosuresim2.fabrikant: WARNING: Surrogate not available for u(phi=%.20e,nu=%.20e); computation will be extremely slow. Pass use_surrogate=False if you really want this!\n" % (phi,nu))
    
    
    return u_nondim(rho/a,phi,nu)*a*tauext/E
    pass


def K_nondim(phi,nu):
    a=1.0
    tauext=1.0
    
    i=0+1j
    tau_xyz = np.array( ((0,0,tauext),
                         (0,0,0),
                         (tauext,0,0)),dtype='d')
    xform = np.array( ((np.cos(phi),np.sin(phi),0.0),
                       (-np.sin(phi),np.cos(phi),0.0),
                       (0.0,0.0,1.0)),dtype='d');

    tau_rhophiz = np.matmul(xform,np.matmul(tau_xyz,xform.T))

    tau = tau_rhophiz[2,0] + i*tau_rhophiz[1,0]

    G2_over_G1 = nu*(2*np.pi)/((2*np.pi)*(2.0-nu))

    inner_integrand1 = lambda rho0,phi0: np.sqrt(a**2.0-rho0**2.0)*tau*rho0/(a**2.0 + rho0**2.0 - 2*a*rho0*np.cos(phi-phi0))

    inner_integrand2 = lambda rho0,phi0: ((3.0*a*np.exp(-i*phi)-rho0*np.exp(-i*phi0))/((a*np.exp(-i*phi)-rho0*np.exp(-i*phi0))**2.0))*np.sqrt(a**2.0-rho0**2.0)*np.conj(tau)*rho0

    #outer_integrand1_full = lambda phi0: complex_quad(inner_integrand1,0,a,phi0)
    # Integrate only to .999985*crack length (a-15e-6) to avoid a nasty message
    # from the QUADPACK integrator. This does cause a .4% error, but
    # that is substantially less than the other expected errors in our
    # various calculations.
    #
    # ... Should probably verify the integral using an independent
    # integration tool ... 
    outer_integrand1 = lambda phi0: complex_quad(inner_integrand1,0,a-15e-6,phi0)
    #outer_integrand2_full = lambda phi0: complex_quad(inner_integrand2,0,a,phi0)
    outer_integrand2 = lambda phi0: complex_quad(inner_integrand2,0,a-15e-6,phi0)

    #print("K complex_quad(outer_integrand1,0,2pi")
    #integral1_full = complex_quad(outer_integrand1_full,0,2.0*np.pi)
    integral1 = complex_quad(outer_integrand1,0,2.0*np.pi)
    
    #print("K complex_quad(outer_integrand2,0,2pi")
    #integral2_full = complex_quad(outer_integrand2_full,0,2.0*np.pi)
    integral2 = complex_quad(outer_integrand2,0,2.0*np.pi)


    
    #K_full = (-1.0/(np.pi**2.0 * np.sqrt(2.0*a))) * (integral1_full + (G2_over_G1*np.exp(i*phi)/a)*integral2_full)
    
    K = (-1.0/(np.pi**2.0 * np.sqrt(2.0*a))) * (integral1 + (G2_over_G1*np.exp(i*phi)/a)*integral2)

    #print("K_full.real=%g" % (K_full.real))
    #print("K.real=%g" % (K.real))
    
    return K.real  # Interested in K_II only

# K_surrogate filled by putting different values
# of nu into the fabrikant_example.py,
# and pasting in the generated output
K_surrogate={
    (np.pi,0.32): 0.5334369467174328,
    (np.pi,0.33): 0.5391115671127011,
    (np.pi,0.342): 0.540547836093071,
    (np.pi,0.3): 0.527132211645343,
    (np.pi,0.292): 0.5246516601415699,
    (np.pi,0.31): 0.5302659261190444,
}

def K(phi,a,tauext,nu,use_surrogate=True):
    
    if use_surrogate and (phi,nu) in K_surrogate:
        return K_surrogate[(phi,nu)]*tauext*np.sqrt(a)

    if use_surrogate:
        sys.stderr.write("crackclosuresim2.fabrikant: WARNING: Surrogate not available for K; computation will be extremely slow\n")
        pass
    
    K=K_nondim(phi,nu)*tauext*np.sqrt(a)
    return K



def Fabrikant_ModeII_CircularCrack_along_midline(E,nu):
    
    # For beta,
    # K = tau*sqrt(pi*a*beta)
    #   = K_nondim*tau*sqrt(a)
    # Therefore K_nondim*tau*sqrt(a) = tau*sqrt(pi*a*beta)
    # Therefore K_nondim = sqrt(pi*beta)
    # Therefore K_nondim/sqrt(pi) = sqrt(beta)
    # Therefore K_nondim^2/pi = beta
    
    # use pi instead of 0 as phi for K because
    # otherwise we (undesirably) get the negative of what we want
    if (np.pi,nu) in K_surrogate:
        K_nd_val = K_surrogate[(np.pi,nu)]
        pass
    else:
        K_nd_val = K_nondim(np.pi,nu)
        pass
    
    # u(x,0.0,xt,tau_applied,obj.E,obj.nu)
    uvec = np.vectorize(u)
    
    return ModeII_Beta_CSD_Formula(E=E,
                                   nu=nu,
                                   beta = lambda obj: (K_nd_val**2.0)/np.pi,
                                   ut = lambda obj,tau_applied,x,xt: uvec(x,0.0,xt,tau_applied,obj.E,obj.nu))

pass



    
