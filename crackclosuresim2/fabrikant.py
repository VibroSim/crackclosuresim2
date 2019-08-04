#Penny-shaped crack revisited: Closed-form solutions; V. I. Fabrikant
#https://www.tandfonline.com/loi/tpha20

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
# of nu into the main routine below,
# and pasting in the generated output
u_surrogate = {
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
}

def array_repr(array):
    return "np.array(%s,dtype=np.%s)" % (np.array2string(array,separator=',',suppress_small=False,threshold=np.inf,floatmode='unique'),repr(array.dtype))

def u(rho,phi,a,tauext,E,nu,use_surrogate=True):

    if use_surrogate and (phi,nu) in u_surrogate:
        (t,c,k) = u_surrogate[(phi,nu)]
        return splev(rho/a,(t,c,k))*a*tauext/E

    if use_surrogate:
        sys.stderr.write("crackclosuresim2.fabrikant: WARNING: Surrogate not available for u; computation will be extremely slow\n")
        pass
    
    
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
    print("K.real=%g" % (K.real))
    
    return K.real  # Interested in K_II only

# K_surrogate filled by putting different values
# of nu into the main routine below,
# and pasting in the generated output
K_surrogate={
    (np.pi,0.33): 0.5391115671127011,
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



    
