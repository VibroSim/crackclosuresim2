#Penny-shaped crack revisited: Closed-form solutions; V. I. Fabrikant
#https://www.tandfonline.com/loi/tpha20

import numpy as np
from scipy.integrate import quad



def complex_quad(func,a,b,*args):
    i=0+1j
    real_part = quad(lambda f: f.real,a,b,args=args)[0]
    imag_part = quad(lambda f: f.imag,a,b,args=args)[0]
    return real_part + i*imag_part

    


def u(rho,phi,a,tauext,E,nu):
    # Assuming this written like a weight function, so the
    # tau is in fact the engineering stress at the crack location

    H=(1-nu**2.0)/(np.pi*E)
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
                       (0.0,0.0,0.0)),dtype='d');

    tau_rhophiz = np.matmul(xform,np.matmul(tau_xyz,xform.T))

    tau = tau_rhophiz[2,0] + i*tau_rhophiz[1,0]
    
    R = lambda rho0,phi0: (rho**2.0 + rho0**2.0 - 2.0*rho*rho0*np.cos(phi-phi0))**0.5

    eta = lambda rho0,x: ((x**2.0 - rho**2.0)**0.5) * ((x**2.0-rho0**2.0)**0.5)/x
    xi = lambda rho0,phi0: (rho*np.exp(i*phi)-rho0*np.exp(i*phi0))/(rho*exp(-i*phi) - rho0*exp(-i*phi0))

    t = lambda rho0, phi0: ((rho*rho0)/(a**2.0))*np.exp(i*(phi-phi0))
    
    
    inner_integrand1 = lambda rho0,phi0: ( (1.0/R(rho0,phi0))*np.arctan(eta(rho0,a)/R(rho0,phi0)) - ((G2**2.0)/(G1**2.0))*(3.0-np.conj(t(rho0,phi0)))*eta(rho0,a)/( (a**2.0)*(1.0-np.conj(t(rho0,phi0)))**2.0 ) )*tau*rho0

    inner_integrand2 = lambda rho0,phi0: ((xi(rho0,phi0)/R(rho0,phi0))*np.arctan(eta(rho0,a)/R(rho0,phi0)) + eta(rho0,a)*(xi(rho0,phi0)-t(rho0,phi0)*np.exp(2.0*i*phi0))/( (a**2.0)*(1-t(rho0,phi0))*(1-np.conj(t(rho0,phi0)))))*np.conj(tau)*rho0

    outer_integrand1 = lambda phi0: complex_quad(inner_integrand1,0,a,phi0)
    outer_integrand2 = lambda phi0: complex_quad(inner_integrand2,0,a,phi0)
    
    integral1 = complex_quad(outer_integrand1,0,2.0*np.pi)
    integral2 = complex_quad(outer_integrand2,0,2.0*np.pi)

    u = (G1/np.pi)*integral1 + (G2/np.pi)*integral2

    u_rhophiz = np.array((u.real,u.imag,0.0),dtype='d')
    u_xyz = np.matmul(xform.T,u_rhophiz[:,np.newaxis])

    # we are interested in u_x
    return u_xyz[0]



if __name__=="__main__":

    from matplotlib import pyplot as pl

    E=208e9
    nu=0.33
    a=5e-3
    tauext=100e6

    x=np.linspace(0,a,50)
    u_eval = np.array([ u(xval,0.0,a,tauext,E,nu) for xval in x])

    pl.figure()
    pl.plot(x*1e3,u_eval*1e6,'-')
    pl.show()
    pass
