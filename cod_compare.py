import numpy as np
from matplotlib import pyplot as pl
from scipy.integrate import quad

if __name__=="__main__":
    from matplotlib import pyplot as pl
    pl.rc('text', usetex=True) # Support greek letters in plot legend
    pass

# What happens when we nondimensionalize crack opening displacement?
# uy = 2(sigma/Eeff)*sqrt(a^2-x^2)
#  Let xbar=x/a -> x=xbar a
# uy = 2(sigma/Eeff)*sqrt(a^2-(xbar^2)(a^2))
# uy = 2(sigma/Eeff)*a*sqrt(1-xbar^2)
# ... we get a dimensionless functional form, e.g. sqrt(1-xbar^2)
# Let the functional form be represented as f(xbar).
# Now uy = 2(sigma/Eeff)*a*f(xbar), where in this case f(xbar)=sqrt(1-xbar^2)

# There must also be an analogous dimensionless functional form
# for the surface COD of a semicircular crack. Since elasticity
# is scale independent, it is just a different f(xbar).
#
# (for an elliptical crack, f would also have an aspect ratio
# parameter, but let us ignore that -- or hold it constant --
# for now)
#
# Likewise the stress intensity factor for a
# semicircular surface crack must have the form
#  K = sigma sqrt(pi*a*beta)
#
# OK. How do
# Weight function = (Eeff/2K) du/da
# h(x,a) = Eeff/(2*sigma*sqrt(pi*a*beta))*du/da
# h(x,a) = Eeff/(2*sigma*sqrt(pi*a*beta))*2*(sigma/Eeff)*(d/da)(a*sqrt(1-xbar^2))
# h(x,a) = 1/(sqrt(pi*a*beta))*(d/da)(a*f(xbar))
# let hbar(x,a) = h(x,a)*sqrt(a)
# hbar(x,a)/sqrt(a) = 1/(sqrt(pi*a*beta))*(d/da)(a*f(xbar))
# hbar(x,a) = 1/(sqrt(pi*beta))*(d/da)(a*f(xbar))
#  ... what about d/da(a*f(xbar)) ? 
#  ... product rule: = 1*f(xbar) + a*(d/da)(f(xbar))
#                    = 1*f(xbar) + a*(d/dxbar)(f(xbar)) * dxbar/da
# since xbar = x/a = x*a^-1, dxbar/da = -xa^-2
#   denote (d/dxbar)(f(xbar)) as f'(xbar)
# now derivative  factor = 1*f(xbar) + a*f'(xbar)*(-x/a^2)
#                        = f(xbar) - (x/a) f'(xbar)
#                        = f(xbar) - xbar*f'(xbar)
# So hbar(x,a) = 1/(sqrt(pi*beta))*[ f(xbar) - xbar*f'(xbar) ]
#
# Therefore if we can get (a) the surface COD function and its derivative
# and (b) the stress intensity factor fudge factor beta,
# we have a weight function and we are done!
#
# ... applies to both normal and shear loading.




# OLD formula valid near crack tip
#Kappa = (3.0-nu)/(1.0+nu)
#
#KI = sigma*np.sqrt(np.pi*(a))
#theta = np.pi
#u = (KI/(2.0*E))*(np.sqrt((a-x)/(2.0*np.pi)))*((1.0+nu)* 
#                                                (((2.0*Kappa+1.0)*(np.sin(theta/2.0)))-np.sin(3.0*theta/2.0)))
#u = (sigma*sqrt(pi*a)/(2.0*E))*(np.sqrt((a-x)/(2.0*np.pi)))*((1.0+nu)* 
#                                                (((2.0*Kappa+1.0)*(np.sin(theta/2.0)))-np.sin(3.0*theta/2.0)))

# let v=a/x -> a=vx
#u = x*(sigma*sqrt(pi*v)/(2.0*E))*(np.sqrt((v-1)/(2.0*np.pi)))*((1.0+nu)* 
#                                                (2.0*Kappa+2.0)
#u = x*(sigma*sqrt(pi*v)/(2.0*E))*(np.sqrt((v-1)/(2.0*np.pi)))*((1.0+nu)* 
#                                                (2.0*Kappa+2.0)
# u*Eeff/(sigma*x) = (sqrt(pi*v)/(2.0))*(np.sqrt((v-1)/(2.0*np.pi)))*((1.0+nu)* 
#                                                (2.0*Kappa+2.0)





# Elliptical COD formula
# Based on Anderson, eq. A2.43
# uy = 2(sigma/Eeff)*sqrt(a^2-x^2)
# uy = 2(sigma/Eeff)*sqrt((a+x)(a-x))
# uy = 2(sigma/Eeff)*x*sqrt((a_ov_x+1)(a_ov_x-1))
# uy*Eeff/(sigma*x) = 2*sqrt((a_ov_x+1)(a_ov_x-1))

# Integration of published weight function:
# m(x,a) = (Eeff/(2K)) du/da
# uy = (2/Eeff) integral_x^a K*m(x,a) da  where K=sigma*sqrt(pi*a)
# and m=(1/sqrt(pi*a))*sqrt((a+x)/(a-x)) # Anderson, example 2.6 with x replaced by x+a to move the origin to the center of the crack 
# uy = (2sigma/Eeff) * integral_x^a sqrt((a+x)/(a-x)) da
# let v = a/x -> dv=da/x -> da=x*dv
# uy = (2sigma/Eeff) * integral_1^(a/x) sqrt((vx+x)/(vx-x)) x dv
# uy = (2sigma/Eeff) * x integral_1^(a_ov_x) sqrt((v+1)/(v-1)) dv
# uy*Eeff/(sigma*x) = 2*integral_1^(a_ov_x) sqrt((v+1)/(v-1)) dv

# Integration of Holland weight function 
# As above but m = (1/sqrt(pi)) * (sqrt(a)/(sqrt(a+x)*sqrt(a-x)))
# uy = (2sigma/Eeff) integral_x^a (a/(sqrt(a+x)*sqrt(a-x))) da  
# let v = a/x -> dv=da/x -> da=x*dv
# uy = (2sigma/Eeff) integral_1^(a/x) (vx/(sqrt(vx+x)*sqrt(vx-x))) x*dv 
# uy = (2sigma/Eeff)x integral_1^(a/x) (v/(sqrt(v+1)*sqrt(v-1))) dv 
# uy*Eeff/(sigma*x) = 2integral_1^(a/x) (v/(sqrt(v+1)*sqrt(v-1))) dv 


a_ov_x=np.linspace(1.0,3.0,100)

eps=1e-6
nu=0.22 # used only by neartip_formula
Kappa = (3.0-nu)/(1.0+nu)

neartip_formula = (np.sqrt(np.pi*a_ov_x)/(2.0))*(np.sqrt((a_ov_x-1)/(2.0*np.pi)))*((1.0+nu)*(2.0*Kappa+2.0))

# Evaluate uy*Eeff/(sigma*x) 
elliptical_formula = 2.0*np.sqrt((a_ov_x+1.0)*(a_ov_x-1.0))
                                                                             
# Evaluate uy*Eeff/(sigma*x) 
tb_weightfun_formula = np.vectorize(lambda a_ov_x : quad(lambda v: 2.0*np.sqrt((v+1.0)/(v-1.0)),1.0+eps,a_ov_x)[0])(a_ov_x)

h_weightfun_formula = np.vectorize(lambda a_ov_x: quad(lambda v: 2.0*(v/(np.sqrt(v+1)*np.sqrt(v-1))),1.0+eps,a_ov_x)[0])(a_ov_x)


pl.figure(1)
pl.clf()
pl.plot(a_ov_x,neartip_formula,'-',
        a_ov_x,elliptical_formula,'-',
        a_ov_x,tb_weightfun_formula,'-',
        a_ov_x,h_weightfun_formula,'--')
pl.xlabel('a/x')
pl.ylabel('COD*$E_{\mbox{eff}}$/$(x\sigma_{\mbox{ext}})$')
pl.legend(('Near-tip COD formula','Elliptical COD formula','Calculated from textbook weightfunction','Calculated from Holland weightfunction'))
pl.grid()
pl.savefig('/tmp/weightfunction_COD.png',dpi=300)
