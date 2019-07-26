import copy
import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize
import scipy as sp

from matplotlib import pylab as pl
#pl.rc('text', usetex=True) # Support greek letters in plot legend

from crackclosuresim2 import inverse_closure,crackopening_from_tensile_closure
from crackclosuresim2 import ModeI_throughcrack_CODformula
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline

from crackclosuresim2.soft_closure import sc_params
from crackclosuresim2.soft_closure import tip_field_integral
from crackclosuresim2.soft_closure import calc_contact
from crackclosuresim2.soft_closure import soft_closure_plots
from crackclosuresim2.soft_closure import sigmacontact_from_displacement
from crackclosuresim2.soft_closure import sigmacontact_from_stress
from crackclosuresim2.soft_closure import calc_du_da


# TODO:
#  * param[0] is 0... why?
#  * sigmacontact from stress goes negative (tensile?) ... why?
#  * du/da shows two concentrated spots and goes negative (?)
#   ... shouldn't be possible...

if __name__=="__main__":
    #####INPUT VALUES
    E = 200e9    #Plane stress Modulus of Elasticity
    Eeff=E
    sigma_yield = 400e6
    tau_yield = sigma_yield/2.0 # limits stress concentration around singularity
    nu = 0.33    #Poisson's Ratio
    
    
    a_input=2.0e-3  # half-crack length (m)
    
    # in this model, we require crack length to line up
    # on an element boundary, so we will round it to
    # the nearest boundary below when we store a
    
    xmax = 5e-3 # as far out in x as we are calculating (m)
    xsteps = 200
    
    
    #fine_refinement=int(4)
    fine_refinement=int(1)

    # 1/Hm has units of m^(3/2)/Pascal
    # Hm has units of Pa/m^(3/2)
    Hm = 10e6/(100e-9**(3.0/2.0))  # rough order of magnitude guess

    scp = sc_params.fromcrackgeom(E,xmax,xsteps,a_input,fine_refinement,Hm)

    
    # Closure state (function of position; positive compression)
    # Use hard closure model to solve for closure state
    #crack_model = ModeI_throughcrack_CODformula(Eeff)
    crack_model = Tada_ModeI_CircularCrack_along_midline(E,nu)
    
    

    # !!!*** NOTE: inverse_closure() fails if first observed_reff element is 0
    # !!!*** Should troubleshoot this.
    observed_reff = np.array([  0.5e-3,  1e-3, 1.5e-3, scp.a  ],dtype='d')
    observed_seff = np.array([ 1e6, 15e6, 30e6, 150e6  ],dtype='d')
    
    sigma_closure = inverse_closure(observed_reff,
                                    observed_seff,
                                    scp.x,scp.x_bnd,scp.dx,scp.a,sigma_yield,
                                    crack_model)

    crack_initial_opening = crackopening_from_tensile_closure(scp.x,scp.x_bnd,sigma_closure,scp.dx,scp.a,sigma_yield,crack_model)

    # ***!!! NOTE: soft closure portion does not yet use crack_model !!!***
    
    # temporary zero out sigma_closure
    #sigma_closure[:]=0.0
    
    # In the soft closure model, sigma_closure can't be negativej
    # (use crack_initial_opening values instead in that domain)
    sigma_closure[sigma_closure < 0.0]=0.0
    


    scp.setcrackstate(sigma_closure,crack_initial_opening)
    


    sigma_ext=50e6
    

    (param,contact_stress,displacement,dsigmaext_dxt_hardcontact) = calc_contact(scp,sigma_ext)

    soft_closure_plots(scp,param,dsigmaext_dxt_hardcontact)
    pl.show()
    pass
