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
from crackclosuresim2.soft_closure import calc_contact
from crackclosuresim2.soft_closure import soft_closure_plots
from crackclosuresim2.soft_closure import sigmacontact_from_displacement
from crackclosuresim2.soft_closure import sigmacontact_from_stress
#from angled_friction_model.asperity_stiffness import asperity_stiffness


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

    # Hm can also be calculated with angled_friction_model.asperity_stiffness()
    # Hm = asperity_stiffness(msqrtR,E,nu,angular_stddev)


    
    # Closure state (function of position; positive compression)
    # Use hard closure model to solve for closure state
    crack_model = ModeI_throughcrack_CODformula(Eeff)
    #crack_model = Tada_ModeI_CircularCrack_along_midline(E,nu)
    

    scp = sc_params.fromcrackgeom(crack_model,xmax,xsteps,a_input,fine_refinement,Hm)
    

    observed_reff = np.array([  0.0e-3,  1e-3, 1.5e-3,
                                1.9e-3,
                                scp.a  ],dtype='d')
    observed_seff = np.array([ 1e6, 15e6, 30e6,
                               150e6,
                               170e6  ],dtype='d')
    
    sigma_closure = inverse_closure(observed_reff,
                                    observed_seff,
                                    scp.x,scp.x_bnd,scp.dx,scp.a,sigma_yield,
                                    crack_model)

    crack_initial_opening = crackopening_from_tensile_closure(scp.x,scp.x_bnd,sigma_closure,scp.dx,scp.a,sigma_yield,crack_model)

    
    
    # In the soft closure model, sigma_closure can't be negativej
    # (use crack_initial_opening values instead in that domain)
    sigma_closure[sigma_closure < 0.0]=0.0
    


    scp.initialize_contact(sigma_closure,crack_initial_opening)
    

    du_da=np.zeros(scp.x.shape[0],dtype='d')

    soft_closure_plots(scp,du_da,titleprefix="Initial: ")

    sigma_ext_compressive=-50e6
    

    (du_da_compressive,contact_stress_compressive,displacement_compressive) = calc_contact(scp,sigma_ext_compressive)

    soft_closure_plots(scp,du_da_compressive,titleprefix="Compressive: ")


    sigma_ext_tensile=50e6
    

    (du_da_tensile,contact_stress_tensile,displacement_tensile) = calc_contact(scp,sigma_ext_tensile)

    soft_closure_plots(scp,du_da_tensile,titleprefix="Tensile: ")

    pl.show()
    pass
