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
#from crackclosuresim2.soft_closure import calc_du_da


# This example demonstrates applying a bias load
# and then incremental changes on top of the bias load
# and illustrates that the results are identical to
# applying the same net loads directly. 


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

    
    # Closure state (function of position; positive compression)
    # Use hard closure model to solve for closure state
    crack_model = ModeI_throughcrack_CODformula(Eeff)
    #crack_model = Tada_ModeI_CircularCrack_along_midline(E,nu)
    

    scp = sc_params.fromcrackgeom(crack_model,xmax,xsteps,a_input,fine_refinement,Hm)
    

    observed_reff = np.array([  0.5e-3,  1e-3, 1.5e-3,
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

    
    # temporary zero out sigma_closure
    #sigma_closure[:]=0.0
    
    # In the soft closure model, sigma_closure can't be negativej
    # (use crack_initial_opening values instead in that domain)
    sigma_closure[sigma_closure < 0.0]=0.0
    
    scp.initialize_contact(sigma_closure,crack_initial_opening)
    

    du_da=np.zeros(scp.x.shape[0],dtype='d')

    soft_closure_plots(scp,du_da,titleprefix="Initial: ")


    sigma_ext_bias = 80e6
    

    sigma_ext_compressive=sigma_ext_bias-50e6
    

    (du_da_compressive,contact_stress_compressive,displacement_compressive) = calc_contact(scp,sigma_ext_compressive)

    soft_closure_plots(scp,du_da_compressive,titleprefix="Compressive: ")


    sigma_ext_tensile=sigma_ext_bias+50e6
    

    (du_da_tensile,contact_stress_tensile,displacement_tensile) = calc_contact(scp,sigma_ext_tensile)

    soft_closure_plots(scp,du_da_tensile,titleprefix="Tensile: ")


    # Demonstrate applying bias load to crack with scp.setcrackstate()
    
    # Step #1: Evaluate bias condition
    (du_da_bias,contact_stress_bias,displacement_bias) = calc_contact(scp,sigma_ext_bias)

    soft_closure_plots(scp,du_da_bias,titleprefix="Bias: ")

    # Step #2: Assign the new crack state. The applied 'crack_initial_opening'
    # should have (contact_stress_bias/scp.Hm)**(2.0/3.0) added in because
    # the sigmacontact_from_displacement() calculation routine starts
    # by subtracting out (sigma_closure/scp.Hm)**(2.0/3.0) where
    # the new sigma_closure will be contact_stress_bias.     
    scp.setcrackstate(contact_stress_bias,displacement_bias + (contact_stress_bias/scp.Hm)**(2.0/3.0))

    du_da_bias_applied=np.zeros(scp.x.shape[0],dtype='d')

    soft_closure_plots(scp,du_da_bias_applied,titleprefix="Bias applied: ")


    # Now recalculate the previous figures but relative to the biased
    # configuration (optimal states should be nearly identical)
    sigma_ext_compressive_biased=-50e6
    

    (du_da_compressive_biased,contact_stress_compressive_biased,displacement_compressive_biased) = calc_contact(scp,sigma_ext_compressive_biased)

    soft_closure_plots(scp,du_da_compressive_biased,titleprefix="Compressive with bias: ")


    sigma_ext_tensile_biased=50e6
    

    (du_da_tensile_biased,contact_stress_tensile_biased,displacement_tensile_biased) = calc_contact(scp,sigma_ext_tensile_biased)

    soft_closure_plots(scp,du_da_tensile_biased,titleprefix="Tensile with bias: ")

    
    pl.show()
    pass
