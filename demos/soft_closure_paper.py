import copy
import sys
import numpy as np
from numpy import sqrt,log,pi,cos,arctan
import scipy.optimize
import scipy as sp

import pandas as pd

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
    E = 117.66e9    #Plane stress Modulus of Elasticity
    Eeff=E
    sigma_yield = 1061e6
    tau_yield = sigma_yield/2.0 # limits stress concentration around singularity
    nu = 0.32   #Poisson's Ratio

    AFVT_018J_closure = pd.read_csv("0000-C18-AFVT-018J_optical_collect_optical_data_dic_closureprofile_closurestress_side1.csv")

    dx=AFVT_018J_closure["Crack radius (m)"].values[1]-AFVT_018J_closure["Crack radius (m)"].values[0]
    #x = np.concatenate((AFVT_018J_closure["Crack radius (m)"].values,np.array((AFVT_018J_closure["Crack radius (m)"].values[-1]+dx,))))
    x=AFVT_018J_closure["Crack radius (m)"].values
    # expand out position base to one sample beyond crack end

    a_input=.495e-3  # half-crack length (m)

    x_bnd = np.concatenate((x-dx/2.0,np.array((x[-1]+dx/2.0,x[-1]+3*dx/2.0))))
    if x_bnd[0] < 0.0:
        x_bnd[0]=0.0
        pass

    #sigma_closure = np.concatenate((AFVT_018J_closure["Closure stress (Pa)"].values,np.array((0.0,))))
    sigma_closure = AFVT_018J_closure["Closure stress (Pa)"].values

    # Remove drop in sigma_closure after peak, as this is probably non-physical
    sc_maxidx=np.argmax(sigma_closure)
    sigma_closure[(sc_maxidx+1):]=sigma_closure[sc_maxidx]
    
    # in this model, we require crack length to line up
    # on an element boundary, so we will round it to
    # the nearest boundary below when we store a
        
    
    #fine_refinement=int(4)
    fine_refinement=int(1)

    # 1/Lm has units of m^(3/2)/Pascal
    # Lm has units of Pa/m^(3/2)
    Lm = 10e6/(100e-9**(3.0/2.0))  # rough order of magnitude guess

    # Lm can also be calculated with angled_friction_model.asperity_stiffness()
    # Lm = asperity_stiffness(msqrtR,E,nu,angular_stddev)


    
    # Closure state (function of position; positive compression)
    # Use hard closure model to solve for closure state
    crack_model = ModeI_throughcrack_CODformula(Eeff)
    #crack_model = Tada_ModeI_CircularCrack_along_midline(E,nu)
    

    scp = sc_params.fromcrackgeom(crack_model,x_bnd[-1],x.shape[0]+1,a_input,fine_refinement,Lm)
    

    crack_initial_opening = crackopening_from_tensile_closure(x,x_bnd,sigma_closure,dx,a_input,sigma_yield,crack_model)
    
    sigma_closure_softmodel = sigma_closure.copy()
    sigma_closure_softmodel[sigma_closure_softmodel < 0.0]=0.0
    scp.initialize_contact(sigma_closure_softmodel,crack_initial_opening)
    
    pl.figure()
    pl.plot(x*1e3,sigma_closure/1e6,'-')
    pl.title("Closure stress distribution")
    pl.xlabel("Position (mm)")
    pl.ylabel("Compressive stress (MPa)")
    pl.grid()
    pl.savefig('/tmp/closurestress.png',dpi=300)
    
    du_da=np.zeros(scp.x.shape[0]+1,dtype='d')

    soft_closure_plots(scp,du_da,titleprefix="Initial: ")

    sigma_ext_compressive=-50e6
    

    (du_da_compressive,contact_stress_compressive,displacement_compressive) = calc_contact(scp,sigma_ext_compressive)

    (sigmacontact_plot_compressive,duda_plot_compressive,displacement_plot_compressive) = soft_closure_plots(scp,du_da_compressive,titleprefix="Compressive: ")

    pl.figure(sigmacontact_plot_compressive.number)
    pl.savefig('/tmp/compressive_sigmacontact.png',dpi=300)

    pl.figure(duda_plot_compressive.number)
    pl.savefig('/tmp/compressive_duda.png',dpi=300)

    pl.figure(displacement_plot_compressive.number)
    pl.savefig('/tmp/compressive_displacement.png',dpi=300)


    sigma_ext_tensile=50e6
    

    (du_da_tensile,contact_stress_tensile,displacement_tensile) = calc_contact(scp,sigma_ext_tensile)

    (sigmacontact_plot_tensile,duda_plot_tensile,displacement_plot_tensile) = soft_closure_plots(scp,du_da_tensile,titleprefix="Tensile: ")


    pl.figure(sigmacontact_plot_tensile.number)
    pl.savefig('/tmp/tensile_sigmacontact.png',dpi=300)

    pl.figure(duda_plot_tensile.number)
    pl.savefig('/tmp/tensile_duda.png',dpi=300)

    pl.figure(displacement_plot_tensile.number)
    pl.savefig('/tmp/tensile_displacement.png',dpi=300)

    # What if we do a second tensile step? 

    pl.show()
    pass
