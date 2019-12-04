import sys
import csv
import ast
import copy
import posixpath
import numpy as np

from matplotlib import pyplot as pl

import limatix.timestamp

from limatix import dc_value
from limatix import xmldoc
from limatix.dc_value import numericunitsvalue as numericunitsv
from limatix.dc_value import hrefvalue as hrefv
from limatix.dc_value import xmltreevalue as xmltreev


from crackclosuresim2 import inverse_closure,solve_normalstress
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2 import perform_inverse_closure,save_closurestress
from crackclosuresim2 import crack_model_normal_by_name
from crackclosuresim2 import crack_model_shear_by_name

# This processtrak step is used in vibrosim to evaluate crack closure
# state from crack tip positions given in an XML element.

# The crack closure state is given as four arrays interpreted as text
# within XML elements of the experiment log,
# e.g:
#   <dc:reff_side1 dcv:units="m" dcv:arraystorageorder="C">
#     <dcv:arrayshape>9</dcv:arrayshape>
#     <dcv:arraydata>
#       .5e-3 .7e-3 .9e-3 1.05e-3 1.2e-3 1.33e-3 1.45e-3 1.56e-3 1.66e-3
#     </dcv:arraydata>
#   </dc:reff_side1>
#   <dc:seff_side1 dcv:units="Pa" dcv:arraystorageorder="C">
#   <dcv:arrayshape>9</dcv:arrayshape>
#   <dcv:arraydata>
#     0.0 50e6 100e6 150e6 200e6 250e6 300e6 350e6 400e6
#   </dcv:arraydata>
# </dc:seff_side1>
#
# <dc:reff_side2 dcv:units="m" dcv:arraystorageorder="C">
#   <dcv:arrayshape>9</dcv:arrayshape>
#   <dcv:arraydata>
#     .5e-3 .7e-3 .9e-3 1.05e-3 1.2e-3 1.33e-3 1.45e-3 1.56e-3 1.66e-3
#   </dcv:arraydata>
# </dc:reff_side2>
# <dc:seff_side2 dcv:units="Pa" dcv:arraystorageorder="C">
#   <dcv:arrayshape>9</dcv:arrayshape>
#   <dcv:arraydata>
#     0.0 50e6 100e6 150e6 200e6 250e6 300e6 350e6 400e6
#   </dcv:arraydata>
# </dc:seff_side2>
#
# The reff (effective tip radius) values are given in meters and the
# seff (corresponding normal stress) values are given in Pascals.
# The radius values should be listed in increasing order. The last
# radius value on each side (side1 - left or side2 - right) should
# correspond to the length of that side of the crack. 

def run(_xmldoc,_element,
        _dest_href,
        _inputfilename,
        dc_measnum_int,
        dc_YoungsModulus_numericunits,
        dc_PoissonsRatio_float,
        dc_YieldStrength_numericunits,
        dc_reff_side1_array, # NOTE: arrayvalue class is not unit-aware!
        dc_seff_side1_array, 
        dc_reff_side2_array,
        dc_seff_side2_array,

        dc_crack_model_normal_str="Tada_ModeI_CircularCrack_along_midline",
        dc_crack_model_shear_str="Fabrikant_ModeII_CircularCrack_along_midline",
        
        dx=5e-6,
        dc_symmetric_cod_bool=True,
        debug_bool=False):


    crack_model_normal = crack_model_normal_by_name(dc_crack_model_normal_str,YoungsModulus,PoissonsRatio)
    crack_model_shear = crack_model_shear_by_name(dc_crack_model_shear_str,YoungsModulus,PoissonsRatio)
                                                    


    reff_side1 = dc_reff_side1_array.value()
    seff_side1 = dc_seff_side1_array.value()

    reff_side2 = dc_reff_side2_array.value()
    seff_side2 = dc_seff_side2_array.value()


    # Fully open crack lengths for left and right side
    aside1 = np.max(reff_side1) 
    aside2 = np.max(reff_side2)

    # Desired approximate step size for calculations
    approximate_xstep=25e-6 # 25um

    num_boundary_steps=np.floor((max(aside1,aside2)+approximate_xstep)/approximate_xstep)
    xmax = num_boundary_steps*approximate_xstep  # Maximum position from center to calculate to;
    # should exceed half-crack lengths 

    numsteps = num_boundary_steps-1
    xstep = (xmax)/(numsteps) # Actual step size so that xmax is a perfect multiple of this number

    x_bnd = xstep*np.arange(num_boundary_steps) # Position of element boundaries
    xrange = (x_bnd[1:] + x_bnd[:-1])/2.0 # Position of element centers

    # Determine closure stress field from observed crack length data
    closure_stress_side1=inverse_closure(reff_side1,seff_side1,xrange,x_bnd,xstep,aside1,sigma_yield,crack_model_normal,verbose=verbose)
    
    closure_stress_side2=inverse_closure(reff_side2,seff_side2,xrange,x_bnd,xstep,aside2,sigma_yield,crack_model_normal,verbose=verbose)
    
    
    # Evaluate initial crack opening gaps from extrapolated tensile closure field
    crack_initial_opening_side1 = crackopening_from_tensile_closure(xrange,x_bnd,closure_stress_side1,xstep,aside1,sigma_yield,crack_model_normal)
    
    crack_initial_opening_side2 = crackopening_from_tensile_closure(xrange,x_bnd,closure_stress_side2,xstep,aside2,sigma_yield,crack_model_normal)
    

    # Plot the evaluated closure state (side1)
    pl.figure()
    pl.plot(xrange[xrange < aside1]*1e3,closure_stress_side1[xrange < aside1]/1e6,'-',
            reff_side1*1e3,seff_side1/1e6,'x')
    for observcnt in range(len(reff_side1)):        
        (effective_length, sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(xrange,x_bnd,closure_stress_side1,xstep,seff_side1[observcnt],aside1,sigma_yield,crack_model_normal)
        pl.plot(effective_length*1e3,seff_side1[observcnt]/1e6,'.')
        pass
    pl.grid(True)
    pl.legend(('Closure stress field','Observed crack tip posn','Recon. crack tip posn'),loc="best")
    pl.xlabel('Radius from crack center (mm)')
    pl.ylabel('Stress (MPa)')
    pl.title('Crack closure state (side1)')

    closure_state_side1_href = hrefv(quote(dc_measident_str+"_closurestate_side1.png"),dc_dest_href)
    pl.savefig(closure_state_side1_href.getpath(),dpi=300)

        # Plot the evaluated closure state (side2)
    pl.figure()
    pl.plot(xrange[xrange < aside2]*1e3,closure_stress_side2[xrange < aside2]/1e6,'-',
            reff_side2*1e3,seff_side2/1e6,'x')
    for observcnt in range(len(reff_side2)):        
        (effective_length, sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(xrange,x_bnd,closure_stress_side2,xstep,seff_side2[observcnt],aside2,sigma_yield,crack_model_normal)
        pl.plot(effective_length*1e3,seff_side2[observcnt]/1e6,'.')
        pass
    pl.grid(True)
    pl.legend(('Closure stress field','Observed crack tip posn','Recon. crack tip posn'),loc="best")
    pl.xlabel('Radius from crack center (mm)')
    pl.ylabel('Stress (MPa)')
    pl.title('Crack closure state (side 2)')
    
    closure_state_side2_href = hrefv(quote(dc_measident_str+"_closurestate_side2.png"),dc_dest_href)
    pl.savefig(closure_state_side2_href.getpath(),dpi=300)



    closurestate_side1_href = hrefv(quote(dc_measident_str+"_closurestate_side1.csv"),dc_dest_href)
    save_closurestress(closurestate_side1_href.getpath(),x,closure_stress_side1,a_side1,crackopening=crack_initial_opening_side1)
    
    closurestate_side2_href = hrefv(quote(dc_measident_str+"_closurestate_side2.csv"),dc_dest_href)
    save_closurestress(closurestate_side2_href.getpath(),x,closure_stress_side2,a_side2,crackopening=crack_initial_opening_side2)
    

    ret = {
        "dc:closureplot_side1": closure_state_side1_href,
        "dc:closureplot_side2": closure_state_side2_href,
        "dc:closurestate_side1": closurestate_side1_href,
        "dc:closurestate_side2": closurestate_side2_href,
        "dc:a_side1": numericunitsv(a_side1,"m"),
        "dc:a_side2": numericunitsv(a_side2,"m"),
    }

    return ret
