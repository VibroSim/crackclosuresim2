import sys
import csv
import ast
import copy
import posixpath

try:
    # py2.x
    from urllib import pathname2url
    from urllib import url2pathname
    from urllib import quote
    from urllib import unquote
    pass
except ImportError:
    # py3.x
    from urllib.request import pathname2url
    from urllib.request import url2pathname
    from urllib.parse import quote
    from urllib.parse import unquote
    pass

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
from crackclosuresim2 import crackopening_from_tensile_closure
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
        dc_dest_href,
        dc_measident_str,
        dc_YoungsModulus_numericunits,
        dc_PoissonsRatio_float,
        dc_YieldStrength_numericunits,
        dc_crack_type_side1_str,
        dc_crack_type_side2_str,
        dc_reff_side1_array=None, # NOTE: arrayvalue class is not unit-aware!
        dc_seff_side1_array=None, 
        dc_reff_side2_array=None,
        dc_seff_side2_array=None,
        dc_symmetric_cod_bool=None, # Should be True for a surface or tunnel crack, False for an edge crack.
        dc_approximate_xstep_numericunits=None,
        dc_crack_model_normal_str="Tada_ModeI_CircularCrack_along_midline",
        dc_crack_model_shear_str="Fabrikant_ModeII_CircularCrack_along_midline",
        dx=5e-6):
    """ NOTE: Returns closure state of each side (unless crack_type is none).
    Also sets a_side1 and a_side2 crack length elements """

    verbose=False

    if dc_crack_model_normal_str=="ModeI_throughcrack_CODformula" and dc_symmetric_cod_bool is None:
        raise ValueError("dc:symmetric_cod must be set to 'true' or 'false' when using ModeI_throughcrack_CODformula crack model")
    elif dc_crack_model_normal_str=="Tada_ModeI_CircularCrack_along_midline" and dc_symmetric_cod_bool is None:
        dc_symmetric_cod_bool=True # Tada crack inherently symmetric
        pass

    if dc_crack_model_shear_str=="ModeII_throughcrack_CSDformula" and dc_symmetric_cod_bool is None:
        raise ValueError("dc:symmetric_cod must be set to 'true' or 'false' when using ModeII_throughcrack_CSDformula crack model")
    elif dc_crack_model_shear_str=="Fabrikant_ModeII_CircularCrack_along_midline" and dc_symmetric_cod_bool is None:
        # Fabrikant model inherently symmetric also
        dc_symmetric_cod_bool=True
        pass


    crack_model_normal = crack_model_normal_by_name(dc_crack_model_normal_str,dc_YoungsModulus_numericunits.value("Pa"),dc_PoissonsRatio_float,dc_symmetric_cod_bool)
    crack_model_shear = crack_model_shear_by_name(dc_crack_model_shear_str,dc_YoungsModulus_numericunits.value("Pa"),dc_PoissonsRatio_float,dc_symmetric_cod_bool)
    
    sigma_yield = dc_YieldStrength_numericunits.value("Pa")

    a_side1=0.0
    a_side2=0.0

    if dc_crack_type_side1_str.lower() != "none":
        if dc_reff_side1_array is None or dc_seff_side1_array is None:
            raise ValueError("dc:reff_side1 and dc_seff_side1 must be specified since dc:crack_type_side1 is \"%s\" and not \"none\"." % (dc_crack_type_side1_str))


        reff_side1 = dc_reff_side1_array.value()
        seff_side1 = dc_seff_side1_array.value()
        # Fully open crack length 
        a_side1 = np.max(reff_side1) 
        pass

    if dc_crack_type_side2_str.lower() != "none":
        if dc_reff_side2_array is None or dc_seff_side2_array is None:
            raise ValueError("dc:reff_side2 and dc:seff_side2 must be specified since dc:crack_type_side2 is \"%s\" and not \"none\"." % (dc_crack_type_side2_str))

        reff_side2 = dc_reff_side2_array.value()
        seff_side2 = dc_seff_side2_array.value()
        # Fully open crack length
        a_side2 = np.max(reff_side2)
        pass

    # Desired approximate step size for calculations
    if dc_approximate_xstep_numericunits is not None:
        approximate_xstep=dc_approximate_xstep_numericunits.value("m")
        pass
    else:
        approximate_xstep=25e-6 # 25um
        pass

    num_boundary_steps=np.floor((max(a_side1,a_side2)+approximate_xstep)/approximate_xstep)
    xmax = num_boundary_steps*approximate_xstep  # Maximum position from center to calculate to;
    # should exceed half-crack lengths 

    numsteps = num_boundary_steps-1
    xstep = (xmax)/(numsteps) # Actual step size so that xmax is a perfect multiple of this number

    x_bnd = xstep*np.arange(num_boundary_steps) # Position of element boundaries
    xrange = (x_bnd[1:] + x_bnd[:-1])/2.0 # Position of element centers


    if dc_crack_type_side1_str.lower() != "none":
        # Determine closure stress field from observed crack length data
        closure_stress_side1=inverse_closure(reff_side1,seff_side1,xrange,x_bnd,xstep,a_side1,sigma_yield,crack_model_normal,verbose=verbose)
        # Evaluate initial crack opening gaps from extrapolated tensile closure field
        crack_initial_opening_side1 = crackopening_from_tensile_closure(xrange,x_bnd,closure_stress_side1,xstep,a_side1,sigma_yield,crack_model_normal)
        pass

    
    if dc_crack_type_side2_str.lower() != "none":
        # Determine closure stress field from observed crack length data
        closure_stress_side2=inverse_closure(reff_side2,seff_side2,xrange,x_bnd,xstep,a_side2,sigma_yield,crack_model_normal,verbose=verbose)
    
        # Evaluate initial crack opening gaps from extrapolated tensile closure field
        crack_initial_opening_side2 = crackopening_from_tensile_closure(xrange,x_bnd,closure_stress_side2,xstep,a_side2,sigma_yield,crack_model_normal)
        pass

    ret={} # return dictionary

    if dc_crack_type_side1_str.lower() != "none":
        # Plot the evaluated closure state (side1)
        pl.figure()
        pl.plot(xrange[xrange < a_side1]*1e3,closure_stress_side1[xrange < a_side1]/1e6,'-',
                reff_side1*1e3,seff_side1/1e6,'x')
        for observcnt in range(len(reff_side1)):        
            (effective_length, sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(xrange,x_bnd,closure_stress_side1,xstep,seff_side1[observcnt],a_side1,sigma_yield,crack_model_normal)
            pl.plot(effective_length*1e3,seff_side1[observcnt]/1e6,'.')
            pass
        pl.grid(True)
        pl.legend(('Closure stress field','Observed crack tip posn','Recon. crack tip posn'),loc="best")
        pl.xlabel('Radius from crack center (mm)')
        pl.ylabel('Stress (MPa)')
        pl.title('Crack closure state (side1)')
        
        closureplot_side1_href = hrefv(quote(dc_measident_str+"_closurestate_side1.png"),dc_dest_href)
        pl.savefig(closureplot_side1_href.getpath(),dpi=300)
        ret["dc:closureplot_side1"] = closureplot_side1_href
        pass



    if dc_crack_type_side2_str.lower() != "none":
        # Plot the evaluated closure state (side2)
        pl.figure()
        pl.plot(xrange[xrange < a_side2]*1e3,closure_stress_side2[xrange < a_side2]/1e6,'-',
                reff_side2*1e3,seff_side2/1e6,'x')
        for observcnt in range(len(reff_side2)):        
            (effective_length, sigma, tensile_displ, dsigmaext_dxt) = solve_normalstress(xrange,x_bnd,closure_stress_side2,xstep,seff_side2[observcnt],a_side2,sigma_yield,crack_model_normal)
            pl.plot(effective_length*1e3,seff_side2[observcnt]/1e6,'.')
            pass
        pl.grid(True)
        pl.legend(('Closure stress field','Observed crack tip posn','Recon. crack tip posn'),loc="best")
        pl.xlabel('Radius from crack center (mm)')
        pl.ylabel('Stress (MPa)')
        pl.title('Crack closure state (side 2)')
        
        closureplot_side2_href = hrefv(quote(dc_measident_str+"_closurestate_side2.png"),dc_dest_href)
        pl.savefig(closureplot_side2_href.getpath(),dpi=300)
        ret["dc:closureplot_side2"]= closureplot_side2_href
        pass


    if dc_crack_type_side1_str.lower() != "none":
        closurestate_side1_href = hrefv(quote(dc_measident_str+"_closurestate_side1.csv"),dc_dest_href)
        save_closurestress(closurestate_side1_href.getpath(),xrange,closure_stress_side1,a_side1,crackopening=crack_initial_opening_side1)

        ret["dc:closurestate_side1"]=closurestate_side1_href
        ret["dc:a_side1"] = numericunitsv(a_side1,"m")

        pass

    if dc_crack_type_side2_str.lower() != "none":
        closurestate_side2_href = hrefv(quote(dc_measident_str+"_closurestate_side2.csv"),dc_dest_href)
        save_closurestress(closurestate_side2_href.getpath(),xrange,closure_stress_side2,a_side2,crackopening=crack_initial_opening_side2)
        ret["dc:closurestate_side2"] = closurestate_side2_href
        ret["dc:a_side2"] = numericunitsv(a_side2,"m")

        pass



    return ret
