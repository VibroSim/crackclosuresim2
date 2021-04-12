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


from crackclosuresim2 import inverse_closure,inverse_closure2,solve_normalstress
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2 import perform_inverse_closure,save_closurestress

# This processtrak step is used in evaluating
# crack closure state from DIC (digital image correlation)
# results. The DIC process yields a file with closure profile
# data indicating the effective crack tip position as a function
# of tensile load on the crack

def run(_xmldoc,_element,
        _dest_href,
        _inputfilename,
        dc_measnum_int,
        dc_closureprofile_href,
        dc_spcYoungsModulus_numericunits,
        dc_spcPoissonsRatio_numericunits,
        dc_spcYieldStrength_numericunits,
        dc_crackpath,
        dc_coordinatetransform,
        dc_specimen_str,
        dc_dest_href=None, # updated replacement for _dest_href
        dx=5e-6,
        dc_symmetric_cod_bool=True,  # No longer seems to be used... probably redundant with (currently hardwired) choice of crack_model
        dc_hascrackside1_bool=True,
        dc_hascrackside2_bool=True,
        dc_use_inverse_closure2_bool=False,
        dc_interpolate_closure_state_bool=True,
        dc_a_side1_numericunits=None,
        dc_a_side2_numericunits=None,        
        debug_bool=False):

    if dc_dest_href is not None:
        _dest_href=dc_dest_href
        pass
    
    E=dc_spcYoungsModulus_numericunits.value("Pa")
    nu = dc_spcPoissonsRatio_numericunits.value("unitless")   #Poisson's Ratio
    #sigma_yield = 400e6
    sigma_yield = dc_spcYieldStrength_numericunits.value("Pa")

    # (temporarily?) fixed parameters
    #dx=25e-6
    
    # We are working in stitched optical image coordinates, so we need
    # to subtract out xshift and yshift from the crack positions encoded
    # in the experiment log


    xshift = numericunitsv.fromxml(_xmldoc, _xmldoc.xpathsinglecontext(dc_coordinatetransform, 'dc:translation/dc:xtranslation')).value('m')
    yshift = numericunitsv.fromxml(_xmldoc, _xmldoc.xpathsinglecontext(dc_coordinatetransform, 'dc:translation/dc:ytranslation')).value('m')
    
    #keypoints = _opxmldoc.xpathcontext(dc_crackpath, 'dc:segment/dc:keypoint')

    crackstartx = numericunitsv.fromxml(_xmldoc,_xmldoc.xpathsinglecontext(dc_crackpath, 'dc:segment/dc:keypoint[1]/dc:xcoordinate')).value('m') - xshift
    crackstarty = numericunitsv.fromxml(_xmldoc,_xmldoc.xpathsinglecontext(dc_crackpath, 'dc:segment/dc:keypoint[1]/dc:ycoordinate')).value('m')

    crackendx = numericunitsv.fromxml(_xmldoc,_xmldoc.xpathsinglecontext(dc_crackpath, 'dc:segment/dc:keypoint[last()]/dc:xcoordinate')).value('m') - xshift
    crackendy = numericunitsv.fromxml(_xmldoc,_xmldoc.xpathsinglecontext(dc_crackpath, 'dc:segment/dc:keypoint[last()]/dc:ycoordinate')).value('m') 


    if dc_hascrackside1_bool and dc_hascrackside2_bool:
        CrackCenterCoords = ((crackstartx+crackendx)/2.0,(crackstarty+crackendy)/2.0)
        pass
    elif dc_hascrackside1_bool:
        CrackCenterCoords =(crackendx,crackendy)
        pass
    elif dc_hascrackside2_bool:
        CrackCenterCoords =(crackstartx,crackstarty)
        pass
    
    CrackCenterX = CrackCenterCoords[0]

    #a_side1 = (crackendx-crackstartx)/2.0 # half-crack length (m)
    #a_side2 = (crackendx-crackstartx)/2.0 # half-crack length (m)

    if not dc_use_inverse_closure2_bool:
        print("crackclosuresim2: eval_closure_state: WARNING: Using obsolete inverse_closure() routine instead of fixed inverse_closure2(). Add <dc:use_inverse_closure2>True</dc:use_inverse_closure2> or <prx:param name=\"dc_use_inverse_closure2\">True</prx:param> to fix")
        pass
    
    (x,x_bnd,a_side1,a_side2,sigma_closure_side1,sigma_closure_side2,side1fig,side2fig) = perform_inverse_closure(dc_closureprofile_href.getpath(),E,nu,sigma_yield,CrackCenterX,dx,dc_specimen_str,dc_hascrackside1_bool,dc_hascrackside2_bool,use_inverse_closure2=dc_use_inverse_closure2_bool)


    if dc_a_side1_numericunits is not None and dc_hascrackside1_bool:
        # Cross-check pre-existing a_side1 for backwards compatibility
        preexist_a_side1=dc_a_side1_numericunits.value("m")
        if a_side1 > preexist_a_side1+1e-6: # a_side1 may be legitimately shorter than physical length if we didn't open it up that far during DIC data collection
            raise ValueError("Crack length inconsistency > 1um on side1: %g vs %g" % (a_side1,preexist_a_side1))
        pass

    if dc_a_side2_numericunits is not None and dc_hascrackside2_bool:
        # Cross-check pre-existing a_side1 for backwards compatibility
        preexist_a_side2=dc_a_side2_numericunits.value("m")
        if a_side2 > preexist_a_side2+1e-6: # a_side2 may be legitimately shorter than physical length if we didn't open it up that far during DIC data collection
            raise ValueError("Crack length inconsistency > 1um on side2: %g vs %g" % (a_side2,preexist_a_side2))
        pass

    
    ret={}

    if sigma_closure_side1 is not None:
        closureplot_side1_href = hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side1.png",contexthref=_dest_href)
        pl.figure(side1fig.number)
        pl.savefig(closureplot_side1_href.getpath(),dpi=300,transparent=True)
        
        closurestress_side1_href=hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side1.csv",contexthref=_dest_href)
        save_closurestress(closurestress_side1_href.getpath(),x,sigma_closure_side1,a_side1)

        ret.update({
            "dc:closurestress_side1": closurestress_side1_href,
            "dc:closureplot_side1": closureplot_side1_href,
        })

        if dc_a_side1_numericunits is None:
            # Backwards compatibility: If dc:a_side1 not previously given, we assign it here
            ret.update({"dc:a_side1": numericunitsv(a_side1,"m")})
            pass
        

        pass

    if sigma_closure_side2 is not None:
        closureplot_side2_href = hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side2.png",contexthref=_dest_href)
        pl.figure(side2fig.number)
        pl.savefig(closureplot_side2_href.getpath(),dpi=300,transparent=True)

        closurestress_side2_href=hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side2.csv",contexthref=_dest_href)
        save_closurestress(closurestress_side2_href.getpath(),x,sigma_closure_side2,a_side2)
        
        ret.update({
            "dc:closurestress_side2": closurestress_side2_href,
            "dc:closureplot_side2": closureplot_side2_href,
        })
        if dc_a_side2_numericunits is None:
            # Backwards compatibility: If dc:a_side2 not previously given, we assign it here
            ret.update({"dc:a_side2": numericunitsv(a_side2,"m")})
            pass
        pass


    
    return ret
    
    
    #pl.show()

    
