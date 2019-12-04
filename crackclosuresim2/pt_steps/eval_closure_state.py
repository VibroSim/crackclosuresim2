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
        dx=5e-6,
        dc_symmetric_cod_bool=True,
        debug_bool=False):
    
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

    crackendx = numericunitsv.fromxml(_xmldoc,_xmldoc.xpathsinglecontext(dc_crackpath, 'dc:segment/dc:keypoint[last()]/dc:xcoordinate')).value('m') - xshift

    
    CrackCenterX = (crackstartx+crackendx)/2.0

    #a_side1 = (crackendx-crackstartx)/2.0 # half-crack length (m)
    #a_side2 = (crackendx-crackstartx)/2.0 # half-crack length (m)


    (x,x_bnd,a_side1,a_side2,sigma_closure_side1,sigma_closure_side2,side1fig,side2fig) = perform_inverse_closure(dc_closureprofile_href.getpath(),E,nu,sigma_yield,CrackCenterX,dx,dc_specimen_str)


    closureplot_side1_href = hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side1.png",contexthref=_dest_href)
    pl.figure(side1fig.number)
    pl.savefig(closureplot_side1_href.getpath(),dpi=300,transparent=True)

    closureplot_side2_href = hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side2.png",contexthref=_dest_href)
    pl.figure(side2fig.number)
    pl.savefig(closureplot_side2_href.getpath(),dpi=300,transparent=True)


    closurestress_side1_href=hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side1.csv",contexthref=_dest_href)
    save_closurestress(closurestress_side1_href.getpath(),x,sigma_closure_side1,a_side1)

    closurestress_side2_href=hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side2.csv",contexthref=_dest_href)
    save_closurestress(closurestress_side2_href.getpath(),x,sigma_closure_side2,a_side2)
    
    return {
        "dc:closurestress_side1": closurestress_side1_href,
        "dc:closureplot_side1": closureplot_side1_href,
        "dc:closurestress_side2": closurestress_side2_href,
        "dc:closureplot_side2": closureplot_side2_href,
        "dc:a_side1": numericunitsv(a_side1,"m"),
        "dc:a_side2": numericunitsv(a_side2,"m"),
    }
    
    #pl.show()

    
