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

def run(_xmldoc,_element,
        _dest_href,
        _inputfilename,
        dc_measnum_int,
        dc_closureprofile_href,
        dc_spcYoungsModulus_numericunits,
        dc_spcPoissonsRatio_numericunits,
        dc_crackpath,
        dc_coordinatetransform,
        dc_symmetric_cod_bool=True,
        debug_bool=False):
    
    # (temporarily?) fixed parameters
    sigma_yield = 400e6
    tau_yield = sigma_yield/2.0 # limits stress concentration around singularity
    E=dc_spcYoungsModulus_numericunits.value("Pa")
    nu = dc_spcPoissonsRatio_numericunits.value("unitless")   #Poisson's Ratio
    specimen_width=25.4e-3


    # read closure profile
    cpdata = np.loadtxt(dc_closureprofile_href.getpath(),skiprows=1,delimiter=',')
    assert(cpdata.shape[1]==3)
    
    loads = cpdata[:,0]
    tippos_side1 = cpdata[:,1]
    tippos_side2 = cpdata[:,2]
    
    sigmaext_max=np.max(loads)


    # We are working in stitched optical image coordinates, so we need
    # to subtract out xshift and yshift from the crack positions encoded
    # in the experiment log


    xshift = numericunitsv.fromxml(_xmldoc, _xmldoc.xpathsinglecontext(dc_coordinatetransform, 'dc:translation/dc:xtranslation')).value('m')
    yshift = numericunitsv.fromxml(_xmldoc, _xmldoc.xpathsinglecontext(dc_coordinatetransform, 'dc:translation/dc:ytranslation')).value('m')
    
    #keypoints = _opxmldoc.xpathcontext(dc_crackpath, 'dc:segment/dc:keypoint')

    crackstartx = numericunitsv.fromxml(_xmldoc,_xmldoc.xpathsinglecontext(dc_crackpath, 'dc:segment/dc:keypoint[1]/dc:xcoordinate')).value('m') - xshift

    crackendx = numericunitsv.fromxml(_xmldoc,_xmldoc.xpathsinglecontext(dc_crackpath, 'dc:segment/dc:keypoint[last()]/dc:xcoordinate')).value('m') - xshift

    
    CrackCenterX = (crackstartx+crackendx)/2.0

    a_side1 = (crackendx-crackstartx)/2.0 # half-crack length (m)
    a_side2 = (crackendx-crackstartx)/2.0 # half-crack length (m)

    # here, x really measures radius past crack center
    xmax_approx = 2.0*a_side1  # x array goes past tip position (twice half-length)
    dx = 25e-6
    xsteps = int(xmax_approx//dx)
    xmax = dx*xsteps

    x_bnd=np.arange(xsteps,dtype='d')*dx
    x = (x_bnd[1:]+x_bnd[:-1])/2.0

    weightfun_epsx = dx/8.0
    crack_model = Tada_ModeI_CircularCrack_along_midline(E,nu)


    # side 1 (left side)
    observed_reff_side1 = CrackCenterX - tippos_side1
    observed_seff_side1 = loads


    if (observed_reff_side1 > a_side1).any():
        a_side1=np.max(observed_reff_side1)
        pass
    
    sigma_closure_side1 = inverse_closure(observed_reff_side1,
                                          observed_seff_side1,
                                          x,x_bnd,dx,a_side1,sigma_yield,
                                          crack_model)


    # side 2 (right side)
    observed_reff_side2 = tippos_side2 - CrackCenterX
    observed_seff_side2 = loads


    if (observed_reff_side2 > a_side2).any():
        a_side2=np.max(observed_reff_side2)
        pass
    
    sigma_closure_side2 = inverse_closure(observed_reff_side2,
                                          observed_seff_side2,
                                          x,x_bnd,dx,a_side2,sigma_yield,
                                          crack_model)

    
    # Forward cross-check of closure
    pl.figure()
    pl.plot(x[x < a_side1]*1e3,sigma_closure_side1[x < a_side1]/1e6,'-',
            observed_reff_side1*1e3,observed_seff_side1/1e6,'x')
    for observcnt in range(len(observed_reff_side1)):        
        (effective_length, sigma, tensile_displ) = solve_normalstress(x,x_bnd,sigma_closure_side1,dx,observed_seff_side1[observcnt],a_side1,sigma_yield,crack_model)
        pl.plot(effective_length*1e3,observed_seff_side1[observcnt]/1e6,'.')
        #pl.plot(x*1e3,tensile_displ*1e15,'-')
        pass
    pl.grid()
    pl.legend(('Closure stress field','Observed crack tip posn','Recon. crack tip posn'),loc="best")
    pl.xlabel('Radius from crack center')
    pl.ylabel('Stress (MPa)')
    pl.title('Side 1 (left)')
    closureplot_side1_href = hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side1.png",contexthref=_dest_href)
    pl.savefig(closureplot_side1_href.getpath(),dpi=300)


    pl.figure()
    pl.plot(x[x < a_side2]*1e3,sigma_closure_side2[x < a_side2]/1e6,'-',
            observed_reff_side2*1e3,observed_seff_side2/1e6,'x')
    for observcnt in range(len(observed_reff_side2)):        
        (effective_length, sigma, tensile_displ) = solve_normalstress(x,x_bnd,sigma_closure_side2,dx,observed_seff_side2[observcnt],a_side2,sigma_yield,crack_model)
        pl.plot(effective_length*1e3,observed_seff_side2[observcnt]/1e6,'.')
        #pl.plot(x*1e3,tensile_displ*1e15,'-')
        pass
    pl.grid()
    pl.legend(('Closure stress field','Observed crack tip posn','Recon. crack tip posn'),loc="best")
    pl.title('Side 2 (right)')
    pl.xlabel('Radius from crack center')
    pl.ylabel('Stress (MPa)')
    closureplot_side2_href = hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side2.png",contexthref=_dest_href)
    pl.savefig(closureplot_side2_href.getpath(),dpi=300)

    closurestress_side1_href=hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side1.csv",contexthref=_dest_href)
    with open(closurestress_side1_href.getpath(),"wb") as csvfile:
        cpwriter = csv.writer(csvfile)
        cpwriter.writerow(["Crack radius (m)","Closure stress (side 1, Pa)"])
        for poscnt in range(np.count_nonzero(x < a_side1)):
            cpwriter.writerow([ x[poscnt], sigma_closure_side1[poscnt]])
            pass
        pass

    closurestress_side2_href=hrefv(posixpath.splitext(dc_closureprofile_href.get_bare_quoted_filename())[0]+"_closurestress_side2.csv",contexthref=_dest_href)
    with open(closurestress_side2_href.getpath(),"wb") as csvfile:
        cpwriter = csv.writer(csvfile)
        cpwriter.writerow(["Crack radius (m)","Closure stress (side 1, Pa)"])
        for poscnt in range(np.count_nonzero(x < a_side2)):
            cpwriter.writerow([ x[poscnt], sigma_closure_side2[poscnt]])
            pass
        pass

    return {
        "dc:closurestress_side1": closurestress_side1_href,
        "dc:closureplot_side1": closureplot_side1_href,
        "dc:closurestress_side2": closurestress_side2_href,
        "dc:closureplot_side2": closureplot_side2_href,
    }
    
    #pl.show()

    
