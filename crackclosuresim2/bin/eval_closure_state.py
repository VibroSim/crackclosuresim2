import sys
import csv
import ast
import copy
import posixpath
import numpy as np
import tempfile
import os
import os.path

from matplotlib import pyplot as pl

from crackclosuresim2 import inverse_closure,solve_normalstress
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2 import perform_inverse_closure,save_closurestress



def main(args=None):
    if args is None:
        args=sys.argv
        pass    

    closureprofilefile=None
    CrackCenterX=None
    YoungsModulus=None
    PoissonsRatio=None
    YieldStrength=None
    
    dx=5e-6
    
    #Symmetric_COD=True # assume a symmetric form for the COD -- appropriate when the data is from surface cracks of length 2a where the center is (roughly) a symmetry point


    if len(args) < 3:
        print("Usage: eval_closure_state <closure_profile_csvfile> <CrackCenterX> <YoungsModulus> <Poisson's Ratio> <Yield Strength> [dx]")
        print(" ")
        print("Process closure profile (tip position as a function of load) CSV file")
        print("To determine closure stress field")
        print("Resulting closure stress field will be written into %s" % (tempfile.gettempdir()))
        print(" ")
        print("Parameters:")
        print("  closure_profile_csvfile: Path to .csv file with tip position data. First line is")
        print("                    presumed to be a header and is ignored. ")
        print("  CrackCenterX:     X position (meters, measured in coordinate frame of csvfile)")
        print("                    of crack center.")
        print("  YoungsModulus:    Value for Young's modulus, in Pascals. Typically")
        print("                    either 113.8e9 for Ti-6-4 or 200e9 for In718")
        print("  Poisson's Ratio:  Value for Poisson's ratio (unitless)")
        print("  Yield Strength:   Value for yield strength, in Pascals.")
        print("  dx:               Step size for resulting closure stress field")
        sys.exit(0)
        pass

    
    closureprofilefile = args[1]
    CrackCenterX = float(args[2])
    YoungsModulus = float(args[3])
    PoissonsRatio = float(args[4])
    YieldStrength = float(args[5])
    
    if len(args) > 6:
        dx=float(args[6])
        pass

    (x,x_bnd,a_side1,a_side2,sigma_closure_side1,sigma_closure_side2,side1fig,side2fig) = perform_inverse_closure(closureprofilefile,YoungsModulus,PoissonsRatio,YieldStrength,CrackCenterX,dx)

    
    outfilename_side1 = os.path.join(tempfile.gettempdir(),os.path.splitext(os.path.split(closureprofilefile)[1])[0]+"_closurestress_side1.csv")
    save_closurestress(outfilename_side1,x,sigma_closure_side1,a_side1)

    outfilename_side2 = os.path.join(tempfile.gettempdir(),os.path.splitext(os.path.split(closureprofilefile)[1])[0]+"_closurestress_side2.csv")
    save_closurestress(outfilename_side2,x,sigma_closure_side2,a_side2)

    pl.show()
    pass
