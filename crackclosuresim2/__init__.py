import sys
import os
import os.path

try:
    # py2.x
    from urllib import pathname2url
    pass
except ImportError:
    # py3.x
    from urllib.request import pathname2url
    pass


class dummy(object):
    pass

pkgpath = sys.modules[dummy.__module__].__file__
pkgdir=os.path.split(pkgpath)[0]

def getstepurlpath():

    return [ pathname2url(os.path.join(pkgdir,"pt_steps")) ]


versionpath = os.path.join(pkgdir,"version.txt")
if os.path.exists(versionpath):
    versionfh = open(versionpath,"r")
    __version__=versionfh.read().strip()
    versionfh.close()
    pass
else:
    __version__="UNINSTALLED"
    pass




from .crackclosure import ModeI_crack_model
from .crackclosure import ModeI_Beta_COD_Formula
from .crackclosure import ModeI_Beta_WeightFunction
from .crackclosure import solve_normalstress
from .crackclosure import inverse_closure
from .crackclosure import crackopening_from_tensile_closure
from .crackclosure import tensile_closure_from_crackopening
from .crackclosure import Glinka_ModeI_ThroughCrack
from .crackclosure import ModeI_throughcrack_weightfun
from .crackclosure import ModeI_throughcrack_CODformula
from .crackclosure import Tada_ModeI_CircularCrack_along_midline
from .crackclosure import perform_inverse_closure
from .crackclosure import save_closurestress
from .crackclosure import load_closurestress
from .crackclosure import indef_integral_of_crack_tip_singularity_times_1_over_r2_pos_crossterm_decay


from .shear_stickslip import ModeII_crack_model
from .shear_stickslip import ModeII_Beta_CSD_Formula
from .shear_stickslip import solve_shearstress
from .shear_stickslip import ModeII_throughcrack_CSDformula
from .shear_stickslip import ModeIII_throughcrack_CSDformula

from .fabrikant import Fabrikant_ModeII_CircularCrack_along_midline

def crack_model_normal_by_name(crack_model_normal_name,YoungsModulus,PoissonsRatio,Symmetric_COD):
    if crack_model_normal_name == "ModeI_throughcrack_CODformula":
        crack_model_normal = ModeI_throughcrack_CODformula(YoungsModulus,Symmetric_COD=Symmetric_COD)# ,PoissonsRatio)
        pass
    elif crack_model_normal_name == "Tada_ModeI_CircularCrack_along_midline":
        assert(Symmetric_COD) # Tada is a symmetric model
        crack_model_normal = Tada_ModeI_CircularCrack_along_midline(YoungsModulus,PoissonsRatio)
        pass
    else:
        raise ValueError("Unknown normal stress crack model %s" % (crack_model_normal_name))

    return crack_model_normal




def crack_model_shear_by_name(crack_model_shear_name,YoungsModulus,PoissonsRatio,Symmetric_CSD):
    if crack_model_shear_name == "ModeII_throughcrack_CSDformula":
        crack_model_shear = ModeII_throughcrack_CSDformula(YoungsModulus,PoissonsRatio,Symmetric_CSD=Symmetric_CSD)
        pass
    elif crack_model_shear_name == "Fabrikant_ModeII_CircularCrack_along_midline":
        assert(Symmetric_CSD) # Fabrikant model is a symmetric cmodel
        crack_model_shear = Fabrikant_ModeII_CircularCrack_along_midline(YoungsModulus,PoissonsRatio)
        pass
    elif crack_model_shear_name == "ModeIII_throughcrack_CSDformula":
        crack_model_shear = ModeIII_throughcrack_CSDformula(YoungsModulus,PoissonsRatio,Symmetric_CSD=Symmetric_CSD)
        pass
    else:
        raise ValueError("Unknown shear stress crack model %s" % (crack_model_shear_name))

    return crack_model_shear


