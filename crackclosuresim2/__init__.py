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
from .crackclosure import Glinka_ModeI_ThroughCrack
from .crackclosure import ModeI_throughcrack_weightfun
from .crackclosure import ModeI_throughcrack_CODformula
from .crackclosure import Tada_ModeI_CircularCrack_along_midline
from .crackclosure import perform_inverse_closure
from .crackclosure import save_closurestress



from .shear_stickslip import ModeII_crack_model
from .shear_stickslip import ModeII_Beta_CSD_Formula
from .shear_stickslip import solve_shearstress
from .shear_stickslip import ModeII_throughcrack_CSDformula


