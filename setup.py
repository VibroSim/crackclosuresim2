import sys
from numpy.distutils.core import setup as numpy_setup, Extension as numpy_Extension

numpy_setup(name="crackclosuresim2",
            description="crackclosuresim2",
            author="Stephen D. Holland",
            url="http://thermal.cnde.iastate.edu",
            packages=["crackclosuresim2"])

