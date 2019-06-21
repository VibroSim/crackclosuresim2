import sys
#from numpy.distutils.core import setup as numpy_setup, Extension as numpy_Extension
import os
import os.path
import subprocess
from setuptools import setup
from setuptools.command.install_lib import install_lib
from setuptools.command.install import install
import setuptools.command.bdist_egg
import sys




class install_lib_save_version(install_lib):
    """Save version information"""
    def run(self):
        install_lib.run(self)
        
        for package in self.distribution.command_obj["build_py"].packages:
            install_dir=os.path.join(*([self.install_dir] + package.split('.')))
            fh=open(os.path.join(install_dir,"version.txt"),"w")
            fh.write("%s\n" % (version))  # version global, as created below
            fh.close()
            pass
        pass
    pass



# Extract GIT version
if os.path.exists(".git"):
    # Check if tree has been modified
    modified = subprocess.call(["git","diff-index","--quiet","HEAD","--"]) != 0
    
    gitrev = subprocess.check_output(["git","rev-parse","HEAD"]).strip()

    version = "git-%s" % (gitrev)

    # See if we can get a more meaningful description from "git describe"
    try:
        version=subprocess.check_output(["git","describe","--tags","--match=v*"],stderr=subprocess.STDOUT).strip()
        pass
    except subprocess.CalledProcessError:
        # Ignore error, falling back to above version string
        pass

    if modified:
        version += "-MODIFIED"
        pass
    pass
else:
    version = "UNKNOWN"
    pass

crackclosuresim2_package_files = [ "pt_steps/*" ]

#console_scripts=["closure_measurement_dic","closure_measurement_coords","closure_measurement_processing"]
#console_scripts_entrypoints = [ "%s = closure_measurements.bin.%s:main" % (script,script.replace("-","_")) for script in console_scripts ]


setup(name="crackclosuresim2",
      description="Crack closure calculations",
      author="Stephen D. Holland",
      version=version,
      url="http://thermal.cnde.iastate.edu",
      zip_safe=False,
      packages=["crackclosuresim2"],
      cmdclass={"install_lib": install_lib_save_version },
      package_data={"crackclosuresim2": crackclosuresim2_package_files},
      entry_points={ "limatix.processtrak.step_url_search_path": [ "limatix.share.pt_steps = crackclosuresim2:getstepurlpath" ],
                     #"console_scripts": console_scripts_entrypoints,
                     })

