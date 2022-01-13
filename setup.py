import sys
#from numpy.distutils.core import setup as numpy_setup, Extension as numpy_Extension
import os
import os.path
import subprocess
import re
from setuptools import setup
from setuptools.command.install_lib import install_lib
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
import setuptools.command.bdist_egg
import sys
import distutils.spawn
import numpy as np
from Cython.Build import cythonize


extra_compile_args = {
    "msvc": ["/openmp"],
    #"unix": ["-O0", "-g", "-Wno-uninitialized"),    # Replace the line below with this line to enable debugging of the compiled extension
    "unix": ["-fopenmp","-O5","-Wno-uninitialized"],
    "clang": ["-fopenmp","-O5","-Wno-uninitialized"],
}

extra_include_dirs = {
    "msvc": [".", np.get_include() ],
    "unix": [".", np.get_include() ],
    "clang": [".", np.get_include() ],
}

extra_libraries = {
    "msvc": [],
    "unix": ["gomp",],
    "clang": [],
}

extra_link_args = {
    "msvc": [],
    "unix": [],
    "clang": ["-fopenmp=libomp"],
}

class build_ext_compile_args(build_ext):
    def build_extensions(self):
        compiler=self.compiler.compiler_type
        for ext in self.extensions:
            if compiler in extra_compile_args:
                ext.extra_compile_args=extra_compile_args[compiler]
                ext.extra_link_args=extra_link_args[compiler]
                ext.include_dirs.extend(list(extra_include_dirs[compiler]))
                ext.libraries.extend(list(extra_libraries[compiler]))
                pass
            else:
                # use unix parameters as default
                ext.extra_compile_args=extra_compile_args["unix"]
                ext.extra_link_args=extra_link_args["unix"]
                ext.include_dirs.extend(list(extra_include_dirs["unix"]))
                ext.libraries.extend(extra_libraries["unix"])
                pass
                
            pass
            
        
        build_ext.build_extensions(self)
        pass
    pass


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
if os.path.exists(".git") and distutils.spawn.find_executable("git") is not None:
    # Check if tree has been modified
    modified = subprocess.call(["git","diff-index","--quiet","HEAD","--"]) != 0
    
    gitrev = subprocess.check_output(["git","rev-parse","HEAD"]).decode('utf-8').strip()

    version = "git-%s" % (gitrev)

    # See if we can get a more meaningful description from "git describe"
    try:
        versionraw=subprocess.check_output(["git","describe","--tags","--match=v*"],stderr=subprocess.STDOUT).decode('utf-8').strip()
        # versionraw is like v0.1.0-50-g434343
        # for compatibility with PEP 440, change it to
        # something like 0.1.0+50.g434343
        matchobj=re.match(r"""v([^.]+[.][^.]+[.][^-.]+)(-.*)?""",versionraw)
        version=matchobj.group(1)
        if matchobj.group(2) is not None:
            version += '+'+matchobj.group(2)[1:].replace("-",".")
            pass
        pass
    except subprocess.CalledProcessError:
        # Ignore error, falling back to above version string
        pass

    if modified and version.find('+') >= 0:
        version += ".modified"
        pass
    elif modified:
        version += "+modified"
        pass
    pass
else:
    version = "UNKNOWN"
    pass

print("version = %s" % (version))

crackclosuresim2_package_files = [ "pt_steps/*" ]

ext_modules=cythonize("crackclosuresim2/*.pyx")
em_dict=dict([ (module.name,module) for module in ext_modules])
#sca_pyx_ext=em_dict["crackclosuresim2.soft_closure_accel"]
#sca_pyx_ext.include_dirs=[".", np.get_include() ]



console_scripts=["eval_closure_state"]
console_scripts_entrypoints = [ "%s = crackclosuresim2.bin.%s:main" % (script,script.replace("-","_")) for script in console_scripts ]



setup(name="crackclosuresim2",
      description="Crack closure calculations",
      author="Stephen D. Holland",
      version=version,
      url="http://thermal.cnde.iastate.edu",
      zip_safe=False,
      ext_modules=ext_modules,
      packages=["crackclosuresim2","crackclosuresim2.bin"],
      cmdclass={"install_lib": install_lib_save_version,
                "build_ext": build_ext_compile_args},
      package_data={"crackclosuresim2": crackclosuresim2_package_files},
      entry_points={ "limatix.processtrak.step_url_search_path": [ "limatix.share.pt_steps = crackclosuresim2:getstepurlpath" ],
                     "console_scripts": console_scripts_entrypoints,
                 })

