crackclosuresim2
----------------

Crack closure is the residual stress field between surfaces of a crack. 
It is a side effect of the permanent stretching that occurs due to 
plasticity during the fatigue process that generated a crack. 

Crackclosuresim2 is a Python package for calculating crack closure
states. It can:
   * Evaluate crack closure stress field as a function of position,
     given a series of loads and corresponding crack opening points
   * Evaluate the shift in crack closure stress field and opening
     point when an external load is applied
   * Evaluate shear stick/slip zones when an external shear stress
     is applied to a crack with a known closure stress field
   * Make predictions for several different crack models for different
     crack geometries
   * Use either a hard closure (perfectly smooth crack surface) model
     or a soft closure (Hertzian contact between crack surfaces) model. 

There are a series of examples in the 'demos/' directory. 

Requirements
------------
(older versions may work but have not been tested)
  * Python 2.7.5 or 3.4.9 or newer. 
  * scipy 1.0.0 or newer
  * numpy 1.14.3 or newer
  * matplotlib 1.5.3 or newer
  * IPython/Jupyter (recommended)
  * Cython 0.28.6 or newer
  * git 2.17.1 or newer 
    (you may delete the .git directory if you prefer not to use version
    control.)
  * C compiler corresponding to your Python version 

On Linux these components are usually available as packages from your
operating system vendor. 

On Windows/Macintosh it is usually easiest to use a Python distribution 
such as Anaconda https://www.anaconda.com or Canopy 
https://www.enthought.com/product/canopy/ 

These distributions typically provide the full 
Python/Numpy/Matplotlib/IPython stack by default, so you only need
a few more pieces such as Cython, git, and the C compiler. 
64-bit versions of the distributions are recommended

On Anaconda, you can install Cython and git by opening up an 
Anaconda terminal/command prompt and typing:
  conda install -c anaconda cython  
  conda install -c anaconda git

On Canopy, you can install Cython by opening a Canopy 
terminal/command prompt and typing:
  edm install Cython

Installing the C compiler may be the trickiest part:
 * Windows: Install the correct version of Visual Studio corresponding to
   your version of Python, as listed here: 
      https://wiki.python.org/moin/WindowsCompilers
   You can install either the full version of Visual Studio, or just the
   freely downloadable build tools: 
   https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
   Either way, install C++ build tools and ensure the latest versions of 
   MSVCv142 - VS 2019 C++ x64/x86 build tools and Windows 10 SDK are checked.
 * Macintosh: Install XCode from https://developer.apple.com/
 * Linux:  
    * For RedHat/Fedora/CentOS 
        sudo yum groupinstall 'Development Tools'
    * For Ubuntu/Debian 
        sudo apt install build-essential

Installing crackclosuresim2
---------------------------
From a terminal, command prompt, Anaconda or Canopy terminal, etc. 
change to the crackclosuresim2 source directory and type:
  python setup.py build
  python setup.py install

Depending on your configuration the 'install' step might require
root or administrator permissions. You can also explicitly specify 
a different Python version or binary. 

Running crackclosuresim2
------------------------

Try the examples in the 'demos/' subdirectory. 
   e.g. python test_soft_closure.py

We recommend using an IPython/Jupyter 
Qt console or similar. Usually you will want to 
start your session by initializing matplotlib mode: 
  %matplotlib qt

Then run one of the demos:
  %run test_soft_closure.py

When writing your own Python code, you can import the crackclosuresim2 package
with: 
  import crackclosuresim2




Sponsor Acknowledgment
----------------------

This material is based on work supported by the Air Force Research
Laboratory under Contract #FA8650-16-C-5238 and performed at Iowa
State University

This material is based upon work supported by the Department of the
Air Force under Awards No. FD20301933322 and FD20301933313, Air Force
SBIRs to Core Parts, LLC.

AFRL Public Release Case #AFRL-2021-3480

Any opinions, findings, and conclusions or recommendations expressed
in this material are those of the author(s) and do not necessarily
reflect views of the Department of the Air Force or Core Parts, LLC.
