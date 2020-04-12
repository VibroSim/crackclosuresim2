import numpy as np
from matplotlib import pyplot as pl
import pickle
import copy
import os 
import os.path
import sys
import scipy
import scipy.optimize
from VibroSim_Simulator.function_as_script import scriptify
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2.soft_closure import duda_shortened__from_duda
from crackclosuresim2.soft_closure import duda_short__from_duda_shortened
from crackclosuresim2.soft_closure import soft_closure_goal_function_with_gradient
from crackclosuresim2.soft_closure_accel import soft_closure_goal_function_with_gradient_accel


# This is useful for debugging situations where soft_closure
# accelerated calculations differ from the Python calculations.

# There are various printfs and print_array() calls
# in soft_closure_accel_ops that can be uncommented
#
# Because this uses scriptify() the various variables inside
# soft_closure_goal_function will be accessible after this
# is run.

# To use, checkpoint the problematic situation with:
# Note that an assumed crack model is hardwired...
# may need to modify this if it is different!

# import pickle
# import copy
# 
# picklefh=open("/tmp/scebug%d_line.pickle" % (os.getpid()),"wb")
# scp_copy=copy.deepcopy(scp)
# scp_copy.crack_model = None
# to_pickle = {
#   "scp": scp_copy,
#   "E": scp.crack_model.E,
#   "nu": scp.crack_model.nu,
#   "res_x": res.x,
#   "closure_index": closure_index}
# pickle.dump(to_pickle,picklefh)
# picklefh.close()

pickle_filename = "/tmp/scdebug25097_line_923.pickle"
pickle_fh=open(pickle_filename,"rb")
vars = pickle.load(pickle_fh)

globals().update(vars)
assert(crack_model_class=="Tada_ModeI_CircularCrack_along_midline")
scp.crack_model=Tada_ModeI_CircularCrack_along_midline(E,nu)

du_da_shortened=duda_shortened__from_duda(du_da,scp.afull_idx,closure_index)

grad_eval = soft_closure_goal_function_with_gradient(du_da_shortened,scp,closure_index)[1]
grad_approx = scipy.optimize.approx_fprime(du_da_shortened,lambda x: soft_closure_goal_function_with_gradient(x,scp,closure_index)[0],sigma_ext/scp.dx/1e6)
grad_sumsquareddiff = np.sqrt(np.sum((grad_eval-grad_approx)**2.0))
grad_sumsquared = np.sqrt(np.sum(grad_eval**2.0))
        

#py = scriptify(soft_closure_goal_function)(res_x,scp,closure_index)
#accel = soft_closure_goal_function_accel(res_x,scp,closure_index)

