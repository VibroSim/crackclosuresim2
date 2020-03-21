import numpy as np
from matplotlib import pyplot as pl
import pickle
import copy
import os 
import os.path
import sys
from VibroSim_Simulator.function_as_script import scriptify
from crackclosuresim2 import Tada_ModeI_CircularCrack_along_midline
from crackclosuresim2.soft_closure import soft_closure_goal_function
from crackclosuresim2.soft_closure_accel import soft_closure_goal_function_accel


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

pickle_filename = "/tmp/scebug26203_line.pickle"
pickle_fh=open(pickle_filename,"rb")
vars = pickle.load(pickle_fh)
scp = vars["scp"]
E=vars["E"]
nu=vars["nu"]
closure_index=vars["closure_index"]
res_x=vars["res_x"]
scp.crack_model=Tada_ModeI_CircularCrack_along_midline(E,nu)


py = scriptify(soft_closure_goal_function)(res_x,scp,closure_index)
accel = soft_closure_goal_function_accel(res_x,scp,closure_index)

print("py terms: %g, %g, %g" % (np.sum(residual[:-1]**2.0),np.sum(negative**2.0),np.sum(displaced**2.0)))
print("py=%g; accel=%g" % (py,accel))

