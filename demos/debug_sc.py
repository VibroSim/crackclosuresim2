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
from crackclosuresim2.soft_closure import soft_closure_goal_function_with_gradient_normalized
from crackclosuresim2.soft_closure_accel import soft_closure_goal_function_with_gradient_normalized_accel
from crackclosuresim2.soft_closure import duda_shortened__from_duda
from crackclosuresim2.soft_closure import duda_short__from_duda_shortened

# This is useful for debugging situations where soft_closure
# calculations fail to terminate properly

# Because this uses scriptify() the various variables inside
# soft_closure_goal_function will be accessible after this
# is run.


pickle_filename = "/tmp/scdebug74457_line_1015.pickle"
pickle_fh=open(pickle_filename,"rb")
vars = pickle.load(pickle_fh)
#scp = vars["scp"]
#E=vars["E"]
#nu=vars["nu"]
#closure_index=vars["closure_index"]
#res_x=vars["res_x"]
globals().update(vars)
assert(crack_model_class=="Tada_ModeI_CircularCrack_along_midline")
scp.crack_model=Tada_ModeI_CircularCrack_along_midline(E,nu)


#py = scriptify(soft_closure_goal_function)(res_x,scp,closure_index)

pl.figure()
pl.plot(scp.x,contact_stress_from_stress,'-',
        scp.x,contact_stress_from_displacement,'-')


epsval=1e-2
epsvalscaled = epsval

du_da_shortened=duda_shortened__from_duda(du_da,scp.afull_idx,closure_index)


du_da_shortened_iniguess=np.ones(scp.afull_idx+2-(closure_index+1),dtype='d')*(1.0/(scp.afull_idx+1))* sigma_ext/scp.dx  #
#du_da_shortened=du_da_shortened_iniguess


(initial_residual,initial_gradient)=soft_closure_goal_function_with_gradient_normalized_accel(du_da_shortened/du_da_normalization,scp,closure_index,du_da_normalization,goal_function_normalization)

res = scipy.optimize.minimize(soft_closure_goal_function_with_gradient_normalized_accel,du_da_shortened/du_da_normalization,args=(scp,closure_index,du_da_normalization,goal_function_normalization),   # was soft_closure_goal_function_accel
                              constraints = {"type":"eq","fun": lambda du_da_shortened_normalized: ((np.sum(duda_short__from_duda_shortened(du_da_shortened_normalized*du_da_normalization,closure_index))*scp.dx)-sigma_ext)/load_constraint_fun_normalization },
                              method="SLSQP",
                              jac=True,
                              options={"eps": epsvalscaled,
                                       "maxiter": 100000,
                                       "ftol": 1e-12})#scp.afull_idx*(np.abs(sigma_ext)+20e6)**2.0/1e14})
