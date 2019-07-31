from . import ModeI_throughcrack_CODformula
from . import Tada_ModeI_CircularCrack_along_midline


import numpy as np
cimport numpy as np


cdef extern from "soft_closure_accel_ops.h":
    cdef struct crack_model_throughcrack_t:
        double Eeff
        double Beta 
        pass
    cdef struct crack_model_tada_t:
        double E
        double nu 
        double Beta 
        pass
    cdef struct modeldat_t:
        crack_model_throughcrack_t through
        crack_model_tada_t tada
        pass

    cdef struct crack_model_t:
        modeldat_t modeldat
        int modeltype
        pass
    
    cdef int CMT_THROUGH
    cdef int CMT_TADA
    pass

    cdef double initialize_contact_goal_function_c(double *du_da_shortened,int du_da_shortened_len,int closure_index,unsigned xsteps,unsigned fine_refinement,int afull_idx_fine,double *sigma_closure_interp,double xfine0,double dx_fine,double Hm,crack_model_t crack_model)
    cdef double soft_closure_goal_function_c(double *du_da_shortened,int du_da_shortened_len,int closure_index,unsigned xsteps,unsigned fine_refinement,int afull_idx_fine,double *crack_initial_opening_interp,double *sigma_closure_interp,double xfine0,double dx_fine,double Hm,crack_model_t crack_model)

def initialize_contact_goal_function_accel(np.ndarray[np.float64_t,ndim=1] du_da_shortened,scp,np.ndarray[np.float64_t,ndim=1] sigma_closure_interp,int closure_index):
    """ NOTE: This should be kept identical functionally to initialize_contact_goal_function in soft_closure.py"""

    # ***NOTE: Could define a separate version that doesn't
    # bother to calculate  dsigmaext_dxt_hardcontact_interp when calculating
    # compressive (sigma_ext < 0) loads. 

    cdef unsigned xsteps
    cdef unsigned fine_refinement
    cdef int afull_idx_fine
    cdef double xfine0  # first refined x position
    cdef double dx_fine
    cdef crack_model_t crack_model;
    cdef double Hm
    
    
    xsteps = scp.xsteps
    fine_refinement = scp.fine_refinement
    afull_idx_fine = scp.afull_idx_fine
    xfine0 = scp.x_fine[0]
    dx_fine = scp.dx_fine
    Hm = scp.Hm

    if isinstance(scp.crack_model,ModeI_throughcrack_CODformula):

        crack_model.modeltype=CMT_THROUGH
        crack_model.modeldat.through.Eeff = scp.crack_model.Eeff
        crack_model.modeldat.through.Beta = scp.crack_model.beta(scp.crack_model)
        pass
    elif isinstance(scp.crack_model,Tada_ModeI_CircularCrack_along_midline):
        crack_model.modeltype=CMT_TADA
        crack_model.modeldat.tada.E = scp.crack_model.E
        crack_model.modeldat.tada.nu = scp.crack_model.nu
        crack_model.modeldat.tada.Beta = scp.crack_model.beta(scp.crack_model)
        pass
    

    return initialize_contact_goal_function_c(<double *>du_da_shortened.data,du_da_shortened.shape[0],closure_index,xsteps,fine_refinement,afull_idx_fine,<double *>sigma_closure_interp.data,xfine0,dx_fine,Hm,crack_model)


def soft_closure_goal_function_accel(np.ndarray[np.float64_t,ndim=1] du_da_shortened,scp,int closure_index):
    """ NOTE: This should be kept identical functionally to soft_closure_goal_function in soft_closure.py"""
    # ***NOTE: Could define a separate version that doesn't
    # bother to calculate  dsigmaext_dxt_hardcontact_interp when calculating
    # compressive (sigma_ext < 0) loads. 

    cdef unsigned xsteps
    cdef unsigned fine_refinement
    cdef int afull_idx_fine
    cdef np.ndarray[np.float64_t,ndim=1] crack_initial_opening_interp
    cdef np.ndarray[np.float64_t,ndim=1] sigma_closure_interp
    cdef double xfine0  # first refined x position
    cdef double dx_fine
    cdef crack_model_t crack_model;
    cdef double Hm
    
    
    xsteps = scp.xsteps
    fine_refinement = scp.fine_refinement 
    afull_idx_fine = scp.afull_idx_fine
    crack_initial_opening_interp = scp.crack_initial_opening_interp
    sigma_closure_interp = scp.sigma_closure_interp
    xfine0 = scp.x_fine[0]
    dx_fine = scp.dx_fine
    Hm = scp.Hm 

    if isinstance(scp.crack_model,ModeI_throughcrack_CODformula):

        crack_model.modeltype=CMT_THROUGH
        crack_model.modeldat.through.Eeff = scp.crack_model.Eeff
        crack_model.modeldat.through.Beta = scp.crack_model.beta(scp.crack_model)
        pass
    elif isinstance(scp.crack_model,Tada_ModeI_CircularCrack_along_midline):
        crack_model.modeltype=CMT_TADA
        crack_model.modeldat.tada.E = scp.crack_model.E
        crack_model.modeldat.tada.nu = scp.crack_model.nu
        crack_model.modeldat.tada.Beta = scp.crack_model.beta(scp.crack_model)
    else:
        crack_model.modeltype=CMT_THROUGH
        crack_model.modeldat.through.Eeff = 0.0
        crack_model.modeldat.through.Beta = 0.0
        raise ValueError("Invalid crack model class")
    
    
    return soft_closure_goal_function_c(<double *>du_da_shortened.data,du_da_shortened.shape[0],closure_index,xsteps,fine_refinement,afull_idx_fine,<double *>crack_initial_opening_interp.data,<double *>sigma_closure_interp.data,xfine0,dx_fine,Hm,crack_model)
