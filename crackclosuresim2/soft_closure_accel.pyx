from . import ModeI_throughcrack_CODformula
from . import Tada_ModeI_CircularCrack_along_midline


import numpy as np
cimport numpy as np


cdef extern from "soft_closure_accel_ops.h":
    cdef struct crack_model_throughcrack_t:
        double Eeff
        double Beta
        double r0_over_a
        pass
    cdef struct crack_model_tada_t:
        double E
        double nu 
        double Beta
        double r0_over_a
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

    cdef double initialize_contact_goal_function_with_gradient_c(double *du_da_shortened,int du_da_shortened_len,int closure_index,unsigned xsteps,int afull_idx,double *scp_sigma_closure,double *sigma_closure,double x0,double dx,double Lm,crack_model_t crack_model,double *du_da_shortened_gradient_out)
    cdef double soft_closure_goal_function_with_gradient_c(double *du_da_shortened,int du_da_shortened_len,int closure_index,unsigned xsteps,int afull_idx,double *crack_initial_opening,double *sigma_closure,double x0,double dx,double Lm,crack_model_t crack_model,double *du_da_shortened_gradient_out)

def initialize_contact_goal_function_with_gradient_accel(np.ndarray[np.float64_t,ndim=1] du_da_shortened,scp,np.ndarray[np.float64_t,ndim=1] sigma_closure,int closure_index):
    """ NOTE: This should be kept identical functionally to initialize_contact_goal_function in soft_closure.py"""

    # ***NOTE: Could define a separate version that doesn't
    # bother to calculate  dsigmaext_dxt_hardcontact_interp when calculating
    # compressive (sigma_ext < 0) loads. 
    cdef np.ndarray[np.float64_t,ndim=1] scp_sigma_closure
    cdef np.ndarray[np.float64_t,ndim=1] gradient
    cdef unsigned xsteps
    cdef int afull_idx
    cdef double x0  # first x position
    cdef double dx
    cdef crack_model_t crack_model;
    cdef double Lm
    cdef double goal_function_value
    

    scp_sigma_closure = scp.sigma_closure
    xsteps = scp.xsteps
    afull_idx = scp.afull_idx
    x0 = scp.x[0]
    dx = scp.dx
    Lm = scp.Lm

    if isinstance(scp.crack_model,ModeI_throughcrack_CODformula):

        crack_model.modeltype=CMT_THROUGH
        crack_model.modeldat.through.Eeff = scp.crack_model.Eeff
        crack_model.modeldat.through.Beta = scp.crack_model.beta(scp.crack_model)
        crack_model.modeldat.through.r0_over_a = scp.crack_model.r0_over_a
        pass
    elif isinstance(scp.crack_model,Tada_ModeI_CircularCrack_along_midline):
        crack_model.modeltype=CMT_TADA
        crack_model.modeldat.tada.E = scp.crack_model.E
        crack_model.modeldat.tada.nu = scp.crack_model.nu
        crack_model.modeldat.tada.Beta = scp.crack_model.beta(scp.crack_model)
        crack_model.modeldat.tada.r0_over_a = scp.crack_model.r0_over_a
        pass
    
    gradient = np.empty(du_da_shortened.shape[0],dtype='d')

    goal_function_value = initialize_contact_goal_function_with_gradient_c(<double *>du_da_shortened.data,du_da_shortened.shape[0],closure_index,xsteps,afull_idx,<double *>scp_sigma_closure.data,<double *>sigma_closure.data,x0,dx,Lm,crack_model,<double *>gradient.data)

    return (goal_function_value,gradient)


def soft_closure_goal_function_with_gradient_accel(np.ndarray[np.float64_t,ndim=1] du_da_shortened,scp,int closure_index):
    """ NOTE: This should be kept identical functionally to soft_closure_goal_function in soft_closure.py"""
    # ***NOTE: Could define a separate version that doesn't
    # bother to calculate  dsigmaext_dxt_hardcontact_interp when calculating
    # compressive (sigma_ext < 0) loads. 

    cdef unsigned xsteps
    cdef int afull_idx
    cdef np.ndarray[np.float64_t,ndim=1] crack_initial_opening
    cdef np.ndarray[np.float64_t,ndim=1] sigma_closure
    cdef np.ndarray[np.float64_t,ndim=1] gradient
    cdef double x0  # first refined x position
    cdef double dx
    cdef crack_model_t crack_model
    cdef double Lm
    cdef double goal_function_value
    
    xsteps = scp.xsteps
    afull_idx = scp.afull_idx
    crack_initial_opening = scp.crack_initial_opening
    sigma_closure = scp.sigma_closure
    x0 = scp.x[0]
    dx = scp.dx
    Lm = scp.Lm 

    if isinstance(scp.crack_model,ModeI_throughcrack_CODformula):

        crack_model.modeltype=CMT_THROUGH
        crack_model.modeldat.through.Eeff = scp.crack_model.Eeff
        crack_model.modeldat.through.Beta = scp.crack_model.beta(scp.crack_model)
        crack_model.modeldat.through.r0_over_a = scp.crack_model.r0_over_a
        pass
    elif isinstance(scp.crack_model,Tada_ModeI_CircularCrack_along_midline):
        crack_model.modeltype=CMT_TADA
        crack_model.modeldat.tada.E = scp.crack_model.E
        crack_model.modeldat.tada.nu = scp.crack_model.nu
        crack_model.modeldat.tada.Beta = scp.crack_model.beta(scp.crack_model)
        crack_model.modeldat.tada.r0_over_a = scp.crack_model.r0_over_a
    else:
        crack_model.modeltype=CMT_THROUGH
        crack_model.modeldat.through.Eeff = 0.0
        crack_model.modeldat.through.Beta = 0.0
        crack_model.modeldat.through.r0_over_a = 0.0
        raise ValueError("Invalid crack model class")

    gradient = np.empty(du_da_shortened.shape[0],dtype='d')
    
    
    goal_function_value = soft_closure_goal_function_with_gradient_c(<double *>du_da_shortened.data,du_da_shortened.shape[0],closure_index,xsteps,afull_idx,<double *>crack_initial_opening.data,<double *>sigma_closure.data,x0,dx,Lm,crack_model,<double *>gradient.data)

    return (goal_function_value,gradient)


