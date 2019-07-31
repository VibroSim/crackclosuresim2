#include <stdlib.h>

struct crack_model_throughcrack_t {
  double Eeff;
  double Beta;
};

struct crack_model_tada_t {
  double E;
  double nu;
  double Beta;
};

union modeldat_t {
    struct crack_model_throughcrack_t through;
    struct crack_model_tada_t tada;
};

struct crack_model_t
{
  union modeldat_t modeldat;
  int modeltype; // see DEFINES below
};

#define CMT_THROUGH 0
#define CMT_TADA 1
  
static void sigmacontact_from_displacement(double *du_da_short,
					   int du_da_short_len,
					   int afull_idx_fine,
					   double *crack_initial_opening_interp,
					   double *sigma_closure_interp,
					   double xfine0,
					   double dx_fine,
					   double Hm,
					   struct crack_model_t crack_model,
					   // Output parameters
					   double *from_displacement,
					   double *displacement)
// NOTE: This should be kept functionally identical to sigmacontact_from_displacement() in soft_closure.py
{
  int cnt;
  int aidx;
  
  for (cnt=0;cnt < du_da_short_len;cnt++) {
    displacement[cnt] = crack_initial_opening_interp[cnt] - pow(sigma_closure_interp[cnt]/Hm,2.0/3.0);

  }

  if (crack_model.modeltype==CMT_THROUGH) {
    for (aidx=afull_idx_fine;aidx >= 0;aidx--) {
      for (cnt=0;cnt < aidx;cnt++) {
	displacement[cnt] += (4.0/crack_model.modeldat.through.Eeff)*du_da_short[aidx]*sqrt((xfine0+aidx*dx_fine + xfine0+cnt*dx_fine)*(xfine0+aidx*dx_fine - xfine0-cnt*dx_fine))*dx_fine;
      }
      displacement[aidx] += (4.0/crack_model.modeldat.through.Eeff)*du_da_short[aidx]*sqrt(2.0*(xfine0+aidx*dx_fine))*pow(dx_fine/2.0,3.0/2.0);
      
    }
  } else if (crack_model.modeltype==CMT_TADA) {
    for (aidx=afull_idx_fine;aidx >= 0;aidx--) {
      for (cnt=0;cnt < aidx;cnt++) {
	displacement[cnt] += (8.0*(1.0-pow(crack_model.modeldat.tada.nu,2.0))/(M_PI*crack_model.modeldat.tada.E))*du_da_short[aidx]*sqrt((xfine0+aidx*dx_fine + xfine0+cnt*dx_fine)*(xfine0+aidx*dx_fine - xfine0-cnt*dx_fine))*dx_fine;
      }
      displacement[aidx] += (8.0*(1.0-pow(crack_model.modeldat.tada.nu,2.0))/(M_PI*crack_model.modeldat.tada.E))*du_da_short[aidx]*sqrt(2.0*(xfine0+aidx*dx_fine))*pow(dx_fine/2.0,3.0/2.0);
      
    }

  } else {
    assert(0);
  }
  
  for (cnt=0;cnt < du_da_short_len;cnt++) {
    if (displacement[cnt] < 0.0) {
      from_displacement[cnt] = pow(-displacement[cnt],3.0/2.0) * Hm;
      
    } else {
      from_displacement[cnt] = 0.0;
      
    }
  }
}



static void sigmacontact_from_stress(double *du_da_short,
					   int du_da_short_len,
					   int afull_idx_fine,
					   double *sigma_closure_interp,
					   double xfine0,
					   double dx_fine,
					   struct crack_model_t crack_model,
					   // Output parameters
					   double *from_stress)
// NOTE: This should be kept functionally identical to sigmacontact_from_stress() in soft_closure.py
{
  int cnt;
  int aidx;
  double betaval=0.0;
  
  for (cnt=0;cnt < du_da_short_len;cnt++) {
    from_stress[cnt] = sigma_closure_interp[cnt];
  
  }
  
  if (crack_model.modeltype==CMT_THROUGH) {
    betaval = crack_model.modeldat.through.Beta;
  } else if (crack_model.modeltype==CMT_TADA) {
    betaval = crack_model.modeldat.tada.Beta;
  } else {
    assert(0);    
  }
  
  for (aidx=0;aidx <= afull_idx_fine;aidx++) {
    for (cnt=aidx+1;cnt <= afull_idx_fine;cnt++) {
      from_stress[cnt] -= du_da_short[aidx]*((betaval/M_SQRT2)*sqrt((xfine0+aidx*dx_fine)/(xfine0+cnt*dx_fine - xfine0-aidx*dx_fine)) + 1.0)*dx_fine;
    }
    from_stress[aidx] -= (du_da_short[aidx]*(betaval/M_SQRT2)*sqrt(xfine0+aidx*dx_fine)*2.0*sqrt(dx_fine/2.0) + du_da_short[aidx]*dx_fine/2.0);
  }
}



static double initialize_contact_goal_function_c(double *du_da_shortened,int du_da_shortened_len,int closure_index,unsigned xsteps,unsigned fine_refinement,int afull_idx_fine,double *sigma_closure_interp,double xfine0,double dx_fine,double Hm,struct crack_model_t crack_model)
// NOTE: This should be kept identical functionally to initialize_contact_goal_function in soft_closure_accel.py
{
  double *du_da_short;
  int du_da_short_len,cnt;
  double residual=0.0;
  double *from_stress;

  du_da_short_len=(closure_index+1+du_da_shortened_len);
  
  du_da_short = malloc(sizeof(double)*du_da_short_len);
  
  for (cnt=0;cnt < du_da_short_len;cnt++) {
    if (cnt <= closure_index) {
      du_da_short[cnt]=0.0;
    } else {
      du_da_short[cnt]=du_da_shortened[cnt-closure_index-1];
    }
  }
  
  from_stress = malloc(sizeof(double)*du_da_short_len);

  
  sigmacontact_from_stress(du_da_short,du_da_short_len,
			   afull_idx_fine,
			   sigma_closure_interp,
			   xfine0,
			   dx_fine,
			   crack_model,
			   from_stress);


  for (cnt=0;cnt < du_da_short_len;cnt++) {
    residual += pow(sigma_closure_interp[cnt]-from_stress[cnt],2.0);
    
  }
  free(from_stress);
  free(du_da_short);

  return residual;
}



static double soft_closure_goal_function_c(double *du_da_shortened,int du_da_shortened_len,int closure_index,unsigned xsteps,unsigned fine_refinement,int afull_idx_fine,double *crack_initial_opening_interp,double *sigma_closure_interp,double xfine0,double dx_fine,double Hm,struct crack_model_t crack_model)
// NOTE: This should be kept identical functionally to soft_closure_goal_function in soft_closure_accel.py
{
  double *du_da_short;
  int du_da_short_len,cnt;
  double residual=0.0;
  double average=0.0;
  double negative=0.0;
  double displaced=0.0;
  double *displacement,*from_displacement,*from_stress;

  du_da_short_len=(closure_index+1+du_da_shortened_len);
  
  du_da_short = malloc(sizeof(double)*du_da_short_len);
  
  for (cnt=0;cnt < du_da_short_len;cnt++) {
    if (cnt <= closure_index) {
      du_da_short[cnt]=0.0;
    } else {
      du_da_short[cnt]=du_da_shortened[cnt-closure_index-1];
    }
  }
  
  from_displacement = malloc(sizeof(double)*du_da_short_len);
  displacement = malloc(sizeof(double)*du_da_short_len);
  from_stress = malloc(sizeof(double)*du_da_short_len);

  // dirty little trick to run sigmacontact_from_displacement()
  // and sigmacontact_from_stress() in parallel if possible with
  // OpenMP: Put them in a loop with two iterations.
  {
#pragma omp parallel default(shared) num_threads(2)
#pragma omp for schedule(static,1)
    for (int iter=0;iter < 2; iter++) {
      if (iter==0) {
	sigmacontact_from_displacement(du_da_short,du_da_short_len,
				       afull_idx_fine,
				       crack_initial_opening_interp,
				       sigma_closure_interp,
				       xfine0,
				       dx_fine,
				       Hm,
				       crack_model,
				       from_displacement,
				       displacement);
      } else {
	sigmacontact_from_stress(du_da_short,du_da_short_len,
				 afull_idx_fine,
				 sigma_closure_interp,
				 xfine0,
				 dx_fine,
				 crack_model,
				 from_stress);
      }
    }
#pragma omp barrier
  }

  
  // We only worry about residual, negative, and displaced
  // up to the point before the last... why?
  //  well the last point corresponds to the crack tip, which
  // CAN hold tension and doesn't have to follow the contact stress
  // law... so we only iterate up to du_da_short_len-1,
  // representing that a stress concentration
  //  at the crack tip is OK for our goal 
  for (cnt=0;cnt < du_da_short_len-1;cnt++) {
    residual += pow(from_displacement[cnt]-from_stress[cnt],2.0);
    average = (from_displacement[cnt]+from_stress[cnt])/2.0;

    if (average < 0.0) {
      negative += pow(average,2.0); // negative sigmacontact means tension on the surfaces, which is not allowed (except at the actual tip)!

      if (displacement[cnt] > 0.0) {
	displaced += pow(average,2.0);  // should not have stresses with positive displacement 
      }
    }
    
  }
  free(from_stress);
  free(displacement);
  free(from_displacement);
  free(du_da_short);

  return residual + negative + displaced;
}
