#include <stdlib.h>

struct crack_model_throughcrack_t {
  double Eeff;
  double Beta;
  double r0_over_a;
};

struct crack_model_tada_t {
  double E;
  double nu;
  double Beta;
  double r0_over_a;
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
					   int du_da_shortened_len,
					   int afull_idx,
					   double *crack_initial_opening_interp,
					   double *sigma_closure_interp,
					   double x0,
					   double dx,
					   double Lm,
					   struct crack_model_t crack_model,
					   int closure_index_for_gradient,
					   // Output parameters
					   double *from_displacement,
					   double *displacement,
					   double *from_displacement_gradient, // axis zero (changes more slowly) is position along crack; axis one (changes more quickly) is du_da_shortened element
					   double *displacement_gradient) // axis zero (changes more slowly) is position along crack; axis one (changes more quickly) is du_da_shortened element
// NOTE: This should be kept functionally identical to sigmacontact_from_displacement() in soft_closure.py
{
  int cnt;
  int du_da_pos;
  int du_da_shortened_index;
  int aidx;
  
  for (cnt=0;cnt < du_da_short_len-1;cnt++) {
    displacement[cnt] = crack_initial_opening_interp[cnt] - pow(sigma_closure_interp[cnt]/Lm,2.0/3.0);

    for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
      displacement_gradient[cnt*du_da_shortened_len + du_da_pos] = 0.0;
    }
  }

  
  if (crack_model.modeltype==CMT_THROUGH) {
    for (aidx=afull_idx;aidx >= 0;aidx--) {
      du_da_shortened_index = aidx - closure_index_for_gradient;

      for (cnt=0;cnt < aidx;cnt++) {
	displacement[cnt] += (4.0/crack_model.modeldat.through.Eeff)*du_da_short[aidx+1]*sqrt((x0+aidx*dx + x0+cnt*dx)*(x0+aidx*dx - x0-cnt*dx))*dx;
	
      }
      
      displacement[aidx] += (4.0/crack_model.modeldat.through.Eeff)*du_da_short[aidx+1]*sqrt(2.0*(x0+aidx*dx))*pow(dx/2.0,3.0/2.0);

      if (aidx+1 >= closure_index_for_gradient+2) {
	for (cnt=0;cnt < aidx;cnt++) {
	  displacement_gradient[cnt*du_da_shortened_len + du_da_shortened_index] += (4.0/crack_model.modeldat.through.Eeff)*sqrt((x0+aidx*dx + x0+cnt*dx)*(x0+aidx*dx - x0-cnt*dx))*dx;
	}

	displacement_gradient[aidx*du_da_shortened_len + du_da_shortened_index] += (4.0/crack_model.modeldat.through.Eeff)*sqrt(2.0*(x0+aidx*dx))*pow(dx/2.0,3.0/2.0);
	
      }
      

      
    }
  } else if (crack_model.modeltype==CMT_TADA) {
    for (aidx=afull_idx;aidx >= 0;aidx--) {
      du_da_shortened_index = aidx - closure_index_for_gradient;
      
      for (cnt=0;cnt < aidx;cnt++) {
	displacement[cnt] += (8.0*(1.0-pow(crack_model.modeldat.tada.nu,2.0))/(M_PI*crack_model.modeldat.tada.E))*du_da_short[aidx+1]*sqrt((x0+aidx*dx + x0+cnt*dx)*(x0+aidx*dx - x0-cnt*dx))*dx;
      }
      displacement[aidx] += (8.0*(1.0-pow(crack_model.modeldat.tada.nu,2.0))/(M_PI*crack_model.modeldat.tada.E))*du_da_short[aidx+1]*sqrt(2.0*(x0+aidx*dx))*pow(dx/2.0,3.0/2.0);

      if (aidx+1 >= closure_index_for_gradient+2) {
	for (cnt=0;cnt < aidx;cnt++) {
	  displacement_gradient[cnt*du_da_shortened_len + du_da_shortened_index] += (8.0*(1.0-pow(crack_model.modeldat.tada.nu,2.0))/(M_PI*crack_model.modeldat.tada.E))*sqrt((x0+aidx*dx + x0+cnt*dx)*(x0+aidx*dx - x0-cnt*dx))*dx;
	}
	displacement_gradient[aidx*du_da_shortened_len + du_da_shortened_index] += (8.0*(1.0-pow(crack_model.modeldat.tada.nu,2.0))/(M_PI*crack_model.modeldat.tada.E))*sqrt(2.0*(x0+aidx*dx))*pow(dx/2.0,3.0/2.0);
      }
    }

  } else {
    assert(0);
  }
  
  for (cnt=0;cnt < du_da_short_len-1;cnt++) {
    if (displacement[cnt] < 0.0) {
      from_displacement[cnt] = pow(-displacement[cnt],3.0/2.0) * Lm;
      for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
	from_displacement_gradient[cnt*du_da_shortened_len + du_da_pos] = -(3.0/2.0)*sqrt(-displacement[cnt])*Lm*displacement_gradient[cnt*du_da_shortened_len + du_da_pos];
	
      }
    } else {
      from_displacement[cnt] = 0.0;
      for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
	from_displacement_gradient[cnt*du_da_shortened_len + du_da_pos] = 0.0;
      }
      
    }
  }
}



static void sigmacontact_from_stress(double *du_da_short,
				     int du_da_short_len,
				     int du_da_shortened_len,
				     int afull_idx,
				     double *scp_sigma_closure_interp, // scp.sigma_closure_interp.... NOT NECESSARILY caller's sigma_closure_interp variable
				     double x0,
				     double dx,
				     struct crack_model_t crack_model,
				     int closure_index_for_gradient,
				     // Output parameters
				     double *from_stress,
				     double *from_stress_gradient)
// NOTE: This should be kept functionally identical to sigmacontact_from_stress() in soft_closure.py
{
  int cnt;
  int aidx;
  int du_da_pos,du_da_shortened_index;
  double sqrt_betaval=0.0;
  double a;
  double r;
  double r0_over_a;
  
  for (cnt=0;cnt < du_da_short_len-1;cnt++) {
    from_stress[cnt] = scp_sigma_closure_interp[cnt] - du_da_short[0]*dx;
    
    from_stress_gradient[cnt*du_da_shortened_len + 0] = -dx;
    for (du_da_pos=1;du_da_pos < du_da_shortened_len;du_da_pos++) {
      from_stress_gradient[cnt*du_da_shortened_len + du_da_pos] = 0.0;
    }

  }
  //printf("scp_sigma_closure_interp[0]=%g; du_da_short[0]=%g; dx=%g\n",scp_sigma_closure_interp[0],du_da_short[0],dx);
  //printf("from_stress[0]=%g\n",from_stress[0]);
  
  if (crack_model.modeltype==CMT_THROUGH) {
    sqrt_betaval = sqrt(crack_model.modeldat.through.Beta);
    r0_over_a = crack_model.modeldat.through.r0_over_a;
  } else if (crack_model.modeltype==CMT_TADA) {
    sqrt_betaval = sqrt(crack_model.modeldat.tada.Beta);
    r0_over_a = crack_model.modeldat.tada.r0_over_a;
  } else {
    assert(0);    
  }
  
  for (aidx=0;aidx <= afull_idx;aidx++) {
    du_da_shortened_index = aidx - closure_index_for_gradient;

    
    a = x0 + aidx*dx;
    
    for (cnt=aidx+1;cnt <= afull_idx;cnt++) {
      r = x0+cnt*dx - a;
      //from_stress[cnt] -= du_da_short[aidx+1]*((sqrt_betaval/M_SQRT2)*sqrt((x0+aidx*dx)/(x0+cnt*dx - x0-aidx*dx)) + 1.0)*dx;
      //from_stress[cnt] -= du_da_short[aidx+1]*((sqrt_betaval/M_SQRT2)*sqrt(a/r) + 1.0)*dx;
      from_stress[cnt] -= du_da_short[aidx+1]*((sqrt_betaval/M_SQRT2)*sqrt(a/r)*exp(-r/(r0_over_a*a)) + 1.0)*dx;
    }
    //from_stress[aidx] -= (du_da_short[aidx+1]*(sqrt_betaval/M_SQRT2)*sqrt(x0+aidx*dx)*2.0*sqrt(dx/2.0) + du_da_short[aidx+1]*dx/2.0);
    //from_stress[aidx] -= (du_da_short[aidx+1]*(sqrt_betaval/M_SQRT2)*sqrt(a)*2.0*sqrt(dx/2.0) + du_da_short[aidx+1]*dx/2.0);
    from_stress[aidx] -= (du_da_short[aidx+1]*(sqrt_betaval/M_SQRT2)*sqrt(a)*sqrt(M_PI*r0_over_a*a)*erf(sqrt(dx/(2.0*r0_over_a*a))) + du_da_short[aidx+1]*dx/2.0);

    if (aidx+1 >= closure_index_for_gradient+2) {
      for (cnt=aidx+1;cnt <= afull_idx;cnt++) {
	r = x0+cnt*dx - a;
	from_stress_gradient[cnt*du_da_shortened_len + du_da_shortened_index] -= ((sqrt_betaval/M_SQRT2)*sqrt(a/r)*exp(-r/(r0_over_a*a)) + 1.0)*dx;
	
      }
      from_stress_gradient[aidx*du_da_shortened_len + du_da_shortened_index] -= (sqrt_betaval/M_SQRT2)*sqrt(a)*sqrt(M_PI*r0_over_a*a)*erf(sqrt(dx/(2.0*r0_over_a*a))) + dx/2.0;
    }
    //if (aidx==0) {
    //  printf("fs[0] subtraction=%g\n",(du_da_short[aidx+1]*(sqrt_betaval/M_SQRT2)*sqrt(a)*sqrt(M_PI*r0_over_a*a)*erf(sqrt(dx/(2.0*r0_over_a*a))) + du_da_short[aidx+1]*dx/2.0));
    //  printf("fs[0] subtraction terms/factors=%g, %g, %g, %g, %g, %g\n",du_da_short[aidx+1],(sqrt_betaval/M_SQRT2),sqrt(a),sqrt(M_PI*r0_over_a*a),erf(sqrt(dx/(2.0*r0_over_a*a))),du_da_short[aidx+1]*dx/2.0);
    //  printf("from_stress[0]=%g\n",from_stress[0]);
    //}
  }
}

static void print_array(char *name,double *ptr,int numelem)
{
  int cnt;
  
  printf("%s = [ ",name);

  for (cnt=0;cnt < numelem;cnt++) {
    if (cnt % 6 == 0) {
      printf("\n");
    }
    printf("%g, ", ptr[cnt]);
  }
  printf(" ]\n");
  
} 


static double initialize_contact_goal_function_with_gradient_c(double *du_da_shortened,int du_da_shortened_len,int closure_index,unsigned xsteps,int afull_idx,double *scp_sigma_closure_interp,double *sigma_closure_interp,double x0,double dx,double Lm,struct crack_model_t crack_model,double *du_da_shortened_gradient_out)
// NOTE: This should be kept identical functionally to initialize_contact_goal_function in soft_closure_accel.py
{
  double *du_da_short;
  int du_da_short_len,cnt,du_da_pos;
  double residual=0.0;
  //double *dresidual;
  double *from_stress,*from_stress_gradient;

  du_da_short_len=(closure_index+1+du_da_shortened_len);
  
  du_da_short = malloc(sizeof(double)*du_da_short_len);
  
  for (cnt=0;cnt < du_da_short_len;cnt++) {
    if (cnt == 0) {
      du_da_short[cnt]=du_da_shortened[cnt];
    } else if (cnt <= closure_index+1) {
      du_da_short[cnt]=0.0;
    } else {
      du_da_short[cnt]=du_da_shortened[cnt-closure_index-1];
    }
  }
  
  from_stress = malloc(sizeof(double)*(du_da_short_len-1));
  from_stress_gradient = malloc(sizeof(double)*(du_da_short_len-1)*du_da_shortened_len); // axis zero (changes more slowly) is position along crack; axis one (changes more quickly) is du_da_shortened element
  
  //dresidual = malloc(sizeof(double)*du_da_shortened_len); // axis zero is du_da_shortened element


  for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
    //dresidual[du_da_pos]=0;
    du_da_shortened_gradient_out[du_da_pos]=0.0;
  }

  
  sigmacontact_from_stress(du_da_short,du_da_short_len,
			   du_da_shortened_len,
			   afull_idx,
			   scp_sigma_closure_interp,
			   x0,
			   dx,
			   crack_model,
			   closure_index,
			   from_stress,
			   from_stress_gradient);


  for (cnt=0;cnt < du_da_short_len-1;cnt++) {
    residual += pow(sigma_closure_interp[cnt]-from_stress[cnt],2.0);
    
    for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
      //dresidual[du_da_pos] += 2.0*(sigma_closure_interp[cnt]-from_stress[cnt])*(-from_stress_gradient[cnt*du_da_shortened_len + du_da_pos]);
      du_da_shortened_gradient_out[du_da_pos] += 2.0*(sigma_closure_interp[cnt]-from_stress[cnt])*(-from_stress_gradient[cnt*du_da_shortened_len + du_da_pos]);
    }
  }
  //print_array("sigma_closure_interp",sigma_closure_interp,du_da_short_len-1);
  //print_array("from_stress",from_stress,du_da_short_len-1);
  
  free(from_stress_gradient);
  free(from_stress);
  free(du_da_short);

  return residual;
}



static double soft_closure_goal_function_with_gradient_c(double *du_da_shortened,int du_da_shortened_len,int closure_index,unsigned xsteps,int afull_idx,double *crack_initial_opening_interp,double *sigma_closure_interp,double x0,double dx,double Lm,struct crack_model_t crack_model,double *du_da_shortened_gradient_out)
// NOTE: This should be kept identical functionally to soft_closure_goal_function in soft_closure_accel.py
{
  double *du_da_short;
  int du_da_short_len,cnt;
  int du_da_pos;
  double residual=0.0;
  double *dresidual;
  double average=0.0;
  double *daverage;
  double negative=0.0;
  double *dnegative;
  double displaced=0.0;
  double *ddisplaced;
  double *displacement,*from_displacement,*displacement_gradient,*from_displacement_gradient,*from_stress,*from_stress_gradient;

  du_da_short_len=(closure_index+1+du_da_shortened_len);
  
  du_da_short = malloc(sizeof(double)*du_da_short_len);
  
  for (cnt=0;cnt < du_da_short_len;cnt++) {
    if (cnt == 0) {
      du_da_short[cnt] = du_da_shortened[cnt];
    } else if (cnt <= closure_index+1) {
      du_da_short[cnt]=0.0;
    } else {
      du_da_short[cnt]=du_da_shortened[cnt-closure_index-1];
    }
  }
  
  from_displacement = malloc(sizeof(double)*(du_da_short_len-1));
  displacement = malloc(sizeof(double)*(du_da_short_len-1));

  from_displacement = malloc(sizeof(double)*(du_da_short_len-1));
  from_displacement_gradient = malloc(sizeof(double)*(du_da_short_len-1)*du_da_shortened_len); // axis zero (changes more slowly) is position along crack; axis one (changes more quickly) is du_da_shortened element
  displacement_gradient = malloc(sizeof(double)*(du_da_short_len-1)*du_da_shortened_len); // axis zero (changes more slowly) is position along crack; axis one (changes more quickly) is du_da_shortened element
  
  from_stress = malloc(sizeof(double)*(du_da_short_len-1));
  from_stress_gradient = malloc(sizeof(double)*(du_da_short_len-1)*du_da_shortened_len); // axis zero (changes more slowly) is position along crack; axis one (changes more quickly) is du_da_shortened element

  dresidual = malloc(sizeof(double)*du_da_shortened_len); // axis zero is du_da_shortened element
  daverage = malloc(sizeof(double)*du_da_shortened_len); // axis zero is du_da_shortened element
  dnegative = malloc(sizeof(double)*du_da_shortened_len); // axis zero is du_da_shortened element
  ddisplaced = malloc(sizeof(double)*du_da_shortened_len); // axis zero is du_da_shortened element

  for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
    dresidual[du_da_pos]=0;
    //daverage[du_da_pos]=0;
    dnegative[du_da_pos]=0;
    ddisplaced[du_da_pos]=0;
    
  }
  
  // dirty little trick to run sigmacontact_from_displacement()
  // and sigmacontact_from_stress() in parallel if possible with
  // OpenMP: Put them in a loop with two iterations.
  {
    int iter;
#pragma omp parallel default(shared) num_threads(2) private(iter)
#pragma omp for schedule(static,1)
    for (iter=0;iter < 2; iter++) {
      if (iter==0) {
	sigmacontact_from_displacement(du_da_short,du_da_short_len,
				       du_da_shortened_len,
				       afull_idx,
				       crack_initial_opening_interp,
				       sigma_closure_interp,
				       x0,
				       dx,
				       Lm,
				       crack_model,
				       closure_index,
				       from_displacement,
				       displacement,
				       from_displacement_gradient,
				       displacement_gradient);
      } else {
	sigmacontact_from_stress(du_da_short,du_da_short_len,
				 du_da_shortened_len,
				 afull_idx,
				 sigma_closure_interp,
				 x0,
				 dx,
				 crack_model,
				 closure_index,
				 from_stress,
				 from_stress_gradient);
      }
    }
#pragma omp barrier
  }

  
  // We only worry about residual, negative, and displaced
  // up to the point before the last... why?
  //  well the last point corresponds to the crack tip, which
  // CAN hold tension and doesn't have to follow the contact stress
  // law... so we only iterate up to du_da_short_len-2,
  // representing that a stress concentration
  //  at the crack tip is OK for our goal 
  for (cnt=0;cnt < du_da_short_len-2;cnt++) {
    residual += pow(from_displacement[cnt]-from_stress[cnt],2.0);
    
    for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
      dresidual[du_da_pos] += 2.0*(from_displacement[cnt]-from_stress[cnt])*(from_displacement_gradient[cnt*du_da_shortened_len + du_da_pos]-from_stress_gradient[cnt*du_da_shortened_len + du_da_pos]);
    }
    
    average = (from_displacement[cnt]+from_stress[cnt])/2.0;
    for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
      daverage[du_da_pos] = (from_displacement_gradient[cnt*du_da_shortened_len + du_da_pos]+from_stress_gradient[cnt*du_da_shortened_len + du_da_pos])/2.0;
    }

    if (average < 0.0) {
      negative += pow(average,2.0); // negative sigmacontact means tension on the surfaces, which is not allowed (except at the actual tip)!
      for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
	dnegative[du_da_pos] += 2.0*average*daverage[du_da_pos];
      }
    }

    if (displacement[cnt] > 0.0) {
      displaced += pow(average,2.0);  // should not have stresses with positive displacement 
      for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
	ddisplaced[du_da_pos] += 2.0*average*daverage[du_da_pos];
      }
    }

  }

  
  for (du_da_pos=0;du_da_pos < du_da_shortened_len;du_da_pos++) {
    du_da_shortened_gradient_out[du_da_pos] = dresidual[du_da_pos] + dnegative[du_da_pos] + ddisplaced[du_da_pos];
  }

  //print_array("from_stress",from_stress,du_da_short_len-1);
  //print_array("from_displacement",from_displacement,du_da_short_len-1);
  //print_array("du_da_short",du_da_short,du_da_short_len);

  free(ddisplaced);
  free(dnegative);
  free(daverage);
  free(dresidual);
  free(from_stress_gradient);
  free(from_stress);
  free(displacement_gradient);
  free(from_displacement_gradient);
  free(displacement);
  free(from_displacement);
  free(du_da_short);

  //printf("residual=%g; negative=%g; displaced=%g\n",residual,negative,displaced);

  return residual + negative + displaced;
}
