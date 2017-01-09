#include "header.h"

// Following the dissertation of Sou Cheng Choi 

static long double *y2, *ty, *ty2, *yUqk;
static long double *dft_tmp0, *dft_tmp1;
static long double *l0[2], *l0k[2], *p;
static long double lambda;

static long double *dy, *halfH, *ink2;
static long double *dyUq, *ydyUq, *yUq;
static long double *tdy, *tdyk;
static long double *ty2dy, *ty2dyk;
static long double *tydy, *tydyk;
static long double *ytydy, *ytydyk;
static long double *ydy, *ydyk;
static long double *y2dy, *y2dyk;
static long double *Ty2dyk;
static long double *Tydy, *Tydyk;
static long double *yTydy, *yTydyk;
static long double *ytdy, *ytdyk;
static long double *y2tdy, *y2tdyk;

static long double *Av, *tmpk;
static long double factor;

int read_guess(params_ptr in) {
  char line[512], v1[80], v2[80], v3[80], v4[80], v5[80];
  long double k;
  long int N_res;
  int j = 0;

  allocate_global_memory(in);
  y2  = fftwl_malloc(nC*sizeof(long double));
  ty2 = fftwl_malloc(nC*sizeof(long double));
  
  in->k0 = 1.L;
  in->g = 1.L;
  // Nondimensional units
  Tau = in->g/(in->k0*powl(in->c,2));
  Ksi = in->v/(in->k0*in->c);
  Sigma = 1.L/(in->k0*in->D);
  lambda = Tau + Ksi;
  printf("Nondimensional Units:\n");
  printf("Tau = %.12LE\nKsi = %.12LE\nSigma = %.12LE\n\n", Tau, Ksi, Sigma);
  
  if (fgets(line, 512, in->fh)) printf("");
  if (fgets(line, 512, in->fh)) sscanf(line, "# N = %s\tL = %s\n", v1, v2);
  N_res = strtol(v1, NULL, 10);	in->L_res = strtold(v2, NULL);
  printf("Restart:\nN = %7ld modes\nL = %.12LE\n", N_res, in->L_res);
  if (fgets(line, 512, in->fh)) printf("%s", line);
  while (fgets(line, 512, in->fh)) {
    sscanf(line, "%s\t%s\t%s\n", v1, v2, v3);
    yk[j] = strtold(v3, NULL);
    k = strtold(v1, NULL);
    j++;
    if (k < 0) break;
    if (j > nC) {
       printf("Restart has more modes than in simulation. ");
       printf("Rerun with at least %d Fourier modes.\n", N_res);
       return 1;
       break;
    }
  }
  if (j < nC-1) {
     printf("Found %4d Fourier cosine coefficients (%4d of %4d). ", j-1, j-1, nC);
     printf("Filled the rest with zeros.\n");
     for (int l = j+1; j < nC; j++) yk[l] = 0.Q;
     return 0;
  } else {
     printf("Restarting from the same number of modes (%4d of %4d).\n", j-1, nC);
     return 0;
  }
}

int init_arrays(params_ptr in) {
    dy = fftwl_malloc(nC*sizeof(long double));
    tdy = fftwl_malloc(nC*sizeof(long double));
    tdyk = fftwl_malloc(nC*sizeof(long double));
    ty2dy = fftwl_malloc(nC*sizeof(long double));
    ty2dyk = fftwl_malloc(nC*sizeof(long double));
    tydy = fftwl_malloc(nC*sizeof(long double));
    tydyk = fftwl_malloc(nC*sizeof(long double));
    ytydy = fftwl_malloc(nC*sizeof(long double));
    ytydyk = fftwl_malloc(nC*sizeof(long double));
    ydy = fftwl_malloc(nC*sizeof(long double));
    ydyk = fftwl_malloc(nC*sizeof(long double));
    y2dy = fftwl_malloc(nC*sizeof(long double));
    y2dyk = fftwl_malloc(nC*sizeof(long double));
    Ty2dyk = fftwl_malloc(nC*sizeof(long double));
    Tydyk = fftwl_malloc(nC*sizeof(long double));
    Tydy = fftwl_malloc(nC*sizeof(long double));
    yTydy = fftwl_malloc(nC*sizeof(long double));
    yTydyk = fftwl_malloc(nC*sizeof(long double));
    ytdy = fftwl_malloc(nC*sizeof(long double));
    ytdyk = fftwl_malloc(nC*sizeof(long double));
    y2tdy = fftwl_malloc(nC*sizeof(long double));
    y2tdyk = fftwl_malloc(nC*sizeof(long double));
    Av = fftwl_malloc(nC*sizeof(long double));		// for Lanczos process
    tmpk = fftwl_malloc(nC*sizeof(long double));	// for Lanczos process
    ink2 = fftwl_malloc(nC*sizeof(long double));	// for preconditioner
    halfH = fftwl_malloc(nC*sizeof(long double));	// for preconditioner
    dUq = fftwl_malloc(nC*sizeof(long double));		// nonuniform grid
    yUqk = fftwl_malloc(nC*sizeof(long double));	// nonuniform grid
    dyUq = fftwl_malloc(nC*sizeof(long double));
    yUq = fftwl_malloc(nC*sizeof(long double));
    ydyUq = fftwl_malloc(nC*sizeof(long double));
	
    long double A = 2.L*(1.L/in->L)/(1.L + powl(1.L/in->L,2));
    long double B = (1.L - powl(1.L/in->L, 2))/(1.L + powl(1.L/in->L, 2));
    long double u = 0, du = 2.L*pi*overN;
    for (long int j = 0; j < nC; j++) {
      u = 2.L*atan2l((in->L)*sinl(0.5L*j*du), cosl(0.5L*j*du));
#if INF_DEPTH
      tk[j] = j;
      H[j] = 1.0L/sqrtl(fabsl(j-lambda));   		
      dUq[j] = A/(1.L + B*cosl(j*du));
#else	
      if (j == 0)	tk[j] = 1.L/in->D;
      else tk[j] = j/tanhl(j*in->D);
      H[j] = 1.0L/sqrtl(fabsl(tk[j]-lambda));	  
      dUq[j] = A/(1.L + B*cosl(j*du)); 
#endif
    }    
    return 0;
}

void fourier_reinterp(long double L1, long double L2) {
   long double du = 2.L*pi*overN;
   memset(y, 0, nC*sizeof(long double));
   for (long int j = 0; j < nC; j++) {
      for (long int l = 0; l < nC; l++) {
	y[j] += 2.L*yk[l]*cosl(2.L*l*atan2l( L1*sinl(0.5L*j*du), L2*cosl(0.5L*j*du)));
      }
      y[j] = y[j] - yk[0];
   }
   dft(y,yk);
   for (long int j = 0; j < nC; j++) yk[j] = yk[j]*overN;
}

int simulate(params_ptr in) {
    long double *pk  = fftwl_malloc(nC*sizeof(long double));
    long double *qk  = fftwl_malloc(nC*sizeof(long double));
    long double *bk  = fftwl_malloc(nC*sizeof(long double));
    long double *Apk  = fftwl_malloc(nC*sizeof(long double));
    linsol	sltn;
    
    sltn.x = fftwl_malloc(nC*sizeof(long double));
    // Note: System is indefinite, the choice of preconditioner is not obvious.
    // Refinement grid conversion:
    if (in->L_res != in->L) {
      printf("Interpolating from L = %.12LE to L = %.12LE\n", in->L_res, in->L);
      fourier_reinterp(in->L_res, in->L);
      compute_rhs(bk, &sltn);
    }
    // Initial Guess:
    for (long int j = 0; j < nC; j++) yk[j] = yk[j]*expl(-0.0L*j); // 0.025
    find_b0(yk, &Beta, &(in->ml)); 
    //printf("Beta = %.12LE\n", Beta); exit(1);
    compute_rhs(bk, &sltn);
    printf("Newton Stage %d: L. Res: %.6LE\tNL Res: %.6LE\tBeta: ", 1, 0.L, sltn.NV);
    printf("%.12LE\tML: %.6LE\n", Beta, in->ml); 
    int it_counter = 0, Max_Newton_Steps = 20000; 	
    while (it_counter < Max_Newton_Steps) {
       minres_classic(bk, 0.L, 24, &sltn);
       //cg_symm(bk, 64, &sltn);
       for (long int j = 0; j < nC; j++) yk[j] = yk[j] + sltn.x[j];
         find_b0(yk, &Beta, &(in->ml)); 
         compute_rhs(bk, &sltn);	
         printf("At Newton Stage %d: L. Res: %.6LE\t", it_counter+2, sltn.phi);
         printf("NL Res: %.6LE\tBeta: %.12LE\tML: %.6LE\n", sltn.NV, Beta, in->ml);
         if (sltn.NV < 2.0E-14L*sqrtl(in->N/131072.L)) {
     	   printf("Converged! Newton Stage %d: ", it_counter+2);
	   printf("NL Res:\t%.12LE\n", sltn.NV);
	   break;
	 }
	 it_counter++;
    }
    dft(yk, y);
    for (long int j = 0; j < nS; j++) {
#if INF_DEPTH
    xk[j] = yk[j+1];  
#else 	
    xk[j] = yk[j+1]/tanhl((j+1)*in->D);
#endif
    }
    dft_sine(xk, x);
    char name[80], fname[80];
    compute_params(in);
    
    FILE *fhs = fopen("steep.txt","r");
    if (fhs == NULL) {
       printf("File steep.txt not found. Creating new one.\n");
       fhs = fopen("steep.txt","w");
       fprintf(fhs, "# 1. H/L 2. c 3. Curvature (conformal) ");
       fprintf(fhs, "4. Max Angle 5. vorticity 6. Depth(Conf) ");
       fprintf(fhs, "7. Depth(Phys)\n\n");
       fclose(fhs);
    }
    printf("Writing Wave parameters to steep.txt\n");
    fhs = fopen("steep.txt","a");
    fprintf(fhs, "%.18LE\t%.18LE\t%.18LE\t", in->HL, in->curv, in->max_angle);
    fprintf(fhs, "%.18LE\t%.18LE\t%.18LE\t%.18LE\n", in->c, in->v, in->D, in->D+yk[0]);
    fclose(fhs);
    
    sprintf(name, "./library/phys/R_%.6Lf.phys.txt", 1.L/sqrtl(in->curv));
    sprintf(fname, "./library/spec/R_%.6Lf.spec.txt", 1.L/sqrtl(in->curv));
    printf("Writing result to:\n%s\n%s\n", name, fname);
    printf("Physical Depth is(reg) \t\t%.12LE\n",  in->D + yk[0]);
    printf("Height of the wave \t\t%.12LE\n", y[nC-1]-y[0]);
    write_FTR(fname, in);
    write_xy(name, in);
    find_spectrum_slope(in);
}


void compute_params(params_ptr in) {
    long double *dx = fftwl_malloc(nC*sizeof(long double));
    long double *dy = fftwl_malloc(nS*sizeof(long double));
    long double *dEta = fftwl_malloc(nC*sizeof(long double));
    long double *d2x = fftwl_malloc(nS*sizeof(long double));
    long double *d2y = fftwl_malloc(nC*sizeof(long double));
    long double Y[3], X[3], dX;
    long double A, B;
    long double xs, ys, du = 2.L*pi*overN;
	
    memset(dx, 0, nC*sizeof(long double));
    memset(dy, 0, nS*sizeof(long double));
    memset(dEta, 0, nC*sizeof(long double));
    for (long int j = 0; j < nS; j++){
	dy[j] = (j+1)*yk[j+1];
	dx[j+1] = (j+1)*xk[j];
    }
    dx[0] = 0.L;
    dft_sine(dy, dy);
    dft(dx, dx);
    long double MinEtax = 0.L;
    for (long int j = 0; j < nC; j++) dx[j] = dx[j]/dUq[j];
    for (long int j = 0; j < nS; j++) {
	dy[j] = dy[j]/dUq[j+1];
	dEta[j] = dy[j]/(1.L + dx[j+1]);
	MinEtax = fminl(MinEtax, dEta[j]);
    }	
    ys = y[nC-1]-y[nC-2];
    xs = (in->L)*du;
    for (long int j = 0; j < 3; j++) {
	Y[j] = (y[nC-1]-y[nC-1-j])/ys;
	X[j] = 2.L*atan2l((in->L)*sinl(0.5*(j*du)), cosl(0.5*(j*du)))/xs;
    }
    B = Y[2]*X[1]*X[1] - Y[1]*X[2]*X[2];
    B = B/(X[1]*X[1]*X[2]*X[2]*(X[2]*X[2] - X[1]*X[1]));
    A = Y[1]/(X[1]*X[1]) - B*(X[1]*X[1]);
    in->curv = 2.0L*A*ys/powl(xs,2);
    in->max_angle = 180.L*atanl(1.L/sqrtl(3.L)-MinEtax)/pi;
    in->HL = 0.5L*(y[nC-1]-y[0])/pi;
    printf("Crest Curvature = %.19LE\n", in->curv);
}

void find_spectrum_slope(params_ptr in) {
    long int IndMin = 0;
    long int IndMax = nC-1;
    long int j = 0;
	
    while (j < nC) {
	if (fabsl(yk[IndMin]) > 1e-10L) {
		IndMin++;
	} else break;
    }
    j = nC-1;
    while (j > 0) {
	if (fabsl(yk[IndMax]) < 1e-16L) {
		IndMax--;
	} else break;
    }
    printf("I_min = %ld\nI_max = %ld\n", IndMin, IndMax);
    if ((IndMax-IndMin) <= 0) {
	printf("Invalid fitting range for Vc\n");
	exit(1);
    } else {
        long int Nr = IndMax - IndMin;
        double *LY = fftwl_malloc(Nr*sizeof(long double));
  	double *LX = fftwl_malloc(Nr*sizeof(long double));
  	double c0, c1, cov00, cov01, cov11, sumsq;
  	for (long int j = IndMin; j < IndMax; j++) {
  	   LY[j-IndMin] = log(fabs(yk[j]));
  	   LX[j-IndMin] = 1.*j;
  	}
  	gsl_fit_linear(LX, 1, LY, 1, Nr, &c0, &c1, &cov00, &cov01, &cov11, &sumsq);
	printf("c0 = %.8E\nc1 = %.8E\nFitting Error = %.8E\n",c0,c1*sqrt(in->L),sumsq/Nr);
    }
}

void compute_rhs(long double *out, linsol_ptr sltn) {
	operator_L0(yk, out);
	sltn->NV = sqrtl(dot_cosine(out,out));
	for (long int j = 0; j < nC; j++) {
		out[j] = -halfH[j]*out[j]; // /(j-lambda);
		if ( fabsl(j-lambda) < 1.0e-10L) {
		  printf("Warning! Small denominator in preconditioner\n");
 		}
	}
}


void cg_symm(long double *bk, const long int maxit, linsol_ptr out) {
	long double *R, *Q, *S;
	long double *Beta, *Chi, *Phi, *Nu, *Mu, *Xi;
	long double Kappa = 1.L, normA = 0.L, Nu_min = 1.L;
	long double normDIRECT = 0.L;
	long int k = 1;
	
	R = fftwl_malloc(nC*sizeof(long double));
	Q = fftwl_malloc(nC*sizeof(long double));
	S = fftwl_malloc(nC*sizeof(long double));
	Beta = fftwl_malloc(maxit*sizeof(long double));
	Chi = fftwl_malloc(maxit*sizeof(long double));	
	Phi = fftwl_malloc(maxit*sizeof(long double));		
	Mu = fftwl_malloc(maxit*sizeof(long double));		
	Nu = fftwl_malloc(maxit*sizeof(long double));		
	Xi = fftwl_malloc(maxit*sizeof(long double));		
	// initialize CG	
	memset(out->x, 0, nC*sizeof(long double));
	memcpy(R, bk, nC*sizeof(long double));
	memcpy(Q, bk, nC*sizeof(long double));
	Beta[0] = sqrtl(dot_cosine(bk,bk));
	Chi[0] = 0.L;
	Phi[0] = Beta[0];
	
	FILE *fhlog = fopen("cg_symm.log","w");
	fprintf(fhlog, "# 1. iter 2. Norm Residual 3. Norm Residual (Direct) 4. Norm Solution\n\n");
	fclose(fhlog);
	
	while (k < maxit) {
		operator_L1_H(Q, S);
		Xi[k-1] = dot_cosine(Q,S);
		Nu[k-1] = Phi[k-1]*Phi[k-1]/Xi[k-1];
		for (long int j = 0; j < nC; j++) {
			out->x[j] = out->x[j] + Nu[k-1]*Q[j];
			R[j] = R[j] - Nu[k-1]*S[j];
		}
		Chi[k-1] = sqrtl(dot_cosine(out->x, out->x));
		Phi[k] = sqrtl(dot_cosine(R,R));
		Mu[k-1] = Phi[k]*Phi[k]/(Phi[k-1]*Phi[k-1]);
		for (long int j = 0; j < nC; j++) Q[j] = R[j] + Mu[k-1]*Q[j];
		Nu_min = fminl(Nu_min, Nu[k-1]);
		normA = fmaxl(normA, Nu[k-1]);
		Kappa = normA/Nu_min;
		
		operator_L1_H(out->x, Av);
		for (long int j = 0; j < nC; j++) Av[j] = Av[j] - bk[j];
		normDIRECT = sqrtl(dot_cosine(Av,Av));
		fhlog = fopen("cg_symm.log","a");
		fprintf(fhlog, "%3d\t%.12LE\t%.12LE\t%.12LE\n", k, Phi[k], normDIRECT, Kappa);
		fclose(fhlog);
		
		out->phi = Phi[k];
		out->chi = Chi[k-1];
		k = k + 1;
	}
	for (long int j = 0; j < nC; j++) out->x[j] = out->x[j]*halfH[j];
}


void minres_classic(long double *bk, long double sigma, const long int maxit, linsol_ptr out) {
	// solve M^-1/2 L1 M^-1/2 dz = - M^-1/2 L0 with MINRES
  char name[80];
  long double *v[3], *d[3], normA;
  long double *Alpha, *Beta, *Epsilon;
  long double	*Phi, *Psi, *Tau, *Chi, *Gamma[2];
  long double *Delta[2], Gamma_min = 1.L;
  long double *c, *s;
  long double Kappa = 1.L; 
  long double normDIRECT = 0.L;
  long int k = 1;
	
  FILE *fhlog = fopen("minres.log","w");
  fprintf(fhlog, "# 1. iter 2. Norm Residual 3. Norm Solution\n\n");
  fclose(fhlog);

  c = fftwl_malloc(maxit*sizeof(long double));	c[0] = -1.L;
  s = fftwl_malloc(maxit*sizeof(long double));	s[0] =  0.L;
  memset(out->x, 0, nC*sizeof(long double));

  Delta[0] = fftwl_malloc(maxit*sizeof(long double)); memset(Delta[0], 0, maxit*sizeof(long double));
  Delta[1] = fftwl_malloc(maxit*sizeof(long double)); memset(Delta[1], 0, maxit*sizeof(long double));
  Alpha = fftwl_malloc(maxit*sizeof(long double)); 	memset(Alpha, 0, maxit*sizeof(long double));
  Beta = fftwl_malloc(maxit*sizeof(long double)); 	Beta[0] = sqrtl(dot_cosine(bk,bk));
  Tau = fftwl_malloc(maxit*sizeof(long double));		Tau[0] = Beta[0];
  Phi = fftwl_malloc(maxit*sizeof(long double));  	Phi[0] = Beta[0];
  Psi = fftwl_malloc(maxit*sizeof(long double));  	memset(Psi, 0, maxit*sizeof(long double));
  Chi = fftwl_malloc(maxit*sizeof(long double));  	Chi[0] = 0.L;
  Gamma[0] = fftwl_malloc(maxit*sizeof(long double));	memset(Gamma[0], 0, maxit*sizeof(long double));
  Gamma[1] = fftwl_malloc(maxit*sizeof(long double)); memset(Gamma[1], 0, maxit*sizeof(long double));
  Epsilon = fftwl_malloc(maxit*sizeof(long double));  memset(Epsilon, 0, maxit*sizeof(long double));	  	  	  	
		
  for (int l = 0; l < 3; l++) {
    v[l] = fftwl_malloc(nC*sizeof(long double));
    d[l] = fftwl_malloc(nC*sizeof(long double));
    memset(v[l], 0, nC*sizeof(long double));
    memset(d[l], 0, nC*sizeof(long double));
  }
  for (long int l = 0; l < nC; l++) v[1][l] = bk[l]/Beta[0];
	
  while (k < maxit) {		
    lanczos_step(v[1], v[0], Beta[k-1], 0.L, &Alpha[k-1], &Beta[k], v[2]);		
    //	last left orthogonalziation on middle two entries in last column of Tk 
    Delta[1][k-1] = c[k-1]*Delta[0][k-1] + s[k-1]*Alpha[k-1];
    Gamma[0][k-1] = s[k-1]*Delta[0][k-1] - c[k-1]*Alpha[k-1];
    //	last left orthogonalization to produce first two entries of Tk+1 ek+1
    Epsilon[k]  = s[k-1]*Beta[k];
    Delta[0][k] = -c[k-1]*Beta[k];
    // current left orthogonalization to zero out Beta[k]
    symortho(Gamma[0][k-1], Beta[k], &c[k], &s[k], &Gamma[1][k-1]);
    // right-hand side , residual norms, and matrix norm
    Tau[k] = c[k]*Phi[k-1];
    Phi[k] = s[k]*Phi[k-1];
    Psi[k-1] = Phi[k-1]*sqrtl(Gamma[0][k-1]*Gamma[0][k-1] + Delta[0][k]*Delta[0][k]);
    if ( k == 1 ) normA = sqrtl(Alpha[0]*Alpha[0] + Beta[1]*Beta[1]);
    else normA = fmaxl(normA, sqrtl(Beta[k-1]*Beta[k-1] + Beta[k]*Beta[k] + Alpha[k-1]*Alpha[k-1]) );
    if (Gamma[1][k-1] != 0.L) {
      for (long int j = 0; j < nC; j++) {
        d[2][j] = (v[1][j]-Delta[1][k-1]*d[1][j]-Epsilon[k-1]*d[0][j])/Gamma[1][k-1];
        out->x[j] = out->x[j] + Tau[k]*d[2][j];
      }
      memcpy(d[0], d[1], nC*sizeof(long double));
      memcpy(d[1], d[2], nC*sizeof(long double));
      Chi[k] = sqrtl(dot_cosine(out->x, out->x));
      Gamma_min = fminl(Gamma_min, Gamma[1][k-1]);
      Kappa = normA/Gamma_min;
    }
    fhlog = fopen("minres.log","a");
    fprintf(fhlog, "%3d\t%.12LE\t%.12LE\n", k, Phi[k], Kappa);
    fclose(fhlog);
    k = k + 1;
    //------------------------------------------
    memcpy(v[0], v[1], nC*sizeof(long double));
    memcpy(v[1], v[2], nC*sizeof(long double));
  }
  out->phi = Phi[k-1];
  out->psi = Phi[k-1]*sqrtl(Gamma[0][k-2]*Gamma[0][k-2] + Delta[0][k-1]*Delta[0][k-1]);
  out->kappa = Kappa;
  out->normA = normA;
  for (long int j = 0; j < nC; j++) out->x[j] = out->x[j]*halfH[j];
  free(c);			free(s);
  free(Delta[0]); 	free(Delta[1]);
  free(Gamma[0]); 	free(Gamma[1]);
  free(Alpha);		free(Beta);
  free(Tau);			free(Phi);
  free(Psi);			free(Chi);
  free(Epsilon);
  for (int l = 0; l < 3; l++) {
    free(v[l]);
    free(d[l]);
  }
}

void lanczos_step(long double *v, long double *vp, long double bk, long double sigma, long double *alpha_k, long double *beta_k, long double *vn) {
  operator_L1_H(v, Av);
  for (long int j = 0; j < nC; j++) {
    tmpk[j] = Av[j] - sigma*v[j];
  }
  *alpha_k = dot_cosine(v, tmpk);
  for (long int j = 0; j < nC; j++) {
    tmpk[j] += - (*alpha_k)*v[j];
    vn[j] = tmpk[j] - bk*vp[j];
  }
  *beta_k = sqrtl(dot_cosine(vn,vn));
  if (*beta_k != 0.L) {
    for (long int j = 0; j < nC; j++) vn[j] = vn[j]/(*beta_k);
  }
}

void symortho(long double a,long double b,long double *c,long double *s,long double *r) {
  long double tau;
  if (b == 0.L) {
    *s = 0.L;
    *r = fabsl(a);
    if (a >= 0.L) *c = 1.L;
    else *c = -1.L; 
  } else if (a == 0.L) {
    *c = 0.L;
    *r = fabsl(b);
    if (b >= 0.L) *s = 1.L;
    else *s = -1.L;
  } else if (fabsl(b) >= fabsl(a)) {
    tau = a/b;
    if (b >= 0.L) {
      *s = 1.L/sqrtl(1.L+tau*tau);
    } else {
      *s = -1.L/sqrtl(1.L+tau*tau);
    }
    *c = (*s)*tau;
    *r = b/(*s);
  } else if (fabsl(a) > fabsl(b)) {
    tau = b/a;
    if (a >= 0.L) {
      *c = 1.L/sqrtl(1.L+tau*tau);
    } else {
      *c = -1.L/sqrtl(1.L+tau*tau);
    }
    *s = (*c)*tau;
    *r = a/(*c);		
  }
}

void gram_schmidt(int n, long double **v, long double *outk) {
   long double a, b, normV;
   for (int j = 0; j < n; j++) {
     a = dot_cosine(v[j], outk);
     b = dot_cosine(v[j], v[j]);
     for (int l = 0; l < nC; l++) {
       outk[l] = outk[l] - (a/b)*v[j][l];
     }
     normV = sqrtl(dot_cosine(outk,outk));
     for (int l = 0; l < nC; l++) outk[l] = outk[l]/normV;
   }
}

void operator_L1_H(long double *ink, long double *outk) {
    long double dBeta;
    memcpy(ink2, ink, nC*sizeof(long double));
    for (long int j = 0; j < nC; j++) ink[j] = halfH[j]*ink[j];
    
    
    //dft(y, yk);
	dft(ink, dy);						// stores dy(x)
	for (long int j = 0; j < nC; j++) {
		tdyk[j] = tk[j]*ink[j];			// changed to kern from j
		dyUq[j]  = dy[j]*dUq[j]*overN;		// new array (not verified)
		yUq[j] = y[j]*dUq[j]*overN;			// new array from Beta
		ydyUq[j]  = y[j]*dyUq[j];			// new array (not verified)
		ty2dy[j] = ty2[j]*dy[j]*overN;
		tydy[j]  = ty[j]*dy[j]*overN;
		ytydy[j] = y[j]*ty[j]*dy[j]*overN;
		ydy[j]   = y[j]*dy[j]*overN;
		y2dy[j]  = y2[j]*dy[j];
	}
	dft(tdyk, tdy);
	dft(tydy, tydyk);
	dft(ytydy, ytydyk);
	dft(y2dy, y2dyk);
	dft(ydy, ydyk);
	dft(ty2dy, ty2dyk);
	
	dft(dyUq, dyUq);	// new array (not verified)
	dft(ydyUq, ydyUq);	// new array (not verified)
	
	for (long int j = 0; j < nC; j++) {
		Tydyk[j]  = tk[j]*ydyk[j];					// changed to kern from j
		ytdy[j]  = y[j]*tdy[j]*overN;
		y2tdy[j] = y[j]*ytdy[j];
	}
	dft(ytdy, ytdyk);
	dft(y2tdy, y2tdyk);
	dft(Tydyk, Tydy);
	
	for (long int j = 0; j < nC; j++) yTydy[j] = y[j]*Tydy[j]*overN;
	dft(yTydy, yTydyk);
	
	dBeta = 2.L*(Ksi - Sigma*Beta)*ink[0] + 2.L*(Sigma*Tau+Ksi*Ksi)*ydyk[0] + Sigma*Ksi*Ksi*y2dyk[0];
	dBeta = dBeta/(1.L + 2.L*Sigma*ink[0]);
	
	for (long int j = 0; j < nC; j++) {
		outk[j] = Beta*(tk[j]*ink[j]-lambda*dyUq[j])
				+  dBeta*(tk[j]*yk[j]-lambda*yUq[j]) 
				- Tau*(tydyk[j] + ytdyk[j] + tk[j]*ydyk[j])
				- 0.5L*Ksi*Ksi*(2.L*ydyUq[j] + tk[j]*y2dyk[j] + 2.L*ytydyk[j] + y2tdyk[j] - 2.L*yTydyk[j] - ty2dyk[j]);
		if (j == 0) outk[j] += 0.5L*dBeta;
		outk[j] = halfH[j]*outk[j];
	}
    memcpy(ink, ink2, nC*sizeof(long double));
}

void find_b0(long double *ink, long double *B0, long double *ML) {
  long double avY = 0.L;
  long double tmp2 = 0.L;
  long double **new_arr;
  long double invD;
 
  new_arr = fftwl_malloc(4*sizeof(long double *));
  for (int j = 0; j < 4; j++) {
     new_arr[j] = fftwl_malloc(nC*sizeof(long double));
  }
  memcpy(new_arr[1], ink, nC*sizeof(long double));
  dft(new_arr[1],new_arr[1]);
  for (long int j = 0; j < nC; j++) {
     new_arr[0][j] = tk[j]*ink[j];						// T'y
     new_arr[2][j] = new_arr[1][j]*new_arr[1][j]*overN;		// y^2
  }
  avY = new_arr[0][0];
  new_arr[0][0] = 0.L;
  dft(new_arr[2], new_arr[2]);
  //dft(ink, new_arr[2]);
  
  //
#if INF_DEPTH  
  //invD = 0.L;
  //ink[0] = -dot_cosine(ink, new_arr[0]); 
#else
  //invD = 1.L/confD;  
  //ink[0] = -0.5L*confD + 0.5L*confD*sqrtl(1.L-4.L*invD*dot_cosine(ink,new_arr[0]));
  ink[0] = -dot_cosine(ink, new_arr[0]); 
  new_arr[0][0] = avY;
#endif
  *ML = dot_cosine(ink, new_arr[0]) + ink[0];
  //tmp1 = dot_cosine(ink, ink);		// <y^2>
  //tmp2 = dot_cosine(ink, new_arr[2]);   // <y^3>

  //*B0 = 2.L/q*(sqrtl(2.L*tau) - invD)*ink[0] + (2.L*tau/q + invD)*tmp1 + 2.L*tau*invD/q*tmp2/3.L;
  //*B0 = (*B0)/(1.L + 2.L*invD*ink[0]);
  
  *B0 = 1.L + 2.L*Ksi*ink[0] + (Sigma*Tau+Ksi*Ksi)*dot_cosine(ink,ink) + Sigma*Ksi*Ksi*dot_cosine(ink,new_arr[2])/3.L;
  *B0 = *B0/(1.L + 2.L*Sigma*ink[0]);

  for (int j = 3; j > -1; j--) {
    fftwl_free(new_arr[j]);
  }
  fftwl_free(new_arr);
}

long double dot_cosine(long double *in1, long double *in2){
	long double tmp = 0.L;
	for (long int j = nC-1; j > -1; j--) {
		tmp += 2.L*in1[j]*in2[j];
	}
	return tmp - in1[0]*in2[0] - in1[nC-1]*in2[nC-1];
}

long double dot(long double *in1, long double *in2){
	long double tmp = 0.L;
	for (long int j = nC-1; j > -1; j--) {
		tmp += 1.L*in1[j]*in2[j];
	}
	return tmp;
}










