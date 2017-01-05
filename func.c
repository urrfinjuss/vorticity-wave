#include "header.h"

// Following the dissertation of Sou Cheng Choi 

static long double *yk, *y, *y2, *ty, *ty2, *xk, *x, *yUqk;
static long double *dft_tmp0, *dft_tmp1;
static long double *l0[2], *l0k[2], *p;
static long double du, q, lambda, tau, gb0, confD;
static long double overN, *kern;
static fftwl_plan dct, dst;
static long int nC, nS, N;

static long double *dy, *halfH, *ink2;
static long double *dyUq, *ydyUq;
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
static long double *dUq;
static long double factor;


int read_guess(params_ptr in) {
  char line[512], v1[80], v2[80], v3[80], v4[80], v5[80];
  long double k;
  long int N_res;
  int j = 0;
  
  N = in->N; 
  overN = 1.L/in->N;
  du = 2.L*pi/in->N;
  nC = (in->N)/2+1;
  nS = (in->N)/2-1;
  
  y = fftwl_malloc(nC*sizeof(long double));
  x = fftwl_malloc(nS*sizeof(long double));
  yk = fftwl_malloc(nC*sizeof(long double));
  xk  = fftwl_malloc(nS*sizeof(long double));
  ty = fftwl_malloc(nC*sizeof(long double));
  y2  = fftwl_malloc(nC*sizeof(long double));
  ty2 = fftwl_malloc(nC*sizeof(long double));

  q	= 1.L/powl(in->c,2);  			//	stores g/c^2
  tau = 0.5L*powl(in->v/in->c,2);		//	stores 0.5 gamma^2
  lambda = q + in->v/in->c;			//  stores lambda = g/c^2 + gamma
  factor = 1.L/(1.L + gb0*q);
  confD = in->D;
  //printf("Lambda = %.12LE\n", lambda);
  //printf("Q = %.12LE\n", q);
  //printf("Tau = %.12LE\n", tau);

  
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
    	printf("Restart has more modes than in simulation. Rerun with at least %d Fourier modes.\n", N_res);
    	return 1;
    	break;
    }
  }
  if (j < nC-1) {
  	printf("Found %4d Fourier cosine coefficients (%4d of %4d). Filled the rest with zeros.\n", j-1, j-1, nC);
  	for (int l = j+1; j < nC; j++) yk[l] = 0.Q;
  	return 0;
  } else {
    // verify L0
    //memset(yk, 0, nC*sizeof(long double));
    //yk[1] = 0.5Q;
    //yk[2] = 0.5Q;
  	printf("Restarting from the same number of modes (%4d of %4d).\n", j-1, nC);
  	return 0;
  }
}

int init_arrays(params_ptr in) {
    // initialize internal arrays and FFTW
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
    ydyUq = fftwl_malloc(nC*sizeof(long double));
	
    kern = fftwl_malloc(nC*sizeof(long double));	// for finite-depth
	
    long double A = 2.L*(1.L/in->L)/(1.L + powl(1.L/in->L,2));
    long double B = (1.L - powl(1.L/in->L, 2))/(1.L + powl(1.L/in->L, 2));
    long double u = 0;
    for (long int j = 0; j < nC; j++) {
        u = 2.L*atan2l((in->L)*sinl(0.5L*j*du), cosl(0.5L*j*du));
#if INF_DEPTH
	kern[j] = j;
   	halfH[j] = 1.0L/sqrtl(fabsl(j-lambda));   		
   	dUq[j] = A/(1.L + B*cosl(j*du));
#else	
	if (j == 0)	kern[j] = 1.L/in->D;
	else kern[j] = j/tanhl(j*in->D);
	halfH[j] = 1.0L/sqrtl(fabsl(kern[j]-lambda));   				   		
   	dUq[j] = A/(1.L + B*cosl(j*du));
#endif
    }    
    dft_tmp0 = fftwl_malloc(nC*sizeof(long double));
    dft_tmp1 = fftwl_malloc(nS*sizeof(long double));
    dct = fftwl_plan_r2r_1d(nC, dft_tmp0, dft_tmp0, FFTW_REDFT00, FFTW_ESTIMATE);
    dst = fftwl_plan_r2r_1d(nS, dft_tmp1, dft_tmp1, FFTW_RODFT00, FFTW_ESTIMATE);  

    return 0;
}

void dft(long double *in, long double *out) {
    // execute Fourier transform
    memcpy(dft_tmp0, in, nC*sizeof(long double));
    fftwl_execute(dct);
    memcpy(out, dft_tmp0, nC*sizeof(long double));
}

void dft_sine(long double *in, long double *out) {
    memcpy(dft_tmp1, in, nS*sizeof(long double));
    fftwl_execute(dst);
    memcpy(out, dft_tmp1, nS*sizeof(long double));
}

void fourier_reinterp(long double L1, long double L2) {
	/*long double *tmp_arrayF, *tmp_arrayP;
	tmp_arrayF = fftwl_malloc(nC*sizeof(long double));
	tmp_arrayP = fftwl_malloc(nC*sizeof(long double));
	
	memcpy(tmp_arrayF, yk, nC*sizeof(long double));
	memcpy(tmp_arrayP, y, nC*sizeof(long double));*/
	// -----  verification
	//dft(yk, y);
	//write_PHE("y_L1.txt", L1, y);
	//write_PHE("dUq_L1.txt", L1, dUq);
	//write_FT("yk_L1.txt", yk);
	// -----  end verification
	memset(y, 0, nC*sizeof(long double));
	for (long int j = 0; j < nC; j++) {
		for (long int l = 0; l < nC; l++) {
			y[j] += 2.L*yk[l]*cosl(2.L*l*atan2l( L1*sinl(0.5L*j*du), L2*cosl(0.5L*j*du) ));
		}
		y[j] = y[j] - yk[0];
	}
	dft(y,yk);
	for (long int j = 0; j < nC; j++) yk[j] = yk[j]*overN;
	// -----  verification
	//write_FT("yk_L2.txt", yk);
	//write_PHE("y_L2.txt", L2, y);
	//write_PHE("dUq_L2.txt", L2, dUq);
	// -----  end verification
}

void write_FT(char *name, long double *in) {
    FILE *fh = fopen(name,"w");
    fprintf(fh, "# 1. k 2. Array (Fourier Side)\n\n");
    for (long int j = 0; j < nC; j++) {
    	fprintf(fh, "%d\t%+.15LE\n", j, in[j]);
    }
    fclose(fh);
}

void write_PHE(char *name, long double L, long double *in) {
	long double u;
    FILE *fh = fopen(name,"w");
    fprintf(fh, "# 1. u 2. Array (Phys Side)\n\n");
    for (long int j = 0; j < nC; j++) {
    	u = 2.L*atan2l(L*sinl(0.5*(j*du)), cosl(0.5*(j*du)));
    	fprintf(fh, "%.15LE\t%+.15LE\n", u, in[nC-1-j]);
    }
    fclose(fh);
}

void write_PHO(char *name, long double L, long double *in) {
	long double u;
    FILE *fh = fopen(name,"w");
    fprintf(fh, "# 1. u 2. Array (Phys Side)\n\n");
   	fprintf(fh, "%.15LE\t%+.15LE\n", 0.L, 0.L);
    for (long int j = 1; j < nS; j++) {
       	u = 2.L*atan2l(L*sinl(0.5*(j*du)), cosl(0.5*(j*du)));
    	fprintf(fh, "%.15LE\t%+.15LE\n", u, in[nS-j]);
    }
    fclose(fh);
}

void write_FTR(char *name, params_ptr in){
	FILE *fh = fopen(name, "w");
    fprintf(fh, "# 1. k 2. |yk| 3. yk\n");
    fprintf(fh, "# N = %ld\tL = %.19LE\tc = %.15LE\tOmega = %.15LE\tDepth = %.15LE\n\n", N, in->L, in->c, in->v, in->D);
 	for (long int j = 0; j < nC; j ++) {
		fprintf(fh, "%+ld\t%+.19LE\t%+.19LE\n", j, fabsl(yk[j]), yk[j]);
 	}
 	fclose(fh);
}

void write_xy(char *name, params_ptr in){
	long double u;
	FILE *fh = fopen(name, "w");
    fprintf(fh, "# 1. u 2. x 3. y\n");
    fprintf(fh, "# N = %ld\tL = %.19LE\tc = %.15LE\tOmega = %.15LE\tDepth = %.15LE\n\n", N, in->L, in->c, in->v, in->D);
    fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", -pi, 0.L, y[0]);
 	for (long int j = 1; j < N; j ++) {
       	u = 2.L*atan2l((in->L)*sinl(0.5*(-pi+j*du)), cosl(0.5*(-pi+j*du)));
 		if (j < N/2) fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u, x[j-1], y[j]);
 		else if (j == N/2) fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u,  0.L, y[nC-1]);
 		else fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u, -x[nS-(j-N/2)], y[nC-1-(j-N/2)]);
 	}
 	fclose(fh);
}

void write_z(char *name, params_ptr in){
	long double u;
	long double *dx = fftwl_malloc(nC*sizeof(long double));
	long double *dy = fftwl_malloc(nS*sizeof(long double));
	
	// compute dz
	memset(dx, 0, nC*sizeof(long double));
	memset(dy, 0, nS*sizeof(long double));
	for (long int j = 0; j < nS; j++){
		dy[j] = -(j+1)*yk[j+1];
		dx[j+1] = (j+1)*xk[j];
	}
	dx[0] = 0.L;
	dft_sine(dy, dy);
	dft(dx, dx);
	for (long int j = 0; j < nS; j++) dy[j] = dy[j]/dUq[j+1];
	for (long int j = 0; j < nC; j++) dx[j] = dx[j]/dUq[j];
	//  end compute dz
	
	FILE *fh = fopen(name, "w");
    fprintf(fh, "# 1. u 2. |z|^2 3. |z_u|^2\n");
    fprintf(fh, "# N = %ld\tL = %.19LE\tc = %.15LE\tOmega = %.15LE\tDepth = %.15LE\n\n", N, in->L, in->c, in->v, in->D);
    fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", -pi, y[0]*y[0], dx[0]*dx[0]);
 	for (long int j = 1; j < N; j ++) {
       	u = 2.L*atan2l((in->L)*sinl(0.5*(-pi+j*du)), cosl(0.5*(-pi+j*du)));
 		if (j < N/2) fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u, x[j-1]*x[j-1] + y[j]*y[j], dy[j-1]*dy[j-1] + dx[j]*dx[j]);
 		else if (j == N/2) fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u,  y[nC-1]*y[nC-1], dx[nC-1]*dx[nC-1]);
 		else fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u, x[nS-(j-N/2)]*x[nS-(j-N/2)] + y[nC-1-(j-N/2)]*y[nC-1-(j-N/2)], 
 						 dx[nC-1-(j-N/2)]*dx[nC-1-(j-N/2)] + dy[nS-(j-N/2)]*dy[nS-(j-N/2)]);
 	}
 	fclose(fh);
}


int simulate(params_ptr in) {
    //long double NV = 1.L;
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
        gb0 = 0.L;
	find_b0(yk, &(in->gb0), &(in->ml)); 
        gb0 = in->gb0;
	compute_rhs(bk, &sltn);
	printf("At Newton Stage %d: L. Res:\t%.6LE\tNL Res:\t%.6LE\tb0:\t%.12LE\tML:\t%.6LE\n", 0, 0.L, sltn.NV, in->gb0, in->ml); 
	int it_counter = 0, Max_Newton_Steps = 20000; 	
	while (it_counter < Max_Newton_Steps) {
  	  minres_classic(bk, 0.L, 24, &sltn);
	  //cg_symm(bk, 64, &sltn);
	  for (long int j = 0; j < nC; j++) yk[j] = yk[j] + sltn.x[j];
            
	    find_b0(yk, &(in->gb0), &(in->ml)); 
	    gb0 = in->gb0;
	    compute_rhs(bk, &sltn);	
	    printf("At Newton Stage %d: L. Res:\t%.6LE\tNL Res:\t%.6LE\tb0:\t%.12LE\tML:\t%.6LE\n", it_counter+1, sltn.phi, sltn.NV, in->gb0, in->ml);
	    if (sltn.NV < 2.0E-14L*sqrtl(in->N/131072.L)) {
		printf("Converged! Newton Stage %d: NL Res:\t%.12LE\n", it_counter+1, sltn.NV);
	        break;
	    }
	    it_counter++;
	  }
	 // restore X:
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
    
    FILE *fhsteep = fopen("steep.txt","r");
    if (fhsteep == NULL) {
    	printf("File steep.txt not found. Creating new one.\n");
    	fhsteep = fopen("steep.txt","w");
    	fprintf(fhsteep, "# 1. H/L 2. c 3. Curvature (conformal) 4. Max Angle 5. vorticity 6. Depth(Conf) 7. Depth(Phys)\n\n");
    	fclose(fhsteep);
    }
   	printf("Writing Wave parameters to steep.txt\n");
    fhsteep = fopen("steep.txt","a");
    fprintf(fhsteep, "%.18LE\t%.18LE\t%.18LE\t%.18LE\t%.18LE\t%.18LE\t%.18LE\n", in->HL, in->c, in->curv, in->max_angle, in->v, in->D, in->D + yk[0]);
    fclose(fhsteep);
    

    
    sprintf(name, "./library/phys/R_%.6Lf.phys.txt", 1.L/sqrtl(in->curv));
    sprintf(fname, "./library/spec/R_%.6Lf.spec.txt", 1.L/sqrtl(in->curv));
    printf("Writing result to:\n%s\n%s\n", name, fname);
    printf("Physical Depth is(reg) \t\t%.12LE\n",  in->D + yk[0]);
    printf("Physical Depth(alt) is \t\t%.12LE\n",  in->D - yk[0]);
    printf("Height of the wave \t\t%.12LE\n", y[nC-1]-y[0]);
	/*write_PHE("y.txt", in->L, y);
	write_PHO("x.txt", in->L, x);
	write_xy("xy.txt", in);
	write_z("z.txt", in);*/
	write_FTR(fname, in);
	write_xy(name, in);
	
	find_spectrum_slope(in);
	//fourier_reinterp(in->L, 0.5L);
	//writer_full("wave.txt", x, yk, y, 2*(nS+1));
	//compute_rhs(pk, bk, &sltn);	
	//printf("Magnitude of Residual: %.12LE\n", sltn.NV);
	
	// what's the spectrum of the linearized operator? 
	// let's find out! initialize v0
	// compute_eigenvalues(); // run this and find out
	
}


void compute_params(params_ptr in) {
	long double *dx = fftwl_malloc(nC*sizeof(long double));
	long double *dy = fftwl_malloc(nS*sizeof(long double));
	long double *dEta = fftwl_malloc(nC*sizeof(long double));
	long double *d2x = fftwl_malloc(nS*sizeof(long double));
	long double *d2y = fftwl_malloc(nC*sizeof(long double));
	long double Y[3], X[3], dX;
	long double A, B;
	long double xs, ys;

	// verification
	//memset(xk, 0, nS*sizeof(long double));
	//memset(yk, 0, nC*sizeof(long double));
	//yk[1] = 1.L; yk[2] = 1.L;
	//xk[0] = 1.L; xk[1] = 1.L;
	//dft_sine(xk, x);
	//dft(yk,y);
	// verification
	//for (long int j = 0; j < nC; j++) d2y[j] = -powl(j,2)*yk[j];
	//for (long int j = 0; j < nS) d2x[j] = -powl(j+1,2)*xk[j];
	//dft(d2y, d2y);	
	//d2y[nC-1] = d2y[nC-1]/powl(in->L,2);
	
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
	// compute ratio for vc
	//long double vc = -d2y[nC-1]/dx[nC-1];
	//printf("Estimate for V_c = %.12LE\n", vc);
	
	
	//write_PHE("dx.txt", in->L, dx);
	//write_PHO("dy.txt", in->L, dy);
	//write_PHE("dEta.txt", in->L, dEta);
	

	ys = y[nC-1]-y[nC-2];
	xs = (in->L)*du;
	//printf("Estimate points after scaling:\n");
	//long double dZ2;
	for (long int j = 0; j < 3; j++) {
		//dZ2 = (dy[nC-1-j]*dy[nC-1-j] + dx[nS-1-j]*dx[nS-1-j]);
		Y[j] = (y[nC-1]-y[nC-1-j])/ys;
		X[j] = 2.L*atan2l((in->L)*sinl(0.5*(j*du)), cosl(0.5*(j*du)))/xs;
		//printf("(%.7LE, %.7LE)\n", X[j], Y[j]);
	}
	//dX = -0.5L*x[nS-1]/atan2l((in->L)*sinl(0.5*du), cosl(0.5*du)); 
	//printf("dX = %.12LE\n", dX);
	
	B = Y[2]*X[1]*X[1] - Y[1]*X[2]*X[2];
	B = B/(X[1]*X[1]*X[2]*X[2]*(X[2]*X[2] - X[1]*X[1]));
	A = Y[1]/(X[1]*X[1]) - B*(X[1]*X[1]);
	
	in->curv = 2.0L*A*ys/powl(xs,2);
	in->max_angle = 180.L*atanl(1.L/sqrtl(3.L)-MinEtax)/pi;
    in->HL = 0.5L*(y[nC-1]-y[0])/pi;
	//find_spectrum_slope(in);
	printf("Crest Curvature = %.19LE\n", in->curv);
	//printf("Estimate for vc = %.12LE\n", 0.5*dX/curv);
	//printf("Higher Order = %.19LE\n", B*ys/powl(xs,4));
	
    /*printf("Estimate points after scaling:\n");
   	for (long int j = 0; j < 3; j++) {
   		Y[j] = (Y[0]-Y[j])/ys;
   		X[j] = X[j]/xs
   		printf("(%.7LE, %.7LE)\n", X[j]/xs, (Y[j]-Y[0])/ys);
   	}*/
    
	
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
  	  //FILE *fh_test = fopen("fitting.txt","w");
  	  for (long int j = IndMin; j < IndMax; j++) {
  	  	LY[j-IndMin] = log(fabs(yk[j]));
  	  	LX[j-IndMin] = 1.*j;
  	  	//fprintf(fh_test,"%.12e\t%.12e\n", 1.*j, log(fabs(yk[j])));
  	  }
  	  //fclose(fh_test);
  	  gsl_fit_linear(LX, 1, LY, 1, Nr, &c0, &c1, &cov00, &cov01, &cov11, &sumsq);
	  printf("c0 = %.8E\nc1 = %.8E\nFitting Error = %.8E\n", c0, c1*sqrt(in->L), sumsq/Nr);
	}
}

void compute_rhs(long double *out, linsol_ptr sltn) {
	operator_L0(yk, out);
	sltn->NV = sqrtl(dot_cosine(out,out));
	for (long int j = 0; j < nC; j++) {
		out[j] = -halfH[j]*out[j]; // /(j-lambda);
		if ( fabsl(j-lambda) < 1.0e-10L) printf("Warning! Small denominator in preconditioner\n");
	}
}


void cg_symm(long double *bk, const long int maxit, linsol_ptr out) {
	// Solve M^-1/2 L1 M^-1/2 dz = - M^-1/2 L0 with CG

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
		/*if (Xi[k-1] <= 0){
			memset(out->x, 0, nC*sizeof(long double));
			Phi[k-1] = Beta[0];
			Chi[k-1] = 0.L;
			break;
		}*/
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

	// write_FT("starting_rhs.txt", bk);
	// verify that MINRES is converging for simple problem: (yes)
	// for (long int j = 0; j < nC; j++) bk[j] = cos(2.L*j*du);
	
	FILE *fhlog = fopen("minres.log","w");
	fprintf(fhlog, "# 1. iter 2. Norm Residual 3. Norm Residual (Direct) 4. Norm Solution\n\n");
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
				d[2][j] = (v[1][j] - Delta[1][k-1]*d[1][j] - Epsilon[k-1]*d[0][j])/Gamma[1][k-1];
				out->x[j] = out->x[j] + Tau[k]*d[2][j];
			}
			memcpy(d[0], d[1], nC*sizeof(long double));
			memcpy(d[1], d[2], nC*sizeof(long double));
			Chi[k] = sqrtl(dot_cosine(out->x, out->x));
			Gamma_min = fminl(Gamma_min, Gamma[1][k-1]);
			Kappa = normA/Gamma_min;
			// verification
			//sprintf(name, "iter_%02d.txt", k);
			//write_FT(name, out->x);
		}
		//printf("Iteration %3d:\t%.8LE\t%.8LE\n", k, Phi[k], Kappa);
		
		//------------------------------------------
		//			Remove to speed-up
		/*
		operator_L1_H(out->x, Av);
		for (long int j = 0; j < nC; j++) Av[j] = Av[j] - bk[j];
		normDIRECT = sqrtl(dot_cosine(Av,Av));
		*/
		//------------------------------------------
		fhlog = fopen("minres.log","a");
		fprintf(fhlog, "%3d\t%.12LE\t%.12LE\n", k, Phi[k], Kappa);
		fclose(fhlog);
		k = k + 1;
		/*if (Phi[k] < 1.0E-16) {
			printf("MINRES Stage Complete at %ld iteration: LIN RES = %.6LE\n", k, Phi[k]);
			break;
		}*/
		//------------------------------------------
		memcpy(v[0], v[1], nC*sizeof(long double));
		memcpy(v[1], v[2], nC*sizeof(long double));
	}
	out->phi = Phi[k-1];
	out->psi = Phi[k-1]*sqrtl(Gamma[0][k-2]*Gamma[0][k-2] + Delta[0][k-1]*Delta[0][k-1]);
	out->kappa = Kappa;
	out->normA = normA;
	for (long int j = 0; j < nC; j++) out->x[j] = out->x[j]*halfH[j];
	//write_FT("solution.txt", out->x);
	//printf("Norm Residual:\t%.12LE\n", out->phi);
	//printf("Norm Solution:\t%.12LE\n", out->kappa);
	//printf("Norm Matrix:\t%.12LE\n", out->normA);
	//printf("Condition Number:\t%.12LE\n", out->kappa);
	
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



void symortho(long double a, long double b, long double *c, long double *s, long double *r) {
	// not verified
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
   //memset(outk, 0, nC*sizeof(long double));
   //outk[1] = 1.L; //sqrtl(0.5L);
   
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


void compute_eigenvalues() {
    const int n_eigs = nC;
	long double normV = 1.L, eig = 0.L, eig_p = 1.L, num;
	int j;
	char name[80];
	
	long double *pk  = fftwl_malloc(nC*sizeof(long double));
    long double *v[n_eigs];
    long double *Apk  = fftwl_malloc(nC*sizeof(long double));
	
    for (int r = 0; r < n_eigs; r++) v[r] = fftwl_malloc(nC*sizeof(long double));	
	FILE *fh = fopen("eigenvalues.txt","w");
	fprintf(fh, "# 1. iter 2. eigenvalue\n\n");
	fclose(fh);

	memset(pk, 0, nC*sizeof(long double));
    pk[1] = sqrtl(0.5L);
	for (int r = 0; r < n_eigs; r++) {
		printf("Computing eigenvalue %3d:\n", r);
		j = 0;
		while(j < 2048) {
			operator_L1(pk, Apk);
			normV = dot_cosine(pk,pk);
			num = dot_cosine(pk,Apk);
			eig_p = eig;
			eig = num/normV; 
			j++;
			for (long int l = 0; l < nC; l++) {
			  pk[l] = Apk[l]/sqrtl(normV);
			}
			gram_schmidt(r, v, pk);
			if ((j % 256) == 0) printf("Iteration %4d: Eigenvalue\t\t%.19LE\n", j, eig);
			//if (abs(eig_p-eig) < 1e-8*eig) break;
			
			//sprintf(name, "power_%02d.txt", j);
			//write_FT(name, pk);
		}
		
		fh = fopen("eigenvalues.txt","a");
		fprintf(fh, "%d\t%.12LE\n", r, eig);
		fclose(fh);
		
		// update library of eigenvectors and call Gram-Schmidt
		printf("Writing eigenvector %d\n", r);
		memcpy(v[r], pk, nC*sizeof(long double));	// v[r] holds r-th eigenvector
		gram_schmidt(r, v, pk);
		
		sprintf(name, "eigenvector_%02d.txt", r);
		write_FT(name, v[r]);

	    
		//printf("Complete\n");
	}

}


void operator_L0(long double *ink, long double *pk) {
    //long double *ty  = fftwl_malloc(nC*sizeof(long double));
    long double *y2Uqk = fftwl_malloc(nC*sizeof(long double));
    long double *tyk = fftwl_malloc(nC*sizeof(long double));
    //long double *y2  = fftwl_malloc(nC*sizeof(long double));
    long double *y2k = fftwl_malloc(nC*sizeof(long double));
    long double *y3  = fftwl_malloc(nC*sizeof(long double));
    long double *y3k = fftwl_malloc(nC*sizeof(long double));
    //long double *ty2  = fftwl_malloc(nC*sizeof(long double));
    long double *ty2k = fftwl_malloc(nC*sizeof(long double));
    long double *ty3k = fftwl_malloc(nC*sizeof(long double));
    
    long double *yty2 = fftwl_malloc(nC*sizeof(long double));
    long double *yty2k = fftwl_malloc(nC*sizeof(long double));
    long double *y2ty = fftwl_malloc(nC*sizeof(long double));
    long double *y2tyk = fftwl_malloc(nC*sizeof(long double));
    long double *yty = fftwl_malloc(nC*sizeof(long double));
    long double *ytyk = fftwl_malloc(nC*sizeof(long double));
    long double ML, corr;
    
    find_b0(ink, &gb0, &ML);
    corr = 1.L/(1.L + q*gb0);
							// ink stores y_k
    dft(ink, y);					// stores y(x)
    for (long int j = 0; j < nC; j++) {
	tyk[j] = j*ink[j];
	yUqk[j] = y[j]*dUq[j]*overN;		// new array (not verified)
	y2Uqk[j] = y[j]*yUqk[j];		// new array (not verified)
	y2[j] = y[j]*y[j]*overN;		// stores y^2
	y3[j] = y2[j]*y[j];				// stores y^3
    }
    dft(tyk, ty);					// l0k[0] stores  k yk
    dft(y2, y2k);					// l0k[1] stores  FT[y^2]
    dft(y3, y3k);					// l0k[2] stores  FT[y^3]
    dft(yUqk, yUqk);				// new array (not verified)
    dft(y2Uqk, y2Uqk);				// new array (not verified)
    for (long int j = 0; j < nC; j++) {
 	ty2k[j] = kern[j]*y2k[j];
	ty3k[j] = kern[j]*y3k[j];
    }
    dft(ty2k,ty2);
    for (long int j = 0; j < nC; j++) {
	yty2[j] = ty2[j]*y[j]*overN;
	y2ty[j] = y2[j]*ty[j];
	yty[j] = ty[j]*y[j]*overN;
    }
    dft(yty, ytyk);
    dft(yty2, yty2k);
    dft(y2ty, y2tyk);
    for (long int j = 0; j < nC; j++) {
       pk[j] = kern[j]*yk[j] - corr*lambda*yUqk[j] - corr*q*(ytyk[j] + 0.5L*ty2k[j]) - tau*corr*(y2Uqk[j] + ty3k[j]/3.L + y2tyk[j] - yty2k[j]);    
    }
    pk[0] = pk[0] + 0.L*q*gb0*corr;
    pk[0] = 0.L;
    //printf("Zero Mode of L0 is %.18LE\t%.18LE\n", pk[0], 0.5L*q*gb0*corr);
	/*write_FT("tyk.txt", tyk);
	write_FT("ytyk.txt", ytyk);
	write_FT("ty2k.txt", ty2k);
	write_FT("y2k.txt",  y2k);
	write_FT("ty3k.txt",  ty3k);
	write_FT("y2tyk.txt",  y2tyk);
	write_FT("yty2k.txt",  yty2k);*/
	
	fftwl_free(tyk);	fftwl_free(yty);	fftwl_free(ytyk);
	fftwl_free(ty2k);

	fftwl_free(y2k);
	fftwl_free(y2ty);		fftwl_free(y2tyk);
	fftwl_free(y3);			fftwl_free(y3k); 		fftwl_free(ty3k);
	fftwl_free(yty2);		fftwl_free(yty2k);
	
}

/*void operator_L1_verify(long double *ink, long double *outk) {
	
	for (long int j = 1; j < nC-1; j++) {
		outk[j] = (ink[j-1]-2.L*ink[j]+ink[j+1])/(du*du);
	}
	outk[0] = -2.L*(ink[0] - ink[1])/(du*du);
	outk[nC-1] = -2.L*(ink[nC-1] - ink[nC-2])/(du*du);	
}*/


void operator_L1(long double *ink, long double *outk) {
    
	dft(ink, dy);						// stores dy(x)
	for (long int j = 0; j < nC; j++) {
		tdyk[j] = kern[j]*ink[j];		// changed to kern
			
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
	
	for (long int j = 0; j < nC; j++) {
		//Ty2dyk[j] = j*y2dyk[j]*overN;  // what is this for?
		Tydyk[j]  = kern[j]*ydyk[j];			// changed to kern
	
		ytdy[j]  = y[j]*tdy[j]*overN;
		y2tdy[j] = y[j]*ytdy[j];
	}
	dft(ytdy, ytdyk);
	dft(y2tdy, y2tdyk);
	dft(Tydyk, Tydy);
	
	for (long int j = 0; j < nC; j++) yTydy[j] = y[j]*Tydy[j]*overN;
	dft(yTydy, yTydyk);
	
	for (long int j = 0; j < nC; j++) {
		//outk[j] = ((j-lambda)*ink[j] - q*(tydyk[j] + ytdyk[j] + j*ydyk[j]) - tau*(2.L*ydyk[j] + j*y2dyk[j] + 2.L*ytydyk[j] + y2tdyk[j] - 2.L*yTydyk[j] - ty2dyk[j])); // /(j-lambda);
		outk[j] = ((kern[j]-lambda)*ink[j] - q*(tydyk[j] + ytdyk[j] + kern[j]*ydyk[j]) - tau*(2.L*ydyk[j] + kern[j]*y2dyk[j] + 2.L*ytydyk[j] + y2tdyk[j] - 2.L*yTydyk[j] - ty2dyk[j])); // /(j-lambda);
	}
	
	/*write_FT("tydyk.txt", tydyk);
	write_FT("ytdyk.txt", ytdyk);
	write_FT("ydyk.txt", ydyk);
	write_FT("y2dyk.txt",  y2dyk);
	write_FT("ytydyk.txt",  ytydyk);
	write_FT("y2tdyk.txt",  y2tdyk);
	write_FT("yTTydyk.txt",  yTydyk);
	write_FT("ty2dyk.txt",  ty2dyk);*/

}


void operator_L1_H(long double *ink, long double *outk) {
    long double uLambda = lambda/(1.L+q*gb0);    
    memcpy(ink2, ink, nC*sizeof(long double));
    for (long int j = 0; j < nC; j++) ink[j] = halfH[j]*ink[j];
    
	dft(ink, dy);						// stores dy(x)
	for (long int j = 0; j < nC; j++) {
		tdyk[j] = kern[j]*ink[j];				// changed to kern from j
		
		dyUq[j]  = dy[j]*dUq[j]*overN;	// new array (not verified)
		ydyUq[j]  = y[j]*dyUq[j];		// new array (not verified)
		
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
		//Ty2dyk[j] = j*y2dyk[j]*overN;  // what is this for?
		Tydyk[j]  = kern[j]*ydyk[j];					// changed to kern from j
	
		ytdy[j]  = y[j]*tdy[j]*overN;
		y2tdy[j] = y[j]*ytdy[j];
	}
	dft(ytdy, ytdyk);
	dft(y2tdy, y2tdyk);
	dft(Tydyk, Tydy);
	
	for (long int j = 0; j < nC; j++) yTydy[j] = y[j]*Tydy[j]*overN;
	dft(yTydy, yTydyk);
	
	for (long int j = 0; j < nC; j++) {
		//outk[j] = halfH[j]*((j-lambda)*ink[j] - q*(tydyk[j] + ytdyk[j] + j*ydyk[j]) - tau*(2.L*ydyk[j] + j*y2dyk[j] + 2.L*ytydyk[j] + y2tdyk[j] - 2.L*yTydyk[j] - ty2dyk[j])); // /(j-lambda);
		//outk[j] = halfH[j]*(j*ink[j] -lambda*dyUq[j] - q*(tydyk[j] + ytdyk[j] + j*ydyk[j]) - tau*(2.L*ydyUq[j] + j*y2dyk[j] + 2.L*ytydyk[j] + y2tdyk[j] - 2.L*yTydyk[j] - ty2dyk[j])); // /(j-lambda);
		//outk[j] = halfH[j]*(kern[j]*ink[j] -lambda*dyUq[j] - q*(tydyk[j] + ytdyk[j] + kern[j]*ydyk[j]) - tau*(2.L*ydyUq[j] + kern[j]*y2dyk[j] + 2.L*ytydyk[j] + y2tdyk[j] - 2.L*yTydyk[j] - ty2dyk[j])); // /(j-lambda);
		outk[j] = halfH[j]*(kern[j]*ink[j]-uLambda*dyUq[j] - q/(1.L+q*gb0)*(tydyk[j] + ytdyk[j] + kern[j]*ydyk[j]) - tau/(1.L+q*gb0)*(2.L*ydyUq[j] + kern[j]*y2dyk[j] + 2.L*ytydyk[j] + y2tdyk[j] - 2.L*yTydyk[j] - ty2dyk[j])); // /(j-lambda);
	}
    memcpy(ink, ink2, nC*sizeof(long double));
	/*write_FT("tydyk.txt", tydyk);
	write_FT("ytdyk.txt", ytdyk);
	write_FT("ydyk.txt", ydyk);
	write_FT("y2dyk.txt",  y2dyk);
	write_FT("ytydyk.txt",  ytydyk);
	write_FT("y2tdyk.txt",  y2tdyk);
	write_FT("yTTydyk.txt",  yTydyk);
	write_FT("ty2dyk.txt",  ty2dyk);*/

}

void find_b0(long double *ink, long double *B0, long double *ML) {
  long double tmp1 = 0.L;
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
     new_arr[0][j] = kern[j]*ink[j];
     new_arr[2][j] = new_arr[1][j]*new_arr[1][j]*overN;
  }
  tmp1 = new_arr[0][0];
  new_arr[0][0] = 0.L;
  dft(new_arr[2], new_arr[2]);
  //dft(ink, new_arr[2]);
  
  //
#if INF_DEPTH  
  invD = 0.L;
  ink[0] = -dot_cosine(ink, new_arr[0]); 
#else
  invD = 1.L/confD;  
  ink[0] = -0.5L*confD + 0.5L*confD*sqrtl(1.L-4.L*invD*dot_cosine(ink,new_arr[0]));
  //ink[0] = -dot_cosine(ink, new_arr[0]); 
  new_arr[0][0] = tmp1;
#endif
  *ML = dot_cosine(ink, new_arr[0]) + ink[0];
  tmp1 = dot_cosine(ink, ink);		// <y^2>
  tmp2 = dot_cosine(ink, new_arr[2]);   // <y^3>

  *B0 = 2.L/q*(sqrtl(2.L*tau) - invD)*ink[0] + (2.L*tau/q + invD)*tmp1 + 2.L*tau*invD/q*tmp2/3.L;
  *B0 = (*B0)/(1.L + 2.L*invD*ink[0]);

  for (int j = 3; j > -1; j--) {
    fftwl_free(new_arr[j]);
  }
  fftwl_free(new_arr);
       /* for (long int j = 0; j < nC; j++) {
		y2dy[j] = ydy[j]*ydy[j]*ydy[j]*overN;
   		ydy[j]  = ydy[j]*ydy[j]*overN;
	}
        dft(ydy, tydy);			// stores FT y^2
        dft(y2dy, ty2dy);		// stores FT y^3
    
        long double overD = 1.L/confD;
#if INF_DEPTH 
	overD = 0.L;         
#endif
	tmp = 2.L*q*(sqrtl(2.L*tau) - overD)*ink[0] + (2*tau/q + overD)*tydy[0] + (2.L/3.L)*tau*ty2dy[0]*overD;
	tmp = tmp/(1.L + 2.L*ink[0]*overD);
	*B0 = tmp;
	*ML = tmp2;*/
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












