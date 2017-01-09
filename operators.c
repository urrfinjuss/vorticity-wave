#include "header.h"

static long double **tmp;
static long double *dft0, *dft1;

// global variables
long int 	nC, nS;
long double	overN, Beta, dBeta;
long double 	*xk, *yk;
long double	*x, *y;
long double 	*tk, *H;	// multipliers for T' and preconditioner
long double 	*dUq;
fftwl_plan	dct, dst;

void allocate_global_memory(params_ptr in) {
  overN = 1.L/in->N;
  nC = (in->N)/2+1; nS = nC-2;
  x  = fftwl_malloc(nS*sizeof(long double));
  y  = fftwl_malloc(nC*sizeof(long double));
  xk = fftwl_malloc(nS*sizeof(long double));
  yk = fftwl_malloc(nC*sizeof(long double));
  tk = fftwl_malloc(nC*sizeof(long double));
  H  = fftwl_malloc(nC*sizeof(long double));
  dUq  = fftwl_malloc(nC*sizeof(long double));
  dft0 = fftwl_malloc(nC*sizeof(long double));
  dft1 = fftwl_malloc(nS*sizeof(long double));
  dct = fftwl_plan_r2r_1d(nC, dft0, dft0, FFTW_REDFT00, FFTW_ESTIMATE);
  dst = fftwl_plan_r2r_1d(nS, dft1, dft1, FFTW_RODFT00, FFTW_ESTIMATE);
}

void allocate_memory() {
  tmp = fftwl_malloc(13*sizeof(long double *));
  for (int j = 0; j < 13; j++) {
    tmp[j] = fftwl_malloc(nC*sizeof(long double));
  }
}

void dft(long double *in, long double *out) {
  memcpy(dft0, in, nC*sizeof(long double));
  fftwl_execute(dct);
  memcpy(out, dft0, nC*sizeof(long double));
}

void dft_sine(long double *in, long double *out) {
  memcpy(dft1, in, nS*sizeof(long double));
  fftwl_execute(dst);
  memcpy(out, dft1, nS*sizeof(long double));
}

void operator_L0_Beta(long double *ink, long double *pk) {
  
}


void operator_L0(long double *ink, long double *pk) {
  long double *y2Uqk = fftwl_malloc(nC*sizeof(long double));
  long double *tyk = fftwl_malloc(nC*sizeof(long double));
  long double *y2k = fftwl_malloc(nC*sizeof(long double));
  long double *y3  = fftwl_malloc(nC*sizeof(long double));
  long double *y3k = fftwl_malloc(nC*sizeof(long double));
  long double *ty2k = fftwl_malloc(nC*sizeof(long double));
  long double *ty3k = fftwl_malloc(nC*sizeof(long double));
    
  long double *yty2 = fftwl_malloc(nC*sizeof(long double));
  long double *yty2k = fftwl_malloc(nC*sizeof(long double));
  long double *y2ty = fftwl_malloc(nC*sizeof(long double));
  long double *y2tyk = fftwl_malloc(nC*sizeof(long double));
  long double *yty = fftwl_malloc(nC*sizeof(long double));
  long double *ytyk = fftwl_malloc(nC*sizeof(long double));
  long double ML, corr;
    
  find_b0(ink, &Beta, &ML);
									// ink stores y_k
  dft(ink, y);					// stores y(x)
  //memcpy(yk, ink);				// stores yk
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
    ty2k[j] = tk[j]*y2k[j];
    ty3k[j] = tk[j]*y3k[j];
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
    pk[j] = Beta*(tk[j]*yk[j] - (Tau+Ksi)*yUqk[j]) - Tau*(ytyk[j] + 0.5L*ty2k[j]) - 0.5L*Ksi*Ksi*(y2Uqk[j] + ty3k[j]/3.L + y2tyk[j] - yty2k[j]);    
  }
  pk[0] = pk[0] + 0.5L*(Beta - 1.L);
  //pk[0] = 0.L;
	
  fftwl_free(tyk);	fftwl_free(yty);	fftwl_free(ytyk);
  fftwl_free(ty2k);
  fftwl_free(y2k);
  fftwl_free(y2ty);		fftwl_free(y2tyk);
  fftwl_free(y3);			fftwl_free(y3k); 		fftwl_free(ty3k);
  fftwl_free(yty2);		fftwl_free(yty2k);
}
