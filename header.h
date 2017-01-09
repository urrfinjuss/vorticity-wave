#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_fit.h>

#define pi acosl(-1.L)
#define INF_DEPTH 0

typedef struct params {
	long double  HL, curv, ml;
	long double c, v, g, k0, D;
	long double L, L_res;
	long double vc, max_angle;
	long int N;
	FILE *fh;
} params, *params_ptr;

typedef struct linear_solver_output {
	long double NV;		// stores nonlinear residual
	long double *x;		// stores solution vector
	long double phi;	// stores 2-norm of r = b - Ax
	long double psi;	// stores 2-norm of Ar
	long double chi;	// stores 2-norm of x
	long double normA;	// stores induced norm of A
	long double kappa;	// stores condition number of A
} linsol, *linsol_ptr;


// global variables
extern long int nC, nS, N;
extern long double overN;
extern long double Tau, Ksi, Sigma;
extern long double *x, *y, *dUq;
extern long double *xk, *yk;
extern long double *tk, *H;
extern long double Beta, dBeta;

// global functions
extern int read_guess(params_ptr in);
extern int init_arrays(params_ptr in);
extern int simulate(params_ptr in);
extern long double dot_cosine(long double *in1, long double *in2);
extern long double dot(long double *in1, long double *in2);

extern void write_FT(char *name, long double *in);
extern void write_FTR(char *name, params_ptr in);
extern void write_PHE(char *name, long double L, long double *in);
extern void write_xy(char *name, params_ptr in);
extern void fourier_reinterp(long double L1, long double L2);
extern void operator_L0(long double *ink, long double *pk);
extern void operator_L1(long double *ink, long double *outk);
extern void operator_L1_H(long double *ink, long double *outk);
extern void operator_L1_H_bad(long double *ink, long double *outk);
extern void gram_schmidt(int n, long double **v, long double *outk);
extern void compute_eigenvalues();
extern void find_spectrum_slope();
extern void find_b0(long double *ink, long double *b0, long double *ml);

extern void compute_rhs(long double *out, linsol_ptr sltn);
extern void compute_params(params_ptr in);
extern void cg_symm(long double *bk, const long int maxit, linsol_ptr out);
extern void minres_classic(long double *bk, long double sigma, const long int maxit, linsol_ptr out);
extern void symortho(long double a, long double b, long double *c, long double *s, long double *r);
extern void lanczos_step(long double *v, long double *vp, long double bk, long double sigma, long double *alpha_k, long double *beta_k, long double *vn);

extern void writer_full(char* str, long double *x, long double *yk, long double *y, long int N);
extern void allocate_global_memory(params_ptr in);
extern void allocate_memory();
extern void dft(long double *in, long double *out);
extern void dft_sine(long double *in, long double *out);


