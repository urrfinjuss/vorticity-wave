#include "header.h"

void write_FT(char *name, long double *in) {
    FILE *fh = fopen(name,"w");
    fprintf(fh, "# 1. k 2. Array (Fourier Side)\n\n");
    for (long int j = 0; j < nC; j++) {
    	fprintf(fh, "%d\t%+.15LE\n", j, in[j]);
    }
    fclose(fh);
}

void write_PHE(char *name, long double L, long double *in) {
    long double u, du = 2.L*pi*overN;
    FILE *fh = fopen(name,"w");
    fprintf(fh, "# 1. u 2. Array (Phys Side)\n\n");
    for (long int j = 0; j < nC; j++) {
    	u = 2.L*atan2l(L*sinl(0.5*(j*du)), cosl(0.5*(j*du)));
    	fprintf(fh, "%.15LE\t%+.15LE\n", u, in[nC-1-j]);
    }
    fclose(fh);
}

void write_PHO(char *name, long double L, long double *in) {
    long double u, du = 2.L*pi*overN;
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
    fprintf(fh, "# N = %ld\tL = %.19LE\tc = %.15LE\t", N, in->L, in->c);
    fprintf(fh, "Omega = %.15LE\tDepth = %.15LE\n\n", in->v, in->D);
    for (long int j = 0; j < nC; j ++) {
        fprintf(fh, "%+ld\t%+.19LE\t%+.19LE\n", j, fabsl(yk[j]), yk[j]);
    }
    fclose(fh);
}

void write_xy(char *name, params_ptr in){
    long double u, du = 2.L*pi*overN;
    FILE *fh = fopen(name, "w");
    fprintf(fh, "# 1. u 2. x 3. y\n");
    fprintf(fh, "# N = %ld\tL = %.19LE\tc = %.15LE\t", N, in->L, in->c);
    fprintf(fh, "Omega = %.15LE\tDepth = %.15LE\n\n", in->v, in->D);
    fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", -pi, 0.L, y[0]);
    for (long int j = 1; j < N; j ++) {
       	u = 2.L*atan2l((in->L)*sinl(0.5*(-pi+j*du)), cosl(0.5*(-pi+j*du)));
 	if (j < N/2) fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u, x[j-1], y[j]);
 	else if (j == N/2) fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u,  0.L, y[nC-1]);
 	else fprintf(fh,"%+.12LE\t%+.12LE\t%+.12LE\n", u, -x[nS-(j-N/2)], y[nC-1-(j-N/2)]);
 	}
    fclose(fh);
}

void write_z(char *name, params_ptr in){
     long double ui, u, du = 2.L*pi*overN;
     long double *dx = fftwl_malloc(nC*sizeof(long double));
     long double *dy = fftwl_malloc(nS*sizeof(long double));
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
     FILE *fh = fopen(name, "w");
     fprintf(fh, "# 1. u 2. |z|^2 3. |z_u|^2\n");
     fprintf(fh, "# N = %ld\tL = %.19LE\tc = %.15LE\t", N, in->L, in->c);
     fprintf(fh, "Omega = %.15LE\tDepth = %.15LE\n\n", in->v, in->D);
     fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", -pi, y[0]*y[0], dx[0]*dx[0]);
     for (long int j = 1; j < N; j ++) {
       	u = 2.L*atan2l((in->L)*sinl(0.5*(-pi+j*du)), cosl(0.5*(-pi+j*du)));
 	if (j < N/2) fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u, x[j-1]*x[j-1] + y[j]*y[j], dy[j-1]*dy[j-1] + dx[j]*dx[j]);
 	else if (j == N/2) fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u,  y[nC-1]*y[nC-1], dx[nC-1]*dx[nC-1]);
 	else fprintf(fh, "%+.12LE\t%+.12LE\t%+.12LE\n", u, x[nS-(j-N/2)]*x[nS-(j-N/2)] + y[nC-1-(j-N/2)]*y[nC-1-(j-N/2)], dx[nC-1-(j-N/2)]*dx[nC-1-(j-N/2)] + dy[nS-(j-N/2)]*dy[nS-(j-N/2)]);
 	}
     fclose(fh);
}
