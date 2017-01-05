#include "header.h"

void writer_full(char* str, long double *x, long double *yk, long double *y, long int N) {
  FILE *fh_out = fopen(str, "w");
  fprintf(fh_out, "#1.u 2.k 3. x-tilde 4. yk 5. y \n\n");
  fprintf(fh_out, "%.15LE\t%ld\t%.15LE\t%.15LE\t%.15LE\n", 0.L, 0, 0.L, yk[0], y[0]);
  for (int j = 1; j < N/2; j++) {
    fprintf(fh_out, "%.15LE\t%ld\t%.15LE\t%.15LE\t%.15LE\n", 2.Q*pi*j/N, j, x[j-1], yk[j], y[j]);
  } 
  fprintf(fh_out, "%.15LE\t%ld\t%.15LE\t%.15LE\t%.15LE\n", pi, N/2, 0.Q, yk[N/2], y[N/2]);
  // must write other half
  for (int j = 1; j < N/2; j++) fprintf(fh_out, "%.15LE\t%ld\t%.15LE\t%.15LE\t%.15LE\n", 2.Q*pi*(j+N/2)/N, j-N, -x[N/2-1-j],
      yk[N/2-j], y[N/2-j]);
  fclose(fh_out);
}
