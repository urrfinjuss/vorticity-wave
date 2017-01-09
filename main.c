#include "header.h"

int main(int argc, char **argv){
  int ierr = 0;
  params run;
  switch(argc) {
    case 13:
        for (int l = 1; l < argc-1; l++) {
	   if (!strcmp(argv[l],"-N")) {
		run.N = strtol(argv[l+1], NULL, 10);
		printf("Number of Modes N =\t%ld\n", run.N);
	   } 
	   if (!strcmp(argv[l],"-c")) {
		run.c = strtold(argv[l+1], NULL);
		printf("Velocity   c =\t%.12LE\n", run.c);
	   } 
	   if (!strcmp(argv[l],"-v")) {
		run.v = strtold(argv[l+1], NULL);
		printf("Vorticity  w =\t%.12LE\n", run.v);
	   }
	   if (!strcmp(argv[l],"-d")) {
		run.D = strtold(argv[l+1], NULL);
		printf("Depth D =\t%.12LE\n", run.D);
	   }
	   if (!strcmp(argv[l],"-l")) {
		run.L = strtold(argv[l+1], NULL);
		printf("Refinement L =\t%.12LE\n", run.L);
	   }
	   if (!strcmp(argv[l],"-r")) {
		printf("Restart from %s\n", argv[l+1]);
		run.fh = fopen(argv[l+1],"r");
		if (run.fh == NULL) {
	  	   printf("Cannot open %s\n", argv[l+1]);
		   ierr = 1; exit(ierr);
		} else {
		   ierr = read_guess(&run);
		   ierr = init_arrays(&run);
		   ierr = simulate(&run);
		   if (ierr) exit(ierr);
		}
	   }
	}
	break;

      default:
         printf("Usage:\n%s -N number_of_modes -c speed -v vorticity", argv[0]);
	 printf(" -d depth -l scaling -r restart.txt\n");
	 ierr = 1;
	 exit(ierr);
	 break;
   }
   return ierr;
}






































