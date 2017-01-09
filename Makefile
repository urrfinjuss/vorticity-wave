FSW_VRT	= vorticity.x

#List of sources
C_FSW_VRT	= main.c new.c prep.c func.c operators.c output.c

# Compilers
CC      = gcc
LINK    = gcc
OPT     = -march=native -Ofast -std=gnu99 -Wno-format -Wcast-qual -Wpointer-arith -Wcast-align -fno-schedule-insns -fschedule-insns2 -fstrict-aliasing -funroll-loops -fprefetch-loop-arrays -fomit-frame-pointer 

#-----------------------------
#generic

LIB_MPI         =
LIB_FFT         = -L/home/orange/fftw3lib/lib -L/usr/local/lib -lfftw3l -lgsl -lgslcblas -lm
INC_MPI         =
INC_FFT         = -I/home/orange/fftw3lib/include -I/home/orange/usr/include
LIB_ADD         =

ifeq ($(HOSTNAME), pequena)
        LIB_MPI         =
        LIB_FFT         = -lfftw3l -lm -lgsl
        INC_MPI         =
        INC_FFT         =
endif
ifeq ($(HOSTNAME), metropolis.RR)
        LIB_MPI         =
        LIB_FFT         = -L/opt/local/fftw/3.3.1/mvapich2/1.7/intel/12.1/lib -lfftw3l -lgsl -lgslcblas -lm
        INC_MPI         =
        INC_FFT         =
endif

#-----------------------------

OBJ_FSW_VRT	= $(C_FSW_VRT:.c=.o) $(F_FSW_VRT:.f=.o)

LIB_FSW         = $(LIB_MPI) $(LIB_FFT) $(LIB_ADD)
INC_FSW         = $(INC_MPI) $(INC_FFT)

#-----------------------------

default: vorticity

.f.o:
	$(FC) $(FFLAGS) -c $<

vorticity:
	$(CC) $(OPT) $(DEF_FSW) $(INC_FSW) -c $(C_FSW_VRT)
	$(LINK) $(OPT) $(OBJ_FSW_VRT) $(LIB_FSW) -o $(FSW_VRT)
	cp -f vorticity.x ./debug/vorticity.x

hostname:
	@echo $(HOSTNAME) $(INC_FFT)

clean:
	@echo "cleaning ..."
	rm -f *~ *.o

#-------------------------------


