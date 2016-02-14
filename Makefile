# Makefile for mandelbrot area code

#
# C compiler and options for Intel
#
#CC=     icc -O3 -openmp
#LIB=    -lm

#
# C compiler and options for PGI 
#
CC=     pgcc -O3 -mp -tp=px
LIB=	-lm

#
# C compiler and options for GNU 
#
#CC=     gcc -O3 -fopenmp
#LIB=	-lm

#
# Object files
#
OBJ=    loops2.o

#
# Compile
#
loops2:   $(OBJ)
	$(CC) -o $@ $(OBJ) $(LIB)

.c.o:
	$(CC) -c $<

#
# Clean out object files and the executable.
#
clean:
	rm *.o loops2
