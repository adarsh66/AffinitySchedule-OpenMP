#include <stdio.h>
#include <math.h>
#define N 729
#define reps 100 
#include <omp.h> 

/* Thread sync
 * Set USE_LOCKS to TRUE if you want Affinity schd to use omp locks 
 * If FALSE, Affinity schd will use critical sections instead
 */
#define TRUE 1
#define FALSE 0
#define USE_LOCKS FALSE
#define MAX_PROCS 64
#define DEBUG FALSE

/******************************************************************
 * Author: Adarsh Janakiramand (adarsh66)
 * Date  : December 2015
******************************************************************/

double a[N][N], b[N][N], c[N];
int jmax[N];  

/* For implementing Affinity schedule */
int remaining_iters[MAX_PROCS], hi[MAX_PROCS], lo[MAX_PROCS];
omp_lock_t remaining_iters_lock[MAX_PROCS];

void init1(void);
void init2(void);
void runloop(int); 
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);

/* Affinity schedule functions */
void get_most_loaded_thread_details(int, int*, int*);
void get_chunks(int, double, int*, int*);
void print_run_details(char*, int, int, int, int, int);
int read_remaining_iters(int);

int main(int argc, char *argv[]) { 

  double start1,start2,end1,end2;
  int r;
  int i;
  
  /* Init locks */
  for(i=0;i<MAX_PROCS;i++) omp_init_lock(&(remaining_iters_lock[i]));

  init1(); 

  start1 = omp_get_wtime(); 

  for (r=0; r<reps; r++){ 
    runloop(1);
  } 

  end1  = omp_get_wtime();  

  valid1(); 

  printf("Total time for %d reps of loop 1 = %f\n",reps, (float)(end1-start1)); 


  init2(); 

  start2 = omp_get_wtime(); 

  for (r=0; r<reps; r++){ 
    runloop(2);
  } 

  end2  = omp_get_wtime(); 

  valid2(); 

  printf("Total time for %d reps of loop 2 = %f\n",reps, (float)(end2-start2)); 

  for(i=0;i<MAX_PROCS;i++) omp_destroy_lock(&(remaining_iters_lock[i]));

} 

void init1(void){
  int i,j; 

  for (i=0; i<N; i++){ 
    for (j=0; j<N; j++){ 
      a[i][j] = 0.0; 
      b[i][j] = 3.142*(i+j); 
    }
  }

}

void init2(void){ 
  int i,j, expr; 

  for (i=0; i<N; i++){ 
    expr =  i%( 3*(i/30) + 1); 
    if ( expr == 0) { 
      jmax[i] = N;
    }
    else {
      jmax[i] = 1; 
    }
    c[i] = 0.0;
  }

  for (i=0; i<N; i++){ 
    for (j=0; j<N; j++){ 
      b[i][j] = (double) (i*j+1) / (double) (N*N); 
    }
  }
 
} 


void runloop(int loopid)  {

#pragma omp parallel default(none) shared(loopid, remaining_iters, hi, lo, remaining_iters_lock) 
  {
    int chunk, start_iter, end_iter, remaining_iters_tmp;
    int next_thread_id;
    int myid  = omp_get_thread_num();
    int nthreads = omp_get_num_threads(); 
    double K = (double) 1/nthreads;//k=1/p
    int ipt = (int) ceil((double)N/(double)nthreads); 
    lo[myid] = myid*ipt;
    hi[myid] = (myid+1)*ipt;
    if (hi[myid] > N) hi[myid] = N;

    remaining_iters_tmp = hi[myid]-lo[myid];
    remaining_iters[myid] = remaining_iters_tmp;

    while(remaining_iters_tmp > 0) { 
	get_chunks(myid, K, &start_iter, &chunk);
	/* Set DEBUG flag to TRUE if you want to see the flow details*/
	if(DEBUG==TRUE) print_run_details("Own", loopid, myid, myid, start_iter, chunk);
	switch(loopid){
		case 1: loop1chunk(start_iter, start_iter+chunk);
		case 2: loop2chunk(start_iter, start_iter+chunk);
	}
	remaining_iters_tmp = read_remaining_iters(myid);
    }//end while loop 1

    get_most_loaded_thread_details(nthreads, &next_thread_id, &remaining_iters_tmp);
    
    while(remaining_iters_tmp >0){
	get_chunks(next_thread_id, K, &start_iter, &chunk);
	
	/* Set DEBUG flag to TRUE if you want to see the flow details*/
	if(DEBUG==TRUE) print_run_details("Affinity", loopid, myid, next_thread_id, start_iter, chunk);
        switch(loopid){
                case 1: loop1chunk(start_iter, start_iter+chunk);
                case 2: loop2chunk(start_iter, start_iter+chunk);
        }
	get_most_loaded_thread_details(nthreads, &next_thread_id, &remaining_iters_tmp);
    }//end while loop 2
	
  }
}

/* finds the thread with the max iterations till to complete 
 * If all iterations are complete, the pointer will return thread 0 by default*/ 
void get_most_loaded_thread_details(int nthreads, int* next_thread_id, int* next_remaining_iters){
	int i, remaining_iters_tmp, max_remaining_iters = 0, max_thread_id=0;
	for (i=0;i<nthreads;i++){
		remaining_iters_tmp = read_remaining_iters(i);
		if (remaining_iters_tmp > max_remaining_iters){
			max_remaining_iters = remaining_iters_tmp;
			max_thread_id = i;
		}
	}
	*next_thread_id = max_thread_id;
	*next_remaining_iters = max_remaining_iters;
}

/* Gets the next chunk of iterations to perform for the given thread_id
 * If USE_LOCKS is set to True, function will implement locks
 * Else it will initiate a critical region for the shared variable read-write
*/
void get_chunks(int thread_id, double K, int* start_iter, int* chunk)
{
	int remaining_iters_num, chunk_size;
	if(USE_LOCKS == FALSE) {
		#pragma omp critical (chunk)
		{
			remaining_iters_num = remaining_iters[thread_id];
        		chunk_size = (int) ceil((double)remaining_iters_num*K);
        		if (chunk_size > remaining_iters_num) chunk_size = remaining_iters_num;
			remaining_iters[thread_id] = (remaining_iters_num - chunk_size);
		}
	} else {
		omp_set_lock(&(remaining_iters_lock[thread_id]));
		remaining_iters_num = remaining_iters[thread_id];
        	chunk_size = (int) ceil((double)remaining_iters_num*K);
        	if (chunk_size > remaining_iters_num) chunk_size = remaining_iters_num;
		remaining_iters[thread_id] = (remaining_iters_num - chunk_size);
		omp_unset_lock(&(remaining_iters_lock[thread_id]));
	}
	*start_iter = hi[thread_id]-remaining_iters_num;
	*chunk = chunk_size;
		
}


/* Function wrapping around reading the shared array remaining_iters
 * If USE_LOCKS is set to True, function will implement locks
 * Else it will initiate a critical region for the shared var read
 */
int read_remaining_iters(int thread_id)
{
	int result;
	if (USE_LOCKS == FALSE) {
		#pragma omp critical (chunk)
		{
			result = remaining_iters[thread_id];
		}
	}else {
		omp_set_lock(&(remaining_iters_lock[thread_id]));
		result = remaining_iters[thread_id];
		omp_unset_lock(&(remaining_iters_lock[thread_id]));
	}
	return result;
}

/* Print out affinity schd flow details to std out */
void print_run_details(char *type, int loopid, int orig_thread_id, int thread_id, int start, int chunk)
{
	printf("%s, %d, %d, %d, %d, %d, %d\n",
			type, loopid, orig_thread_id, thread_id, start, start+chunk, chunk);
}

void loop1chunk(int lo, int hi) { 
  int i,j; 
  
  for (i=lo; i<hi; i++){ 
    for (j=N-1; j>i; j--){
      a[i][j] += cos(b[i][j]);
    } 
  }

} 



void loop2chunk(int lo, int hi) {
  int i,j,k; 
  double rN2; 

  rN2 = 1.0 / (double) (N*N);  

  for (i=lo; i<hi; i++){ 
    for (j=0; j < jmax[i]; j++){
      for (k=0; k<j; k++){ 
	c[i] += (k+1) * log (b[i][j]) * rN2;
      } 
    }
  }

}

void valid1(void) { 
  int i,j; 
  double suma; 
  
  suma= 0.0; 
  for (i=0; i<N; i++){ 
    for (j=0; j<N; j++){ 
      suma += a[i][j];
    }
  }
  printf("Loop 1 check: Sum of a is %lf\n", suma);

} 


void valid2(void) { 
  int i; 
  double sumc; 
  
  sumc= 0.0; 
  for (i=0; i<N; i++){ 
    sumc += c[i];
  }
  printf("Loop 2 check: Sum of c is %f\n", sumc);
} 
 

