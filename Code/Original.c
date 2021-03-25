#include <stdio.h>
#include <stdlib.h>

// Define here as constant for easy change
#define REAL double

void printCheck ( REAL V[], int N )
{
  int x;

  REAL S=0;
  for (x=0; x<=N+1; x++)
    S = S + V[x];

  printf("\nCheckSum = %1.10e\nSome values: ", S);

  for (x=0; x<10; x++)
    printf("(%d)=%1.10f, ", x*N/10, V[x*N/10]);

  printf("(%d)=%1.10f\n", x*N/10, V[x*N/10]);
}


void SimulationStep ( REAL *In, REAL *Out, REAL L, int N )
{
  for (int x=1; x<=N; x++)
    Out[x] = In[x] - 2.0f*L*In[x] + L*In[x+1] + L*In[x-1];
}


void CopyVector ( REAL *In, REAL *Out, int N )
{
  for (int x=1; x<=N; x++)
    Out[x] = In[x];
}


int main(int argc, char **argv)
{
  int  x, t, N= 10000000, T=1000;
  REAL L= 0.123456, L2, S;
  REAL *U1, *U2;

  if (argc>1) { T = atoi(argv[1]); } // get  first command line parameter
  if (argc>2) { N = atoi(argv[2]); } // get second command line parameter
  if (argc>3) { L = atof(argv[3]); } // get  third command line parameter
 
  if (N < 1 || T < 1 || L >= 0.5) {
    printf("arguments: T N L (T: steps, N: vector size, L < 0.5)\n");
    return 1;
  }

  U1 = (REAL *) malloc ( sizeof(REAL)*(N+2) );
  U2 = (REAL *) malloc ( sizeof(REAL)*(N+2) );
  if (!U1 || !U2) { printf("Cannot allocate vectors\n"); exit(1); }

  // initialize temperatures at time t=0  
  for (x=0; x<=N+1; x++)
    U1[x] = x*3.1416;
 
  // initialize fixed boundary conditions on U1
  {
    U1[0]  = 1.2345678e+12;
    U1[N+1]= -1.2345678e+16;
  }

  printf("Challenge #3: Simulate %d steps on 1-D vector of %d elements with L=%1.10e\n", T, N, L);

  for (t=1; t<=T; t++)
  {  // loop on time
    SimulationStep ( U1, U2, L, N ); 
    CopyVector     ( U2, U1, N );
  }

  printCheck(U1,N);
  free(U1); free(U2);
}
