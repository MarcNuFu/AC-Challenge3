#include <stdio.h>
#include <stdlib.h>

#include <xmmintrin.h>

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

void SimulationStep ( REAL * __restrict In, REAL L, int N, REAL bot, int diff)
{
  REAL temp = 1/L - 2.0f;
  REAL InAnterior = bot;
  REAL InActual = In[0];
  REAL InFuturo;
  
 _mm_prefetch((const char*) &In[8], _MM_HINT_NTA);
 _mm_prefetch((const char*) &In[16], _MM_HINT_NTA);
 _mm_prefetch((const char*) &In[24], _MM_HINT_NTA);
 _mm_prefetch((const char*) &In[32], _MM_HINT_NTA); 
 
  for (int x=0; x<N-diff; x+=16)
  {
    for(int j=0; j<16; j++)
    {
      InFuturo = In[x+j+1];
      In[x+j] = L*(InActual*temp + InFuturo + InAnterior);
      InAnterior = InActual;
      InActual = InFuturo;
    }
    _mm_prefetch((const char*) &In[x+40], _MM_HINT_NTA);
    _mm_prefetch((const char*) &In[x+48], _MM_HINT_NTA);        
  }
  
  for (int x=N-diff; x<N; x++)
  {
      InFuturo = In[x+1];
      In[x] = L*(InActual*temp + InFuturo + InAnterior);
      InAnterior = InActual;
      InActual = InFuturo;
  }
}


void SimulationStep2 ( REAL * __restrict In, REAL L, int N, REAL bot, int diff)
{
  REAL temp = 1/L - 2.0f;
  REAL InAnterior = bot;
  REAL InAnterior2 = bot;
  REAL InActual = In[0];
  REAL InActual2;
  REAL InFuturo, InFuturo2;
  
   _mm_prefetch((const char*) &In[8], _MM_HINT_NTA);
   _mm_prefetch((const char*) &In[16], _MM_HINT_NTA);
   _mm_prefetch((const char*) &In[24], _MM_HINT_NTA);
   _mm_prefetch((const char*) &In[32], _MM_HINT_NTA); 
 
 
   for(int j=0; j<16; j++)
   {
      InFuturo = In[j+1];
      In[j] = L*(InActual*temp + InFuturo + InAnterior);
      InAnterior = InActual;
      InActual = InFuturo;
   }
  _mm_prefetch((const char*) &In[40], _MM_HINT_NTA);
  _mm_prefetch((const char*) &In[48], _MM_HINT_NTA);
  
  
  
  InActual2 = In[0];
  for (int x=16; x<N-diff; x+=16)
  {
    for(int j=0; j<16; j++)
    {
      InFuturo = In[x+j+1];
      In[x+j] = L*(InActual*temp + InFuturo + InAnterior);
      InAnterior = InActual;
      InActual = InFuturo;
    }
    
    for(int j=0; j<16; j++)
    {
      InFuturo2 = In[x+j-15];
      In[x+j-16] = L*(InActual2*temp + InFuturo2 + InAnterior2);
      InAnterior2 = InActual2;
      InActual2 = InFuturo2;
    }
    
    _mm_prefetch((const char*) &In[x+40], _MM_HINT_NTA);
    _mm_prefetch((const char*) &In[x+48], _MM_HINT_NTA);        
  }
  
  
  
  for (int x=N-diff; x<N; x++)
  {
      InFuturo = In[x+1];
      In[x] = L*(InActual*temp + InFuturo + InAnterior);
      InAnterior = InActual;
      InActual = InFuturo;
  }
  
  
  for(int j=N-diff-16; j<N-diff; j++)
  {
    InFuturo2 = In[j+1];
    In[j] = L*(InActual2*temp + InFuturo2 + InAnterior2);
    InAnterior2 = InActual2;
    InActual2 = InFuturo2;
  }
  
  for (int x=N-diff; x<N; x++)
  {
      InFuturo2 = In[x+1];
      In[x] = L*(InActual2*temp + InFuturo2 + InAnterior2);
      InAnterior2 = InActual2;
      InActual2 = InFuturo2;
  }
  
}

void SimulationStepMenor16 ( REAL * __restrict In, REAL L, int N, REAL bot)
{
  REAL temp = 1/L - 2.0f;
  REAL InAnterior = bot;
  REAL InActual = In[0];
  REAL InFuturo;
  
  for (int x=0; x<N; x++)
  {
    InFuturo = In[x+1];
    In[x] = L*(InActual*temp + InFuturo + InAnterior);
    InAnterior = InActual;
    InActual = InFuturo;
  }
}


void CopyVector ( REAL *In, REAL *Out, int N, REAL bot )
{
  int x;
  Out[0] = bot;
  for (x=0; x<N+1; x++)
    Out[x+1] = In[x];
}


int main(int argc, char **argv)
{
  int  x, t, N= 10000000, T=1000;
  REAL L= 0.123456, L2, S;
  REAL *U1, *U2;
  REAL bot;
  int diff = N%16;
  int diffCiclos = T%2;
  if (argc>1) { T = atoi(argv[1]); } // get  first command line parameter
  if (argc>2) { N = atoi(argv[2]); } // get second command line parameter
  if (argc>3) { L = atof(argv[3]); } // get  third command line parameter
 
  if (N < 1 || T < 1 || L >= 0.5) {
    printf("arguments: T N L (T: steps, N: vector size, L < 0.5)\n");
    return 1;
  }
  U1 = (REAL *) malloc ( sizeof(REAL)*(N+1) );
  U2 = (REAL *) malloc ( sizeof(REAL)*(N+2) );
  if (!U1 || !U2) { printf("Cannot allocate vectors\n"); exit(1); }
  
  // initialize temperatures at time t=0  
  for (x=0; x<N+1; x++)
    U1[x] = (x+1)*3.1416;
 
  // initialize fixed boundary conditions on U1
  {
    bot = 1.2345678e+12;
    U1[N]= -1.2345678e+16;
  }
  
  printf("Challenge #3: Simulate %d steps on 1-D vector of %d elements with L=%1.10e\n", T, N, L);

  if(N >= 16)
  {
    if(diffCiclos != 0)
      SimulationStep ( U1, L, N, bot, diff ); 

    for (int t=diffCiclos; t<T; t+=2)
    {  // loop on time
      SimulationStep2 ( U1, L, N, bot, diff ); 
    }
  }
  
  else
  {
      for (int t=0; t<T; t++)
      {  // loop on time
        SimulationStepMenor16 ( U1, L, N, bot ); 
      }
  }


  CopyVector(U1, U2, N, bot);

  printCheck(U2,N);
  free (U1); free (U2);
}
