// Threaded two-dimensional Discrete FFT transform
// C.V. Chaitanya Krishna
// ECE8893 Project 2

#include <iostream>
#include <string>
#include <math.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

// Declaring global variables visible to all threads
Complex* data;
Complex* res1;
int N,ht,wd,startCount;
int nThreads=16;
int flag=0;

pthread_mutex_t startCountMutex;
pthread_mutex_t exitMutex;
pthread_cond_t exitCond;
pthread_cond_t exitBarr;

// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.
unsigned ReverseBits(unsigned v)
{ //  Provided to students
  unsigned n = N; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value
   
  for (--n; n > 0; n >>= 1)
    {
      r <<= 1;        // Shift return value
      r |= (v & 0x1); // Merge in next bit
      v >>= 1;        // Shift reversal value
    }
  return r;
}

// This program precomputes the weights W and Wconj
void PrecomputeW(int N, Complex* W, Complex* Wconj)
{
	double theta=(2*M_PI/N);
	for (int i=0; i<(N/2); i++)
	{
		W[i]=Complex(cos(theta*i),-sin(theta*i));
		Wconj[i]=W[i].Conj();
	}
}

// This program transposes the matrix mat and stores it back to mat
void Transpose(int N, Complex* mat)
{
	Complex temp;
	for (int i=0;i<N;i++)
	{
		for (int j=i+1; j<N; j++)
		{
			temp=mat[i*N+j];
			mat[i*N+j]=mat[N*j+i];
			mat[N*j+i]=temp;
		}
	}
}

// Call MyBarrier_Init once in main to initialize the mutex and condition variables
void MyBarrier_Init()
{
	pthread_mutex_init(&exitMutex,0);
	pthread_mutex_init(&startCountMutex,0);
	pthread_cond_init(&exitCond, 0);
	pthread_cond_init(&exitBarr, 0);
	pthread_mutex_lock(&exitMutex);
	startCount=nThreads;			// setting active count to number of threads
}

// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier()
{
	pthread_mutex_lock(&startCountMutex);
	startCount--;
	if (startCount==0)
	{
		startCount=nThreads;
		pthread_cond_broadcast(&exitBarr);
	}
	else
	{
		pthread_cond_wait(&exitBarr,&startCountMutex);
	}
	pthread_mutex_unlock(&startCountMutex);
}

// This function implements 1D FFT using Danielson-Lanczos method                    
void Transform1D(Complex* h, int N, Complex* W, Complex* Wconj)
{
  int rowsPerThread=ht/nThreads;		
  for (int a=0; a<rowsPerThread; a++)
  {
  	  Complex* h1 = h+a*N ;			// h1 points to the first element of starting row for the thread
	  Complex H[N];					// to store partial results (only for rowsPerThread rows)
	  Complex* temp;				// using a local var to point towards data pointed by h1
	  Complex* res;					// to store final result of FFT (global access required)
	  temp=h1;
	  
	  for (int i = 0; i < N; i++)
	  {
	    H[ReverseBits(i)]=*temp;	// transform h from natural ordering to bit reversed ordering 
	    temp++;   
	  }

	  int indx=0,x1=0;

	  for (int i = 1; i <= log2 (N); i++)
	  {
	    int x2= (pow(2,i)/2); 
	    indx=0;

	    for (int j = (N/(pow(2,i))); j >0; j--)
	    { 
	      x1=indx;
	      int k1=0;

	      for (int k = pow(2,(i-1)); k >0; k--)
	      {
	        Complex oldH;
	        oldH= H[x1];						// to store previous H[x1] value as we do inplace DFT
	        int c=((k1*N)/(pow(2,i)));			// index for Weights

	        if (flag==0)						// if flag = 0, perform forward FFT
	        {
	        	H[x1]=H[x1] + (W[c] * H[x1+x2]);		
		        H[x1+x2] = oldH - (W[c] * H[x1+x2]);
	        }
		    else								// when flag = 1 when performing inverse transform
	        {
	        	H[x1]=H[x1] + (Wconj[c] * H[x1+x2]);		
		        H[x1+x2] = oldH - (Wconj[c] * H[x1+x2]);
		        if ((i==log2(N)) and (j==1))
		        {
		        	H[x1]=Complex(H[x1].real/N, H[x1].imag/N);
		        	H[x1+x2]=Complex(H[x1+x2].real/N, H[x1+x2].imag/N);
		        }
	        }
		        
	        x1++; 
	        k1++;
	      }

	      indx= indx + pow(2,i); //increment index by power of 2

	    }
	  }

	  res=h1;	//points towards location pointed by h1 which constitues global pointer data

	  for (int i=0;i<N;i++)
	  {
	  	*res=H[i];
	  	res++;	
	  }
  }
}

void* Transform2DTHread(void* v)
{
	unsigned long myID=(unsigned long)v;
	int rowsPerThread=ht/nThreads;
	Complex* h=data+(myID*rowsPerThread*N);
	Complex* W=new Complex[N/2];
	Complex* Wconj=new Complex[N/2];
	PrecomputeW(N,W,Wconj);
	// cout<<"Process "<<myID<<" started\n";
	Transform1D(h,N,W,Wconj);		// Compute 1D FFT row-wise
	// cout<<"Process "<<myID<<" completed\n";
	MyBarrier();	// All threads waiting to finish 1D FFT row-wise
	if(!myID)
	{
		Transpose(N,data);			// Transpose results of 1D FFT to compute column-wise transform
	}
	MyBarrier();	// All threads waiting for thread 0 to transpose the 1D FFT result
	Transform1D(h,N,W,Wconj);		// Compute 1D FFT column-wise
	MyBarrier();	// All threads waiting to finish 1D FFT column-wise
	if(!myID)
	{
		Transpose(N,data);			// Transpose results of 2D FFT to get final result
	}
	MyBarrier();	// All threads waiting for thread 0 to transpose 2D FFT results to get the final result 

	if (!myID)		
	{
		pthread_mutex_lock(&exitMutex);
		pthread_cond_signal(&exitCond);	// thread 0 signals the condition variable
		pthread_mutex_unlock(&exitMutex);
	}
	return 0;
}

// This function is used to create threads which will start at a later time
void createThreads(int nThreads)
{
  for (int i=0;i<nThreads;i++)
  {
  	pthread_t pt;
  	pthread_create(&pt, 0, Transform2DTHread, (void*)i);
  }
}

// This function is called in the main and it compute 2D FFT 
void Transform2D(const char* inputFN) 
{ 
  InputImage image(inputFN);   	// Create the helper object for reading the image
  data = image.GetImageData();	// global pointer for image data
  ht = image.GetHeight();		// height of image
  wd = image.GetWidth();		// width of image
  N=wd;							// N=1024

  cout<<"2D FFT started"<<endl;
  createThreads(nThreads);		// create threads to compute 2D FFT
  pthread_cond_wait(&exitCond, &exitMutex);	// wait till all threads have computed 2D FFT

  image.SaveImageData("MyAfter2D.txt",data,wd,ht);	//save 2D FFT results
  cout<<"2D FFT completed\n"<<endl;  
}

void Transform2dInverse(const char* inputFN) 
{
  InputImage image(inputFN);
  flag=1;						// set falg to 1 to perform Inverse transform

  cout<<"Inverse 2D FFT started"<<endl;
  createThreads(nThreads);		// create threads to compute inverse 2D FFT
  pthread_cond_wait(&exitCond, &exitMutex);	// wait till all threads have computed inverse 2D FFT

  image.SaveImageData("MyAfterInverse.txt",data,wd,ht);	//save inverse 2D FFT results
  cout<<"Inverse 2D FFT completed\n"<<endl;
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  	// if name specified on cmd line
  MyBarrier_Init();						// Initialize mutex and condition vars
  Transform2D(fn.c_str());				// computing 2D FFT
  Transform2dInverse(fn.c_str());		// computing Inverse 2D FFT
}  