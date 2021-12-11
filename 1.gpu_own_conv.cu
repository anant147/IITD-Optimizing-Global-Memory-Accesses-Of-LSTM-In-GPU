#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include <sys/time.h>
#include<time.h>

#define min(a,b) (a<b?a:b)

typedef struct GpuLstmStore {
   //for partial matrix vector results of gates for current and next time steps
   double *i,*f,*g,*o;
   double *c,*h; //present time step computations
   int evenOrOdd; //0 = even, 1 = Odd;

   double *onChipI; //on-chip storage for fixing part of Ri,Rf,Rg,Ro matrices data
   double *onChipF; //on-chip storage for fixing part of Ri,Rf,Rg,Ro matrices data
   double *onChipG; //on-chip storage for fixing part of Ri,Rf,Rg,Ro matrices data
   double *onChipO; //on-chip storage for fixing part of Ri,Rf,Rg,Ro matrices data
   int squareDim; // dimensions beyond which data is stored on-chip
}GpuLstmStore;

typedef struct GpuLstmLayer{
   int inputSize;
   int numHiddenUnits;
   double *Wi,*Wf,*Wg,*Wo;
   double *Ri,*Rf,*Rg,*Ro;
   double *bi,*bf,*bg,*bo;
   double *hiddenState;
   double *cellState;

   double *xinp;
   double *xi,*xf,*xg,*xo;

   GpuLstmStore store; //storage for temporary variables
}GpuLstmLayer;

double* readMatrixFromFile( FILE *fp, int R, int C)
{
    double *buff = NULL;

   if (fp != NULL)
   {
      buff = (double*)malloc(sizeof(double) * R * C);

      for (int i=0;i<R;i++)
      {
         for (int j=0;j<C;j++)
         {
            if(fscanf(fp, "%lf", &buff[i*C+j]) != 1)
               exit(1);
         }
      }
   }

   return buff;
}


void initializeMatrix(double *mat, int R, int C)
{
    srand(time(0));

    for (int i=0;i<R;i++)
    {
        for (int j=0;j<C;j++)
        {
            mat[i*C+j] = (rand()%10000) * 0.0001;
        }
    }
}

__global__  void  gpu_matrixVectorMultWX(double *dmatA, double *dvecB, double *dvecC, int matrow,int matcol)
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int tindex=tidx+gridDim.x*blockDim.x*tidy;


    if(tindex<matrow)
    {
      int i;
      int m=tindex*matcol;
      double sum = 0.0;
    
     for(i=0;i<matcol;i++)
     {
         sum += dmatA[m+i] * dvecB[i];
     }

     dvecC[tindex] = sum;
  
    }

      // __syncthreads();
}

__global__  void  gpu_matrixVectorMultRH(double *dmatA, double *dvecB, double *dvecC, int matrow,int matcol)
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int tindex=tidx+gridDim.x*blockDim.x*tidy;


    if(tindex<matrow)
    {
      int i;
      int m=tindex*matcol;
      double sum = 0.0;
    
     for(i=0;i<matcol;i++)
     {
         sum += dmatA[m+i] * dvecB[i];
     }

     dvecC[tindex] = sum;
  
    }

      // __syncthreads();
}

__global__ void gpu_sigmoid( double *gateval, double *temp1, double *temp2, double *bias, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < n)
    {
        double val1 = temp1[index] + temp2[index] + bias[index];
        gateval[index] = 1/(1+exp(-val1));
    }
}

__global__ void  gpu_tanh( double *gateval, double *temp1, double *temp2, double *bias, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < n)
    {
        double val1 = temp1[index] + temp2[index] + bias[index];
        gateval[index] = tanh(val1);
    }
}

__global__ void gpu_cellAndHiddenFunc( double *hiddenval, double *cellval, double *ival, double *fval, double *gval, double *oval, int n)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < n)
    {
        cellval[index] = fval[index] * cellval[index] + ival[index] * gval[index];
        hiddenval[index] = oval[index] * tanh(cellval[index]);
    }
}


void LSTMForwardStep( double *x, GpuLstmLayer *gpulstmlayer)
{
    /////////// declaration and initailization of variables 

    int inputSize = gpulstmlayer->inputSize;
    int numHiddenUnits = gpulstmlayer->numHiddenUnits;
 
    double *devX;
    double *devWt, *devRt, *devBias; 
    double *temp1, *temp2;
    double *ival, *fval, *gval, *oval;
 
    double *devhpr, *cellval, *hiddenval;
 
    /// assigning gpu memory locations 
    devhpr = gpulstmlayer->hiddenState;
    cellval = gpulstmlayer->cellState;
    hiddenval = gpulstmlayer->hiddenState;
 
    // doing gpu memory allocations cpu to gpu transfers
    cudaMemcpy(gpulstmlayer->xinp, x, sizeof(double) * inputSize, cudaMemcpyHostToDevice);
    devX = gpulstmlayer->xinp;
 
    cudaMalloc(&temp1, sizeof(double) * numHiddenUnits);
    cudaMalloc(&temp2, sizeof(double) * numHiddenUnits);
 
    ival = gpulstmlayer->xi;
    fval = gpulstmlayer->xf;
    gval = gpulstmlayer->xg;
    oval = gpulstmlayer->xo;
 
    /// doing block and grid setting for multiplication and linear functions
    int blockSize = 16;
    int maxSize = blockSize * blockSize;
    int blocksPerGrid = numHiddenUnits/maxSize + 1;
    dim3 dimBlock(blockSize, blockSize);

    if (numHiddenUnits%maxSize == 0)
    {
        blocksPerGrid--;
    }
 
    dim3 dimGrid(1, blocksPerGrid);
 
    int linBlockSize = 1024;
    int linBlockPerGrid = numHiddenUnits/linBlockSize + 1;

    if (numHiddenUnits <= linBlockSize )
    {
        linBlockSize = numHiddenUnits;
        linBlockPerGrid = 1;
    }
    else if (numHiddenUnits%linBlockSize == 0)
    {
        linBlockPerGrid--;   
    }
 

    /// for input gates
    devWt = gpulstmlayer->Wi;
    devRt = gpulstmlayer->Ri;
    devBias = gpulstmlayer->bi;
  
    gpu_matrixVectorMultWX<<< dimGrid, dimBlock >>>(devWt, devX, temp1, numHiddenUnits, inputSize );
    gpu_matrixVectorMultRH<<< dimGrid, dimBlock >>>(devRt, devhpr, temp2, numHiddenUnits, numHiddenUnits );

    gpu_sigmoid<<< linBlockPerGrid , linBlockSize  >>> (ival, temp1, temp2, devBias, numHiddenUnits);


    /// for forget gate
    devWt = gpulstmlayer->Wf;
    devRt = gpulstmlayer->Rf;
    devBias = gpulstmlayer->bf;
  
 
    gpu_matrixVectorMultWX<<< dimGrid, dimBlock >>>(devWt, devX, temp1, numHiddenUnits, inputSize );
    gpu_matrixVectorMultRH<<< dimGrid, dimBlock >>>(devRt, devhpr, temp2, numHiddenUnits, numHiddenUnits );

    gpu_sigmoid<<< linBlockPerGrid , linBlockSize  >>> (fval, temp1, temp2, devBias, numHiddenUnits);


    /// for cell input gate
    devWt = gpulstmlayer->Wg;
    devRt = gpulstmlayer->Rg;
    devBias = gpulstmlayer->bg;
 
    gpu_matrixVectorMultWX<<< dimGrid, dimBlock >>>(devWt, devX, temp1, numHiddenUnits, inputSize );
    gpu_matrixVectorMultRH<<< dimGrid, dimBlock >>>(devRt, devhpr, temp2, numHiddenUnits, numHiddenUnits );

    gpu_tanh<<< linBlockPerGrid , linBlockSize  >>> (gval, temp1, temp2, devBias, numHiddenUnits);


    /// for output gate
    devWt = gpulstmlayer->Wo;
    devRt = gpulstmlayer->Ro;
    devBias = gpulstmlayer->bo;
 
    gpu_matrixVectorMultWX<<< dimGrid, dimBlock >>>(devWt, devX, temp1, numHiddenUnits, inputSize );
    gpu_matrixVectorMultRH<<< dimGrid, dimBlock >>>(devRt, devhpr, temp2, numHiddenUnits, numHiddenUnits );

    gpu_sigmoid<<< linBlockPerGrid , linBlockSize  >>> (oval, temp1, temp2, devBias, numHiddenUnits);


    /// for calculation of cell state and cell hidden state vector 
 
    gpu_cellAndHiddenFunc<<< linBlockPerGrid , linBlockSize >>>( hiddenval, cellval, ival, fval, gval, oval, numHiddenUnits);

    
    // free the variables

    cudaFree( temp1);
    cudaFree( temp2);

 
    // return time;
}


void createGpuLSTMLayer(GpuLstmLayer *gpulstmlayer,int numHiddenUnits,int inputSize,
                                            double *W,double *R,double *bias)
{
    ///
    cudaMalloc(&(gpulstmlayer->Wi), sizeof(double)*numHiddenUnits * inputSize);
    cudaMemcpy(gpulstmlayer->Wi, W, sizeof(double) * numHiddenUnits * inputSize, cudaMemcpyHostToDevice);

    cudaMalloc(&(gpulstmlayer->Wf), sizeof(double)*numHiddenUnits * inputSize);
    cudaMemcpy(gpulstmlayer->Wf, W + numHiddenUnits*inputSize, sizeof(double) * numHiddenUnits * inputSize, cudaMemcpyHostToDevice);

    cudaMalloc(&(gpulstmlayer->Wg), sizeof(double)*numHiddenUnits * inputSize);
    cudaMemcpy(gpulstmlayer->Wg, W + 2 * numHiddenUnits*inputSize, sizeof(double) * numHiddenUnits * inputSize, cudaMemcpyHostToDevice);

    cudaMalloc(&(gpulstmlayer->Wo), sizeof(double)*numHiddenUnits * inputSize);
    cudaMemcpy(gpulstmlayer->Wo, W + 3 * numHiddenUnits*inputSize, sizeof(double) * numHiddenUnits * inputSize, cudaMemcpyHostToDevice);
 
    ///
    cudaMalloc(&(gpulstmlayer->Ri), sizeof(double)*numHiddenUnits * numHiddenUnits);
    cudaMemcpy(gpulstmlayer->Ri, R, sizeof(double) * numHiddenUnits * numHiddenUnits, cudaMemcpyHostToDevice);

    cudaMalloc(&(gpulstmlayer->Rf), sizeof(double)*numHiddenUnits * numHiddenUnits);
    cudaMemcpy(gpulstmlayer->Rf, R + numHiddenUnits * numHiddenUnits, sizeof(double) * numHiddenUnits * numHiddenUnits, cudaMemcpyHostToDevice);
 
    cudaMalloc(&(gpulstmlayer->Rg), sizeof(double)*numHiddenUnits * numHiddenUnits);
    cudaMemcpy(gpulstmlayer->Rg, R + 2 * numHiddenUnits * numHiddenUnits, sizeof(double) * numHiddenUnits * numHiddenUnits, cudaMemcpyHostToDevice);

    cudaMalloc(&(gpulstmlayer->Ro), sizeof(double)*numHiddenUnits * numHiddenUnits);
    cudaMemcpy(gpulstmlayer->Ro, R + 3 * numHiddenUnits * numHiddenUnits, sizeof(double) * numHiddenUnits * numHiddenUnits, cudaMemcpyHostToDevice);

    ////
    cudaMalloc(&(gpulstmlayer->bi), sizeof(double) * numHiddenUnits);
    cudaMemcpy(gpulstmlayer->bi, bias, sizeof(double) * numHiddenUnits, cudaMemcpyHostToDevice);

    cudaMalloc(&(gpulstmlayer->bf), sizeof(double) * numHiddenUnits);
    cudaMemcpy(gpulstmlayer->bf, bias + numHiddenUnits, sizeof(double) * numHiddenUnits, cudaMemcpyHostToDevice);
 
    cudaMalloc(&(gpulstmlayer->bg), sizeof(double) * numHiddenUnits);
    cudaMemcpy(gpulstmlayer->bg, bias + 2 * numHiddenUnits, sizeof(double) * numHiddenUnits, cudaMemcpyHostToDevice);

    cudaMalloc(&(gpulstmlayer->bo), sizeof(double) * numHiddenUnits);
    cudaMemcpy(gpulstmlayer->bo, bias + 3 * numHiddenUnits, sizeof(double) * numHiddenUnits, cudaMemcpyHostToDevice);

    //initialize the cell state and hidden units
    cudaMalloc(&(gpulstmlayer->cellState), sizeof(double) * numHiddenUnits);
    cudaMalloc(&(gpulstmlayer->hiddenState), sizeof(double) * numHiddenUnits);
 
    cudaMemset(gpulstmlayer->cellState, 0, sizeof(double) * numHiddenUnits);
    cudaMemset(gpulstmlayer->hiddenState, 0, sizeof(double) * numHiddenUnits);
    
    gpulstmlayer->inputSize = inputSize;
    gpulstmlayer->numHiddenUnits = numHiddenUnits;

    //memory for gates, even and odd 
    cudaMalloc(&(gpulstmlayer->store.i), sizeof(double) * numHiddenUnits);
    cudaMemset(gpulstmlayer->store.i, 0, sizeof(double) * numHiddenUnits);
 
    cudaMalloc(&(gpulstmlayer->store.f), sizeof(double) * numHiddenUnits);
    cudaMemset(gpulstmlayer->store.f, 0, sizeof(double) * numHiddenUnits);
 
    cudaMalloc(&(gpulstmlayer->store.g), sizeof(double) * numHiddenUnits);
    cudaMemset(gpulstmlayer->store.g, 0, sizeof(double) * numHiddenUnits);

    cudaMalloc(&(gpulstmlayer->store.o), sizeof(double) * numHiddenUnits);
    cudaMemset(gpulstmlayer->store.o, 0, sizeof(double) * numHiddenUnits);
 
    cudaMalloc(&(gpulstmlayer->store.c), sizeof(double) * numHiddenUnits);
    cudaMemset(gpulstmlayer->store.c, 0, sizeof(double) * numHiddenUnits);
 
    cudaMalloc(&(gpulstmlayer->store.h), sizeof(double) * numHiddenUnits);
    cudaMemset(gpulstmlayer->store.h, 0, sizeof(double) * numHiddenUnits);
 
    cudaMalloc(&(gpulstmlayer->xinp), sizeof(double) * inputSize);
    
    cudaMalloc(&(gpulstmlayer->xi), sizeof(double) * numHiddenUnits);
    cudaMalloc(&(gpulstmlayer->xf), sizeof(double) * numHiddenUnits);
    cudaMalloc(&(gpulstmlayer->xg), sizeof(double) * numHiddenUnits);
    cudaMalloc(&(gpulstmlayer->xo), sizeof(double) * numHiddenUnits);
 

    gpulstmlayer->store.evenOrOdd = 0; //start with even, lower diagonal matrix.
}



void freeGpuLSTMLayer(GpuLstmLayer *gpulstmlayer)
{
    cudaFree(gpulstmlayer->Wi);
    cudaFree(gpulstmlayer->Wf);
    cudaFree(gpulstmlayer->Wg);
    cudaFree(gpulstmlayer->Wo);

    cudaFree(gpulstmlayer->Ri);
    cudaFree(gpulstmlayer->Rf);
    cudaFree(gpulstmlayer->Rg);
    cudaFree(gpulstmlayer->Ro);

    cudaFree(gpulstmlayer->bi);
    cudaFree(gpulstmlayer->bf);
    cudaFree(gpulstmlayer->bg);
    cudaFree(gpulstmlayer->bo);

    cudaFree(gpulstmlayer->store.i);
    cudaFree(gpulstmlayer->store.f);
    cudaFree(gpulstmlayer->store.g);
    cudaFree(gpulstmlayer->store.o);

    cudaFree(gpulstmlayer->store.c);
    cudaFree(gpulstmlayer->store.h);
 
    cudaFree(gpulstmlayer->xinp);
 
    cudaFree(gpulstmlayer->xi);
    cudaFree(gpulstmlayer->xf);
    cudaFree(gpulstmlayer->xg);
    cudaFree(gpulstmlayer->xo);

    // free(lstmlayer->Wi); //these were allocated outside createLSTMLayer
    // free(lstmlayer->Ri);//these were allocated outside createLSTMLayer
    // free(lstmlayer->bi);
    cudaFree(gpulstmlayer->hiddenState);
    cudaFree(gpulstmlayer->cellState);
}


void writeMatrixIntoFile( FILE *fp, double *mat, int R, int C)
{
   for (int i=0;i<R;i++)
   {
      for (int j=0;j<C;j++)
      {
         fprintf(fp, "%lf ", mat[i*C+j] );
      }
      fprintf(fp,"\n");
   }
}

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d", stat);
  return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime) {
  printf("%s%lf seconds\n", str, endtime - starttime);
}


int main()
{
   printf(" Method 1 :- Conventional GPU implementation  method :- \n");
   int inputSize, numHiddenUnits, timeSteps;

   // char Fx[40], FinWts[40], FrecWts[40], Fbias[40] ;
   // char Fy[40];

   // printf("Enter the parameters of LSTM layer :- \n");

   // printf(" Enter the value of input size : ");
   // scanf("%d", &inputSize);
   // printf(" Enter the value of number of hidden units : ");
   // scanf("%d", &numHiddenUnits);
   // printf(" Enter the value of number of time Steps : ");
   // scanf("%d", &timeSteps);

   // printf("\n");
   
   // printf(" file name containing input vector : ");
   // scanf("%s", Fx);
   // printf(" file name containing input matrix weights :");
   // scanf("%s", FinWts);
   // printf(" file name containing hidden matrix weights : ");
   // scanf("%s", FrecWts);
   // printf(" file name containing bias vector : ");
   // scanf("%s", Fbias);
   // printf(" file name containing output vector : ");
   // scanf("%s", Fy);

   inputSize = 1024;
   numHiddenUnits = 1024;
   timeSteps = 2;
   printf(" inputSize = %d , numHiddenUnits = %d , timeSteps = %d \n", inputSize, numHiddenUnits, timeSteps);

   // strcpy( Fx, "Fx_8_2.txt");
   // strcpy( FinWts, "FinWts_8_8.txt");
   // strcpy( FrecWts, "Frec_8_8.txt");
   // strcpy( Fbias, "Fbias_8_1.txt");
   // strcpy( Fy, "Fy_gpu_conv_8_2.txt");

   // FILE *fpX = fopen(Fx, "r");
   // FILE *fpInWts = fopen(FinWts, "r");
   // FILE *fpRecWts = fopen(FrecWts, "r");
   // FILE *fpBias = fopen(Fbias, "r");
   // FILE *fpY = fopen(Fy, "w");

   // if ((fpX == NULL) || (fpInWts == NULL) || ( fpRecWts == NULL) || (fpBias == NULL) )
   // {
      // printf(" file opening error ");
      // return -1;
   // }

   double *X, *W, *R, *bias;
   double *Y;

    // X = readMatrixFromFile( fpX, timeSteps, inputSize);
    // W = readMatrixFromFile( fpInWts, 4 * numHiddenUnits, inputSize);
    // R = readMatrixFromFile( fpRecWts, 4 * numHiddenUnits, numHiddenUnits);
    // bias = readMatrixFromFile( fpBias, 4 * numHiddenUnits, 1);

    // fclose( fpX);
    // fclose( fpInWts);
    // fclose( fpRecWts);
    // fclose( fpBias);

    W = (double*)malloc(sizeof(double) * 4 * numHiddenUnits * inputSize);
    R = (double*)malloc(sizeof(double) * 4 * numHiddenUnits * numHiddenUnits);
    bias = (double*)malloc(sizeof(double) * 4 * numHiddenUnits * 1);
    X = (double*)malloc(sizeof(double) * inputSize * timeSteps);
    Y = (double*)malloc(sizeof(double) * numHiddenUnits * timeSteps);

    initializeMatrix(W, 4*numHiddenUnits, inputSize);
    initializeMatrix(R, 4*numHiddenUnits, numHiddenUnits);
    initializeMatrix(bias, 4*numHiddenUnits, 1);
    initializeMatrix(X, inputSize, timeSteps);
    // initializeMatrix(Y, numHiddenUnits, timeSteps);
    memset(Y, 0, sizeof(numHiddenUnits * timeSteps));

    // Y = (double*)malloc(sizeof(double) * numHiddenUnits * timeSteps);
    // memset(Y, 0, sizeof(double) * numHiddenUnits * timeSteps);

    GpuLstmLayer gpulstmlayer;
    createGpuLSTMLayer(&gpulstmlayer,numHiddenUnits,inputSize,W,R,bias);

    double total_time = 0.0;
 
    //double startTime , endTime;
    struct timeval t1, t2;
 
    for (int i=0;i<timeSteps;i++)
    {
      // startTime = rtclock();
     
      gettimeofday(&t1, 0);
      LSTMForwardStep(&X[i*inputSize],&gpulstmlayer);
      cudaMemcpy(&Y[i*numHiddenUnits],gpulstmlayer.hiddenState,sizeof(double)*numHiddenUnits, cudaMemcpyDeviceToHost);
      
      //endTime = rtclock();
      gettimeofday(&t2, 0);
     
      double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
      
      // total_time += endTime - startTime;
     
      total_time += time;
      printf(" time for this step is %lf milliseconds \n ", time);
      printf("time after %d time step is %lf milliseconds \n", i, total_time );
    }
    printf("\n");


    printf("total time taken by program is %lf milliseconds for input size of %d , numHiddenUnits of %d and timeSteps of %d \n", total_time, inputSize, numHiddenUnits, timeSteps );
    // writeMatrixIntoFile(fpY,  Y, timeSteps, inputSize);
    // fclose(fpY);

    freeGpuLSTMLayer( &gpulstmlayer);
    free(W);
    free(R);
    free(bias);
    free(X);
    free(Y);
}