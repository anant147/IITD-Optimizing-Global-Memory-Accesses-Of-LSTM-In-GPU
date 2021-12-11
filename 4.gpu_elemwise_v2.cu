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


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif


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

__device__ double sigmoid_elementwise(double val)
{
    return 1/(1+exp(-val));
}

__global__ void gateValueComputation_Cond0(double *devHpr, double *devCpr, 
                                           double *devRi, double *devRf, double *devRg, double *devRo,
                                           double *devBi, double *devBf, double *devBg, double *devBo,
                                           double *xi, double *xf, double *xg, double *xo,
                                           double *ival, double *fval,double *gval,double *oval, double *cval, double *hval,
                                           int numHiddenUnits)
{
    __shared__ double iPres, fPres, gPres, oPres;  
    __shared__ double iNext ,fNext ,gNext , oNext;
 
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index == 0)
    {
        iPres = 0.0; fPres = 0.0; gPres = 0.0; oPres = 0.0;
        iNext = 0.0; fNext = 0.0; gNext = 0.0; oNext = 0.0;
    }
 
    __syncthreads();

    for (int i=0; i<numHiddenUnits; i++)
    {
        if (index < i)
        {
            double tmpHpr = devHpr[index];
            double tmphval = hval[index];
        
            double tRi = devRi[i * numHiddenUnits + index];
            atomicAdd(&iPres, tmpHpr * tRi);
            atomicAdd(&iNext, tmphval * tRi);
        
            double tRf = devRf[i * numHiddenUnits + index];
            atomicAdd(&fPres, tmpHpr * tRf);
            atomicAdd(&fNext, tmphval * tRf);
        
            double tRg = devRg[i * numHiddenUnits + index];
            atomicAdd(&gPres, tmpHpr * tRg);
            atomicAdd(&gNext, tmphval * tRg);
        
            double tRo = devRo[i * numHiddenUnits + index];
            atomicAdd(&oPres, tmpHpr * tRo);
            atomicAdd(&oNext, tmphval * tRo);
        }

        __syncthreads();
     
        if (i==index)
        {
            double tmpHpr = devHpr[i];

            double tRi = devRi[i * numHiddenUnits + index];
            atomicAdd(&iPres, tmpHpr * tRi);
        
            double tRf = devRf[i * numHiddenUnits + index];
            atomicAdd(&fPres, tmpHpr * tRf);
        
            double tRg = devRg[i * numHiddenUnits + index];
            atomicAdd(&gPres, tmpHpr * tRg);
        
            double tRo = devRo[i * numHiddenUnits + index];
            atomicAdd(&oPres, tmpHpr * tRo);
        
            iPres += ival[i];
            fPres += fval[i];
            gPres += gval[i];
            oPres += oval[i];
        
            iPres = sigmoid_elementwise(iPres + xi[i] + devBi[i]);
            fPres = sigmoid_elementwise(fPres + xf[i] + devBf[i]);
            gPres = tanh(gPres + xg[i] + devBg[i]);
            oPres = sigmoid_elementwise(oPres + xo[i] + devBo[i]);
        
            cval[i] = fPres * devCpr[i] + iPres * gPres;
            hval[i] = oPres * tanh(cval[i]);
        
            double tmphval = hval[i];
        
            iNext += tmphval * tRi;
            fNext += tmphval * tRf;
            gNext += tmphval * tRg;
            oNext += tmphval * tRo;
        
            ival[i] = iNext;
            fval[i] = fNext;
            gval[i] = gNext;
            oval[i] = oNext;

            iPres = 0.0; fPres = 0.0; gPres = 0.0; oPres = 0.0;
            iNext = 0.0; fNext = 0.0; gNext = 0.0; oNext = 0.0;  
        
        }

        __syncthreads();
    }
 
}

__global__ void outputElementComputation_Cond1(double *devCpr, double *xi, double *xf, double *xg, double *xo, 
                                               double *devBi, double *devBf, double *devBg, double *devBo,
                                               double *ival, double *fval, double *gval, double *oval, double *cval,double *hval,
                                               int rindex, int numHiddenUnits)
{
    ival[rindex] = sigmoid_elementwise(ival[rindex] + xi[rindex] + devBi[rindex]);
    fval[rindex] = sigmoid_elementwise(fval[rindex] + xf[rindex] + devBf[rindex]);
    gval[rindex] = tanh(gval[rindex] + xg[rindex] + devBg[rindex]);
    oval[rindex] = sigmoid_elementwise(oval[rindex] + xo[rindex] + devBo[rindex]);
    cval[rindex] = fval[rindex] * devCpr[rindex] + ival[rindex] * gval[rindex];
    hval[rindex] = oval[rindex] * tanh(cval[rindex]);
 
    if (rindex == numHiddenUnits-1)
    {
        ival[rindex] = 0.0;
        fval[rindex] = 0.0;
        gval[rindex] = 0.0;
        oval[rindex] = 0.0;
    }
}


__global__ void gateValueComputation_Cond1(double *devHpr, double *devCpr, 
                                           double *devRi, double *devRf, double *devRg, double *devRo,
                                           double *devBi, double *devBf, double *devBg, double *devBo,
                                           double *xi, double *xf, double *xg, double *xo,
                                           double *ival, double *fval,double *gval,double *oval, double *cval, double *hval,
                                           int numHiddenUnits)
{
    __shared__ double iPres, fPres, gPres, oPres;  
    __shared__ double iNext ,fNext ,gNext , oNext;
 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
 
    index = numHiddenUnits-1 - index;

    if (index == numHiddenUnits-1)
    {
        iPres = 0.0; fPres = 0.0; gPres = 0.0; oPres = 0.0;
        iNext = 0.0; fNext = 0.0; gNext = 0.0; oNext = 0.0;
    }
 
    __syncthreads();
 
    for (int i=numHiddenUnits-2; i>=0; i--)
    {
        if (i<index)
        {
            double tmpHpr = devHpr[index];
            double tmphval = hval[index];
        
            double tRi = devRi[i * numHiddenUnits + index];
            atomicAdd(&iPres, tmpHpr * tRi);
            atomicAdd(&iNext, tmphval * tRi);
        
            double tRf = devRf[i * numHiddenUnits + index];
            atomicAdd(&fPres, tmpHpr * tRf);
            atomicAdd(&fNext, tmphval * tRf);
        
            double tRg = devRg[i * numHiddenUnits + index];
            atomicAdd(&gPres, tmpHpr * tRg);
            atomicAdd(&gNext, tmphval * tRg);
        
            double tRo = devRo[i * numHiddenUnits + index];
            atomicAdd(&oPres, tmpHpr * tRo);
            atomicAdd(&oNext, tmphval * tRo);
        }

        __syncthreads();
    
        if (index == numHiddenUnits-1)
        {
            iPres += ival[i];
            fPres += fval[i];
            gPres += gval[i];
            oPres += oval[i];
        
            iPres = sigmoid_elementwise(iPres + xi[i] + devBi[i]);
            fPres = sigmoid_elementwise(fPres + xf[i] + devBf[i]);
            gPres = tanh(gPres + xg[i] + devBg[i]);
            oPres = sigmoid_elementwise(oPres + xo[i] + devBo[i]);
        
            cval[i] = fPres * devCpr[i] + iPres * gPres;
            hval[i] = oPres * tanh(cval[i]);
        
            ival[i] = iNext;
            fval[i] = fNext;
            gval[i] = gNext;
            oval[i] = oNext;
         
            iPres = 0.0; fPres = 0.0; gPres = 0.0; oPres = 0.0;
            iNext = 0.0; fNext = 0.0; gNext = 0.0; oNext = 0.0; 
        }
     
        __syncthreads();
    }
}


void LSTMForwardStepReuse(double *x, GpuLstmLayer *gpulstmlayer)
{
   int inputSize = gpulstmlayer->inputSize;
   int numHiddenUnits = gpulstmlayer->numHiddenUnits;
   int condition = gpulstmlayer->store.evenOrOdd;

   double *devX;
   double *devWi, *devWf, *devWg, *devWo;
   double *devRi, *devRf, *devRg, *devRo;
   double *devBi, *devBf, *devBg, *devBo;
   double *xi, *xf, *xg, *xo;
   double *hval, *cval, *ival, *fval, *gval, *oval;
   double *devHpr, *devCpr;
 
   /// doing gpu memory allocation and copying cpu data into gpu memory
   cudaMemcpy(gpulstmlayer->xinp, x, sizeof(double) * inputSize, cudaMemcpyHostToDevice);
   devX = gpulstmlayer->xinp;
 
   ///
   devWi = gpulstmlayer->Wi;
   devWf = gpulstmlayer->Wf;
   devWg = gpulstmlayer->Wg;
   devWo = gpulstmlayer->Wo;
 
   ///
   devRi = gpulstmlayer->Ri;
   devRf = gpulstmlayer->Rf;
   devRg = gpulstmlayer->Rg;
   devRo = gpulstmlayer->Ro;

   devBi = gpulstmlayer->bi;
   devBf = gpulstmlayer->bf;
   devBg = gpulstmlayer->bg;
   devBo = gpulstmlayer->bo;
 
   ///
   xi = gpulstmlayer->xi;
   xf = gpulstmlayer->xf;
   xg = gpulstmlayer->xg;
   xo = gpulstmlayer->xo;
    
   ///
   hval = gpulstmlayer->store.h;
   cval = gpulstmlayer->store.c;
   ival = gpulstmlayer->store.i;
   fval = gpulstmlayer->store.f;
   gval = gpulstmlayer->store.g;
   oval = gpulstmlayer->store.o;

   /// 
   devHpr = gpulstmlayer->hiddenState;
   devCpr = gpulstmlayer->cellState;
 

    /// doing block and grid setting for multiplication functions
    int blockSize = 16;
    int maxSize = blockSize * blockSize;
    int blocksPerGrid = numHiddenUnits/maxSize + 1;
    dim3 dimBlock(blockSize, blockSize);

    if (numHiddenUnits%maxSize == 0)
    {
        blocksPerGrid--;
    }
 
    dim3 dimGrid(1, blocksPerGrid);
 
    //// computing xi, xf, xg, xo, by matrix vector multiplication of W and x
   gpu_matrixVectorMultWX<<< dimGrid , dimBlock >>>(devWi, devX, xi, numHiddenUnits, inputSize );
   gpu_matrixVectorMultWX<<< dimGrid , dimBlock >>>(devWf, devX, xf, numHiddenUnits, inputSize );
   gpu_matrixVectorMultWX<<< dimGrid , dimBlock >>>(devWg, devX, xg, numHiddenUnits, inputSize );
   gpu_matrixVectorMultWX<<< dimGrid , dimBlock >>>(devWo, devX, xo, numHiddenUnits, inputSize );


   if (condition == 0)
   {
        int threadInBlock = numHiddenUnits;
        int linBlockPerGrid = 1;
       
        gateValueComputation_Cond0<<< linBlockPerGrid , threadInBlock >>>(devHpr, devCpr, devRi, devRf, devRg, devRo, devBi, devBf, devBg, devBo,
                                                                              xi, xf, xg, xo, ival, fval, gval, oval, cval, hval, numHiddenUnits);
    
   }
   else if (condition == 1)
   {
       
       outputElementComputation_Cond1<<< 1, 1 >>>( devCpr, xi, xf, xg, xo, devBi, devBf, devBg, devBo, ival, fval, gval, oval, cval, hval, numHiddenUnits-1, numHiddenUnits);

       int threadInBlock = numHiddenUnits-1;
       int linBlockPerGrid = 1;

       gateValueComputation_Cond1<<< linBlockPerGrid , threadInBlock >>>(devHpr, devCpr, devRi, devRf, devRg, devRo, devBi, devBf, devBg, devBo,
                                                                              xi, xf, xg, xo, ival, fval, gval, oval, cval, hval, numHiddenUnits);   

   }
   else
   {
       printf(" Wrong input \n");
       return;
   }
    

   //// changing value of condition for next iteration 
   gpulstmlayer->store.evenOrOdd = 1 - condition;

   /// copy operations 
   cudaMemcpy(devHpr, hval, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToDevice);
   cudaMemcpy(devCpr, cval, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToDevice);

   /// re-initialization operations
   //cudaMemset(hval, 0, sizeof(double) * numHiddenUnits);
   //cudaMemset(cval, 0, sizeof(double) * numHiddenUnits);

   /// free gpu variables
  
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
   printf(" Variant2 (using for loop in kernel function) of Method 2 :-  GPU elementwise implementation  method :- \n");
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
      LSTMForwardStepReuse(&X[i*inputSize],&gpulstmlayer);
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


    printf("total time taken by gpu elementwise program (variant2 - for loop in kernel func) is %lf milliseconds for input size of %d , numHiddenUnits of %d and timeSteps of %d \n", total_time, inputSize, numHiddenUnits, timeSteps );
    // writeMatrixIntoFile(fpY,  Y, timeSteps, inputSize);
    // fclose(fpY);

    freeGpuLSTMLayer( &gpulstmlayer);
    free(W);
    free(R);
    free(bias);
    free(X);
    free(Y);
}