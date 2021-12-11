#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include <sys/time.h>
#include<time.h>

#define min(a,b) (a<b?a:b)
#define TILE_WIDTH 4

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

////////////////////////////////////////////////////////////////

typedef struct LstmStore {
   //for partial matrix vector results of gates for current and next time steps
   double *i,*f,*g,*o;
   double *c,*h; //present time step computations
   int evenOrOdd; //0 = even, 1 = Odd;

   double *onChipI; //on-chip storage for fixing part of Ri,Rf,Rg,Ro matrices data
   double *onChipF; //on-chip storage for fixing part of Ri,Rf,Rg,Ro matrices data
   double *onChipG; //on-chip storage for fixing part of Ri,Rf,Rg,Ro matrices data
   double *onChipO; //on-chip storage for fixing part of Ri,Rf,Rg,Ro matrices data
   int squareDim; // dimensions beyond which data is stored on-chip
}LstmStore;

typedef struct LstmLayer{
   int inputSize;
   int numHiddenUnits;
   double *Wi,*Wf,*Wg,*Wo;
   double *Ri,*Rf,*Rg,*Ro;
   double *bi,*bf,*bg,*bo;
   double *hiddenState;
   double *cellState;

   LstmStore store; //storage for temporary variables
}LstmLayer;

//////////////////////////////////////////////////////////////////

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


/////////////////////////////////////////////////
   ////BLOCKWISE MAIN CODE ////

__device__ void getChunkOfArray(double *temp, double *store, int index, int rCeil, int numHiddenUnits)
{
     int rind = index * TILE_WIDTH;
 
     for (int i=0;i<TILE_WIDTH;i++)
     {
       temp[i] = store[rind + i];
     }
}

__device__ void getBlockOfArray2D(double *temp, double *store, int index, int rindex, int rCeil, int numHiddenUnits)
{
  int rind = rindex * TILE_WIDTH;
  int cind = index * TILE_WIDTH;
  
  for (int i=0;i<TILE_WIDTH;i++)
  {
   for (int j=0;j<TILE_WIDTH;j++)
    {
       temp[i*TILE_WIDTH+j] = store[ (rind+i) * numHiddenUnits + cind + j];
    }
  }
 
}


__device__ void getBlockChunkMultiply(double *temp, double *block, double *chunk, int index, int rindex, int rCeil, int numHiddenUnits)
{
    for (int i=0;i<TILE_WIDTH;i++)
    {
        double sum = 0.0;
        
        for (int j=0;j<TILE_WIDTH;j++)
        {
            sum += block[i*TILE_WIDTH +j] * chunk[j];
        }
        temp[i] += sum;
    }
}

__device__ void addGateValue(double *temp, double *store, int rindex, int rCeil, int numHiddenUnits)
{
    int rind = rindex * TILE_WIDTH;
 
    for (int i=0;i<TILE_WIDTH;i++)
    {
        temp[i] += store[rind + i];
    }
}

__device__ double sigmoid_elementwise(double val)
{
    return 1/(1+exp(-val));
}

__device__ void sigmoid_blockwise(double *temp, double *x, double *bias, int rindex, int rCeil, int numHiddenUnits)
{
    int rind = rindex * TILE_WIDTH;

    for (int i=0;i<TILE_WIDTH;i++)
    {
        temp[i] = sigmoid_elementwise(temp[i] + x[rind + i] + bias[rind + i]);
    } 
}

__device__ void tanh_blockwise(double *temp, double *x, double *bias, int rindex, int rCeil, int numHiddenUnits)
{
    int rind = rindex * TILE_WIDTH;

    for (int i=0;i<TILE_WIDTH;i++)
    {
        temp[i] = tanh(temp[i] + x[rind + i] + bias[rind + i]);
    } 
}

__device__ void cellAndHiddenVal_blockwise(double *hval, double *cval, double *devCpr,
                                           double *iPres, double *fPres, double *gPres, double *oPres,
                                           int rindex, int rCeil, int numHiddenUnits)
{
    int rind = rindex * TILE_WIDTH;

    for (int i=0;i<TILE_WIDTH;i++)
    {
        cval[rind + i] = fPres[i] * devCpr[rind + i] + iPres[i] * gPres[i];
        hval[rind + i] = oPres[i] * tanh(cval[rind + i]);
    }
}

__device__ void assignNextGateValue(double *store, double *temp, int rindex, int rCeil, int numHiddenUnits)
{
    int rind = rindex * TILE_WIDTH;

    for (int i=0;i<TILE_WIDTH;i++)
    {
        store[rind + i] = temp[i];
    }
}

__device__ void assignZeroGateValue(double *store, int rindex, int rCeil, int numHiddenUnits)
{
    int rind = rindex * TILE_WIDTH;

    for (int i=0;i<TILE_WIDTH;i++)
    {
        store[rind + i] = 0.0;
    }
}

//// Maximum size of hiddenunits for which shared memory will not overflow for the case of double type ////
#define SMEM_SIZE 768

__global__ void gateValueComputation_Cond0(double *devHpr, double *devCpr,
                                           double *devRi, double *devRf, double *devRg, double *devRo,
                                           double *devBi, double *devBf, double *devBg, double *devBo,
                                           double *xi, double *xf, double *xg, double *xo,
                                           double *ival, double *fval, double *gval, double *oval, double *cval, double *hval,
                                           int rCeil, int numHiddenUnits)
{
    
   __shared__ double iPres[SMEM_SIZE], fPres[SMEM_SIZE], gPres[SMEM_SIZE], oPres[SMEM_SIZE];
   __shared__ double iNext[SMEM_SIZE], fNext[SMEM_SIZE], gNext[SMEM_SIZE], oNext[SMEM_SIZE];

   int index = blockDim.x * blockIdx.x + threadIdx.x;

   if (index == 0)
   {
       for (int i=0;i<numHiddenUnits;i++)
       {
           iPres[i] = ival[i];
           fPres[i] = fval[i];
           gPres[i] = gval[i];
           oPres[i] = oval[i];

           iNext[i] = 0.0;
           fNext[i] = 0.0;
           gNext[i] = 0.0;
           oNext[i] = 0.0;
       }
   }

   __syncthreads();

   for (int i=0;i<rCeil;i++)
   {
       if (index == i)
       {
           
           double tmpHpr[TILE_WIDTH];
           getChunkOfArray(tmpHpr, devHpr, i , rCeil, numHiddenUnits);

           int rind = i * TILE_WIDTH;
           
           double tRi[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRi, devRi, i, i, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&iPres[rind], tRi, tmpHpr, i, i, rCeil, numHiddenUnits);

           double tRf[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRf, devRf, i, i, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&fPres[rind], tRf, tmpHpr, i, i, rCeil, numHiddenUnits);

           double tRg[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRg, devRg, i, i, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&gPres[rind], tRg, tmpHpr, i, i, rCeil, numHiddenUnits);
        
           double tRo[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRo, devRo, i, i, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&oPres[rind], tRo, tmpHpr, i, i, rCeil, numHiddenUnits);
        
           sigmoid_blockwise(&iPres[rind], xi, devBi, i, rCeil, numHiddenUnits);
           sigmoid_blockwise(&fPres[rind], xf, devBf, i, rCeil, numHiddenUnits);
           tanh_blockwise(&gPres[rind], xg, devBg, i, rCeil, numHiddenUnits);
           sigmoid_blockwise(&oPres[rind], xo, devBo, i, rCeil, numHiddenUnits);
          
           cellAndHiddenVal_blockwise(hval, cval, devCpr, &iPres[rind], &fPres[rind], &gPres[rind], &oPres[rind], i, rCeil, numHiddenUnits);
        
           double tmphval[TILE_WIDTH];
           getChunkOfArray(tmphval, hval, i, rCeil, numHiddenUnits);

           getBlockChunkMultiply(&iNext[rind], tRi, tmphval, i, i, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&fNext[rind], tRf, tmphval, i, i, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&gNext[rind], tRg, tmphval, i, i, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&iNext[rind], tRi, tmphval, i, i, rCeil, numHiddenUnits);
        
           assignNextGateValue(ival, &iNext[rind], i, rCeil, numHiddenUnits);
           assignNextGateValue(fval, &fNext[rind], i, rCeil, numHiddenUnits);
           assignNextGateValue(gval, &gNext[rind], i, rCeil, numHiddenUnits);
           assignNextGateValue(oval, &oNext[rind], i, rCeil, numHiddenUnits);
         
       }

       __syncthreads();
    
       if (index > i)
       {
           
           double tmpHpr[TILE_WIDTH];
           double tmphval[TILE_WIDTH];
           
           getChunkOfArray(tmpHpr, devHpr, i , rCeil, numHiddenUnits);
           getChunkOfArray(tmphval, hval, i, rCeil, numHiddenUnits);

           int rind = index * TILE_WIDTH;
        
           double tRi[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRi, devRi, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&iPres[rind], tRi, tmpHpr, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&iNext[rind], tRi, tmphval, i, index, rCeil, numHiddenUnits);

           double tRf[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRf, devRf, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&fPres[rind], tRf, tmpHpr, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&fNext[rind], tRf, tmphval, i, index, rCeil, numHiddenUnits);

           double tRg[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRg, devRg, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&gPres[rind], tRg, tmpHpr, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&gNext[rind], tRg, tmphval, i, index, rCeil, numHiddenUnits);
        
           double tRo[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRo, devRo, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&oPres[rind], tRo, tmpHpr, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&oNext[rind], tRo, tmphval, i, index, rCeil, numHiddenUnits);
              
       }

       __syncthreads();
   }

}  


__global__ void outputElementComputation_Cond1( double *devCpr, double *xi, double *xf, double *xg, double *xo,
                                               double *devBi, double *devBf, double *devBg, double *devBo,
                                               double *ival, double *fval, double *gval, double *oval, double *cval, double *hval,
                                               int rindex, int rCeil, int numHiddenUnits)
{
    int rind = rindex * TILE_WIDTH;
 
    sigmoid_blockwise(&ival[rind], xi, devBi, rindex, rCeil, numHiddenUnits);
    sigmoid_blockwise(&fval[rind], xf, devBf, rindex, rCeil, numHiddenUnits);
    tanh_blockwise(&gval[rind], xg, devBg, rindex, rCeil, numHiddenUnits);
    sigmoid_blockwise(&oval[rind], xo, devBo, rindex, rCeil, numHiddenUnits);

    cellAndHiddenVal_blockwise(hval, cval, devCpr, &ival[rind], &fval[rind], &gval[rind], &oval[rind], rindex, rCeil, numHiddenUnits);

    if (rindex == rCeil-1)
    {
       assignZeroGateValue(ival, rindex, rCeil, numHiddenUnits);   
       assignZeroGateValue(fval, rindex, rCeil, numHiddenUnits); 
       assignZeroGateValue(gval, rindex, rCeil, numHiddenUnits); 
       assignZeroGateValue(oval, rindex, rCeil, numHiddenUnits); 
    }
    
}


__global__ void gateValueComputation_Cond1( double *devHpr, double *devCpr,
                                          double *devRi, double *devRf, double *devRg, double *devRo,
                                          double *devBi, double *devBf, double *devBg, double *devBo,
                                          double *xi, double *xf, double *xg, double *xo,
                                          double *ival, double *fval, double *gval, double *oval, double *cval, double *hval,
                                          int rCeil, int numHiddenUnits)
{

   __shared__ double iPres[SMEM_SIZE], fPres[SMEM_SIZE], gPres[SMEM_SIZE], oPres[SMEM_SIZE];
   __shared__ double iNext[SMEM_SIZE], fNext[SMEM_SIZE], gNext[SMEM_SIZE], oNext[SMEM_SIZE];

   int index = blockDim.x * blockIdx.x + threadIdx.x;

   if (index == 0)
   {
       for (int i=0;i<numHiddenUnits;i++)
       {
           iPres[i] = ival[i];
           fPres[i] = fval[i];
           gPres[i] = gval[i];
           oPres[i] = oval[i];
        
           iNext[i] = 0.0;
           fNext[i] = 0.0;
           gNext[i] = 0.0;
           oNext[i] = 0.0;
       }
   }
 
   __syncthreads();

   for (int i=rCeil-1;i>=1;i--)
   {
       if (index < i)
       {
           double tmpHpr[TILE_WIDTH];
           double tmphval[TILE_WIDTH];
           
           getChunkOfArray(tmpHpr, devHpr, i , rCeil, numHiddenUnits);
           getChunkOfArray(tmphval, hval, i, rCeil, numHiddenUnits);

           int rind = index * TILE_WIDTH;
        
           double tRi[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRi, devRi, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&iPres[rind], tRi, tmpHpr, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&iNext[rind], tRi, tmphval, i, index, rCeil, numHiddenUnits); 

           double tRf[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRf, devRf, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&fPres[rind], tRf, tmpHpr, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&fNext[rind], tRf, tmphval, i, index, rCeil, numHiddenUnits);

           double tRg[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRg, devRg, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&gPres[rind], tRg, tmpHpr, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&gNext[rind], tRg, tmphval, i, index, rCeil, numHiddenUnits);
        
           double tRo[TILE_WIDTH * TILE_WIDTH];
           getBlockOfArray2D( tRo, devRo, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&oPres[rind], tRo, tmpHpr, i, index, rCeil, numHiddenUnits);
           getBlockChunkMultiply(&oNext[rind], tRo, tmphval, i, index, rCeil, numHiddenUnits);
        
       }

       __syncthreads();

       if (index == i-1)
       {
           int rind = index * TILE_WIDTH;

           sigmoid_blockwise(&iPres[rind], xi, devBi, index, rCeil, numHiddenUnits);
           sigmoid_blockwise(&fPres[rind], xf, devBf, index, rCeil, numHiddenUnits);
           tanh_blockwise(&gPres[rind], xg, devBg, index, rCeil, numHiddenUnits);
           sigmoid_blockwise(&oPres[rind], xo, devBo, index, rCeil, numHiddenUnits);
          
           cellAndHiddenVal_blockwise(hval, cval, devCpr, &iPres[rind], &fPres[rind], &gPres[rind], &oPres[rind], index, rCeil, numHiddenUnits);

           assignNextGateValue(ival, &iNext[rind], index, rCeil, numHiddenUnits);
           assignNextGateValue(fval, &fNext[rind], index, rCeil, numHiddenUnits);
           assignNextGateValue(gval, &gNext[rind], index, rCeil, numHiddenUnits);
           assignNextGateValue(oval, &oNext[rind], index, rCeil, numHiddenUnits);
        
           
       }

       __syncthreads();
   }
 
}   


void GpuLSTMForwardStepBlockReuse(double *x, GpuLstmLayer *gpulstmlayer)
{
   int inputSize = gpulstmlayer->inputSize;
   int numHiddenUnits = gpulstmlayer->numHiddenUnits;
   int condition = gpulstmlayer->store.evenOrOdd;

   if (numHiddenUnits%TILE_WIDTH != 0 || TILE_WIDTH > numHiddenUnits || SMEM_SIZE < numHiddenUnits)
   {
       printf(" wrong input of inputsize and number of hidden units \n");
       return;
   }

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

   double *writer;
   writer = (double*)malloc(sizeof(double) * numHiddenUnits);

   printf("\n printing the required values :- \n ");
 
   printf(" printing previous hidden state :- \n ");
   cudaMemcpy(writer, devHpr, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToHost);
   for (int i=0;i<numHiddenUnits;i++)
   {
       printf("%lf ", writer[i]);
   }
   printf("\n");

   printf(" printing previous cell state :- \n");
   cudaMemcpy(writer, devCpr, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToHost);
   for (int i=0;i<numHiddenUnits;i++)
   {
       printf("%lf ", writer[i]);
   }
   printf("\n");

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
        int rCeil = (numHiddenUnits + TILE_WIDTH - 1)/TILE_WIDTH;
    
        int threadInBlock = rCeil;
        int linBlockPerGrid = 1;
    
        gateValueComputation_Cond0<<< linBlockPerGrid , threadInBlock >>>(devHpr, devCpr, devRi, devRf, devRg, devRo, devBi, devBf, devBg, devBo,
                                                                              xi, xf, xg, xo, ival, fval, gval, oval, cval, hval, rCeil, numHiddenUnits);
    
   }
   else if (condition == 1)
   { 
       int rCeil = (numHiddenUnits + TILE_WIDTH - 1)/TILE_WIDTH;

       outputElementComputation_Cond1<<< 1, 1 >>>( devCpr, xi, xf, xg, xo, devBi, devBf, devBg, devBo, ival, fval, gval, oval, cval, hval, rCeil-1, rCeil, numHiddenUnits);

       int threadInBlock = rCeil-1;
       int linBlockPerGrid = 1;

       if (threadInBlock != 0)
       {
           gateValueComputation_Cond1<<< linBlockPerGrid , threadInBlock >>>(devHpr, devCpr, devRi, devRf, devRg, devRo, devBi, devBf, devBg, devBo,
                                                                              xi, xf, xg, xo, ival, fval, gval, oval, cval, hval, rCeil, numHiddenUnits);   
       }
   }
   else
   {
       printf(" Wrong input \n");
       return;
   }

     printf(" printing input gate  :- \n ");
    cudaMemcpy(writer, ival, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToHost);
    for (int i=0;i<numHiddenUnits;i++)
    {
       printf("%lf ", writer[i]);
    }
    printf("\n");

    printf(" printing forget gate  :- \n ");
    cudaMemcpy(writer, fval, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToHost);
    for (int i=0;i<numHiddenUnits;i++)
    {
       printf("%lf ", writer[i]);
    }
    printf("\n");



    printf(" printing cell state  :- \n ");
    cudaMemcpy(writer, cval, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToHost);
    for (int i=0;i<numHiddenUnits;i++)
    {
       printf("%lf ", writer[i]);
    }
    printf("\n");

     printf(" printing cell input gate  :- \n ");
    cudaMemcpy(writer, gval, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToHost);
    for (int i=0;i<numHiddenUnits;i++)
    {
       printf("%lf ", writer[i]);
    }
    printf("\n");

    printf(" printing output gate  :- \n ");
    cudaMemcpy(writer, oval, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToHost);
    for (int i=0;i<numHiddenUnits;i++)
    {
       printf("%lf ", writer[i]);
    }
    printf("\n");

    printf(" printing hidden state  :- \n ");
    cudaMemcpy(writer, hval, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToHost);
    for (int i=0;i<numHiddenUnits;i++)
    {
       printf("%lf ", writer[i]);
    }
    printf("\n");
    

   //// changing value of condition for next iteration 
   gpulstmlayer->store.evenOrOdd = 1 - condition;

   /// copy operations 
   cudaMemcpy(devHpr, hval, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToDevice);
   cudaMemcpy(devCpr, cval, sizeof(double) * numHiddenUnits, cudaMemcpyDeviceToDevice);

   /// re-initialization operations
   //cudaMemset(hval, 0, sizeof(double) * numHiddenUnits);
   //cudaMemset(cval, 0, sizeof(double) * numHiddenUnits);

   /// free gpu variables
   //free(writer);
 
}


///////////////////////////////////////////////



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


///////////////////////////////////////////////////////////

void freeLSTMLayer(LstmLayer *lstmlayer)
{
    free(lstmlayer->store.i);
    free(lstmlayer->store.f);
    free(lstmlayer->store.g);
    free(lstmlayer->store.o);

    free(lstmlayer->store.c);
    free(lstmlayer->store.h);

    // free(lstmlayer->Wi); //these were allocated outside createLSTMLayer
    // free(lstmlayer->Ri);//these were allocated outside createLSTMLayer
    // free(lstmlayer->bi);
    free(lstmlayer->hiddenState);
    free(lstmlayer->cellState);
}


void createLSTMLayer(LstmLayer *lstmlayer,int numHiddenUnits,int inputSize,
                                            double *W,double *R,double *bias)
{
    lstmlayer->Wi = W;
    lstmlayer->Wf = lstmlayer->Wi+numHiddenUnits*inputSize;
    lstmlayer->Wg = lstmlayer->Wf+numHiddenUnits*inputSize;
    lstmlayer->Wo = lstmlayer->Wg+numHiddenUnits*inputSize;

    lstmlayer->Ri = R;
    lstmlayer->Rf = lstmlayer->Ri+numHiddenUnits*numHiddenUnits;
    lstmlayer->Rg = lstmlayer->Rf+numHiddenUnits*numHiddenUnits;
    lstmlayer->Ro = lstmlayer->Rg+numHiddenUnits*numHiddenUnits;

    lstmlayer->bi = bias;
    lstmlayer->bf = lstmlayer->bi+numHiddenUnits;
    lstmlayer->bg = lstmlayer->bf+numHiddenUnits;
    lstmlayer->bo = lstmlayer->bg+numHiddenUnits;

    //initialize the cell state and hidden units
    lstmlayer->hiddenState =(double*)malloc(sizeof(double)*numHiddenUnits);
    lstmlayer->cellState =(double*)malloc(sizeof(double)*numHiddenUnits);
    
    memset(lstmlayer->hiddenState,0,sizeof(double)*numHiddenUnits);
    memset(lstmlayer->cellState,0,sizeof(double)*numHiddenUnits);
    
    lstmlayer->inputSize = inputSize;
    lstmlayer->numHiddenUnits = numHiddenUnits;

    //memory for gates, even and odd 
    lstmlayer->store.i = (double*)malloc(sizeof(double) * numHiddenUnits);
    memset(lstmlayer->store.i,0,sizeof(double)*numHiddenUnits);
    lstmlayer->store.f = (double*)malloc(sizeof(double) * numHiddenUnits);
    memset(lstmlayer->store.f,0,sizeof(double)*numHiddenUnits);
    lstmlayer->store.g = (double*)malloc(sizeof(double) * numHiddenUnits);
    memset(lstmlayer->store.g,0,sizeof(double)*numHiddenUnits);
    lstmlayer->store.o = (double*)malloc(sizeof(double) * numHiddenUnits);
    memset(lstmlayer->store.o,0,sizeof(double)*numHiddenUnits);

    lstmlayer->store.c = (double*)malloc(sizeof(double) * numHiddenUnits);
    memset(lstmlayer->store.c,0,sizeof(double)*numHiddenUnits);

    lstmlayer->store.h = (double*)malloc(sizeof(double) * numHiddenUnits);
    memset(lstmlayer->store.h,0,sizeof(double)*numHiddenUnits);

    lstmlayer->store.evenOrOdd = 0; //start with even, lower diagonal matrix.
}

double sigmoid(double x)
{
    return 1/(1+exp(-x));
}


void matrixVectorMult(double* W, double* h, double* hOut,int Rows, int Cols)
{

  for(int r=0;r<Rows;r++) {
     //hOut[r] = 0;
     for(int c = 0; c< Cols; c++) {
        hOut[r] += W[r*Cols+c]*h[c];

     }
  }

}

void CPULSTMForwardStepReuse(double * x, LstmLayer *lstmlayer)
{
    double *hPr =lstmlayer->hiddenState; 
    double *cPr =lstmlayer->cellState;
    double *h = lstmlayer->store.h;
    double *c = lstmlayer->store.c;

    int numHiddenUnits=lstmlayer->numHiddenUnits; 
    int inputSize=lstmlayer->inputSize;

    double *Wi=lstmlayer->Wi;
    double *Wf=lstmlayer->Wf;
    double *Wg=lstmlayer->Wg;
    double *Wo=lstmlayer->Wo;
    
    double *Ri=lstmlayer->Ri;
    double *Rf=lstmlayer->Rf;
    double *Rg=lstmlayer->Rg;
    double *Ro=lstmlayer->Ro;

    double *bi=lstmlayer->bi;
    double *bf=lstmlayer->bf;
    double *bg=lstmlayer->bg;
    double *bo=lstmlayer->bo;
   
    double *i = lstmlayer->store.i;
    double *f = lstmlayer->store.f;
    double *g = lstmlayer->store.g;
    double *o = lstmlayer->store.o;

    printf("\n printing the required values :- \n");
 
    printf(" printing prev hidden state :- \n");
    for (int n=0;n<numHiddenUnits;n++)
    {
        printf("%lf ", hPr[n]);
    }
    printf("\n");
    
    printf(" printing prev cell state :- \n");
    for (int n=0;n<numHiddenUnits;n++)
    {
        printf("%lf ", cPr[n]);
    }
    printf("\n");

    int accWh=0;    
    
    double* xi = (double*)malloc(sizeof(double) * numHiddenUnits);
    double* xf = (double*)malloc(sizeof(double) * numHiddenUnits);
    double* xg = (double*)malloc(sizeof(double) * numHiddenUnits);
    double* xo = (double*)malloc(sizeof(double) * numHiddenUnits);

    memset(xi,0,sizeof(double)*numHiddenUnits);
    memset(xf,0,sizeof(double)*numHiddenUnits);
    memset(xg,0,sizeof(double)*numHiddenUnits);
    memset(xo,0,sizeof(double)*numHiddenUnits);

    matrixVectorMult(Wi, x, xi, numHiddenUnits, inputSize);
    matrixVectorMult(Wf, x, xf, numHiddenUnits, inputSize);
    matrixVectorMult(Wg, x, xg, numHiddenUnits, inputSize);
    matrixVectorMult(Wo, x, xo, numHiddenUnits, inputSize);

    double tRi,tRf,tRg,tRo; //temporary variables
    double iNext,fNext,gNext,oNext; //accumlates the temporary computations for next time step for parital row of gate matrices

    if(lstmlayer->store.evenOrOdd == 0) {
      //loop all lower diagonal and diagonal elements 

      //lower diagonal element
      for(int r=0;r<numHiddenUnits;r++) {
        // for next time step values
        iNext=fNext=gNext=oNext=0;
        // for current time step , get the values computed previously from store
        for(int cc=0;cc<=r;cc++) {

            tRi = Ri[r*numHiddenUnits+cc];
            tRf = Rf[r*numHiddenUnits+cc];
            tRg = Rg[r*numHiddenUnits+cc];
            tRo = Ro[r*numHiddenUnits+cc];
            accWh += 4;
   
            //process all gates i,f,g,o to compute i[r] and inext[r]
            i[r] += tRi*hPr[cc];
            f[r] += tRf*hPr[cc];
            g[r] += tRg*hPr[cc];
            o[r] += tRo*hPr[cc];
           
            if( cc == r) {
                i[r] = sigmoid(i[r]+ xi[r]+bi[r]);
                f[r] = sigmoid(f[r]+ xf[r]+bf[r]);
                g[r] = tanh(g[r] + xg[r]+bg[r]);
                o[r] = sigmoid(o[r]+ xo[r]+bo[r]);
                c[r] = f[r]*cPr[r] + i[r]*g[r];
                h[r] = o[r]*tanh(c[r]);
            }

            iNext += tRi*h[cc];
            fNext += tRf*h[cc];
            gNext += tRg*h[cc];
            oNext += tRo*h[cc];
        }
            i[r] = iNext;
            f[r] = fNext;
            g[r] = gNext;
            o[r] = oNext;        
      }
      
      //expect c and h to be ready and now copy to cPr and hPr
      memcpy(hPr,h,sizeof(double)*numHiddenUnits);
      memset(h,0,sizeof(double)*numHiddenUnits);

      memcpy(cPr,c,sizeof(double)*numHiddenUnits);
      memset(c,0,sizeof(double)*numHiddenUnits);

      lstmlayer->store.evenOrOdd = 1;

    }
    else if (lstmlayer->store.evenOrOdd == 1) {

        //loop all upper diagonal elements from bottom to top, right to left
      for(int r=numHiddenUnits-1;r>=0;r--) {
        iNext=fNext=gNext=oNext=0;
        for(int cc=numHiddenUnits-1;cc>r;cc--) {

            tRi = Ri[r*numHiddenUnits+cc];
            tRf = Rf[r*numHiddenUnits+cc];
            tRg = Rg[r*numHiddenUnits+cc];
            tRo = Ro[r*numHiddenUnits+cc];
            accWh += 4;

            i[r] += tRi*hPr[cc];
            f[r] += tRf*hPr[cc];
            g[r] += tRg*hPr[cc];
            o[r] += tRo*hPr[cc];

            iNext += tRi*h[cc];
            fNext += tRf*h[cc];
            gNext += tRg*h[cc];
            oNext += tRo*h[cc];
        }
        i[r] = sigmoid(i[r]+ xi[r]+bi[r]);
        f[r] = sigmoid(f[r]+ xf[r]+bf[r]);
        g[r] = tanh(g[r]+ xg[r]+bg[r]);
        o[r] = sigmoid(o[r]+ xo[r]+bo[r]);
        c[r] = f[r]*cPr[r] + i[r]*g[r];
        h[r] = o[r]*tanh(c[r]);

        i[r] = iNext;
        f[r] = fNext;
        g[r] = gNext;
        o[r] = oNext;        
      }

      //expect c and h to be ready and now copy to cPr and hPr
      memcpy(hPr,h,sizeof(double)*numHiddenUnits);
      memset(h,0,sizeof(double)*numHiddenUnits);

      memcpy(cPr,c,sizeof(double)*numHiddenUnits);
      memset(c,0,sizeof(double)*numHiddenUnits);

      lstmlayer->store.evenOrOdd = 0;
    }
    else {
        printf("evenOrOdd flag incorrect value\n");
        return ;
    }
    
     printf(" printing input gate :- \n");
    for (int n=0;n<numHiddenUnits;n++)
    {
        printf("%lf ", i[n]);
    }
    printf("\n");

    printf(" printing forget gate :- \n");
    for (int n=0;n<numHiddenUnits;n++)
    {
        printf("%lf ", f[n]);
    }
    printf("\n");
 
    printf(" printing cell input gate :- \n");
    for (int n=0;n<numHiddenUnits;n++)
    {
        printf("%lf ", g[n]);
    }
    printf("\n");

    printf(" printing output gate :- \n");
    for (int n=0;n<numHiddenUnits;n++)
    {
        printf("%lf ", o[n]);
    }
    printf("\n");


    printf(" printing present cell state :- \n");
    for (int n=0;n<numHiddenUnits;n++)
    {
        printf("%lf ", cPr[n]);
    }
    printf("\n");

    printf(" printing presnet hidden state :- \n");
    for (int n=0;n<numHiddenUnits;n++)
    {
      printf("%lf ", hPr[n]);
    }
    printf("\n \n");
    
    //  i = extractdata(sigmoid(dlarray(Wi * x + Ri * hPr + bi)));
      //  f = extractdata(sigmoid(dlarray(Wf * x + Rf * hPr + bf)));
     //  g = tanh(Wg * x + Rg * hPr + bg);
     //  o = extractdata(sigmoid(dlarray(Wo * x + Ro * hPr + bo)));
     //  c = f.*cPr + i.*g;
    //  h = o .* tanh(c);
    //  %update for the next time step
    //  hPr = h;
    //  cPr = c;
 
    // printf("Access : inputWts:%d reurWeights:%d\n",accWx,accWh);
    free(xi);
    free(xf);
    free(xg);
    free(xo);
}

void CpuLSTMForwardStep(double * x, LstmLayer *lstmlayer)
{
    double *hPr =lstmlayer->hiddenState; 
    double *cPr =lstmlayer->cellState;
    int numHiddenUnits=lstmlayer->numHiddenUnits; 
    int inputSize=lstmlayer->inputSize;
 
    //printf("\n printing the required values :- \n");
 
    //printf(" printing prev hidden state :- \n");
    //for (int n=0;n<numHiddenUnits;n++)
    //{
      //  printf("%lf ", hPr[n]);
    //}
    //printf("\n");
    
   // printf(" printing prev cell state :- \n");
    //for (int n=0;n<numHiddenUnits;n++)
    //{
      //  printf("%lf ", cPr[n]);
    //}
    //printf("\n");

    double *Wi=lstmlayer->Wi;
    double *Wf=lstmlayer->Wf;
    double *Wg=lstmlayer->Wg;
    double *Wo=lstmlayer->Wo;
    
    double *Ri=lstmlayer->Ri;
    double *Rf=lstmlayer->Rf;
    double *Rg=lstmlayer->Rg;
    double *Ro=lstmlayer->Ro;

    double *bi=lstmlayer->bi;
    double *bf=lstmlayer->bf;
    double *bg=lstmlayer->bg;
    double *bo=lstmlayer->bo;

    double *temp1,*temp2;
    double *i,*f,*g,*o;

    temp1 = (double*)malloc(sizeof(double) * numHiddenUnits);
    temp2 = (double*)malloc(sizeof(double) * numHiddenUnits);

    //memory for gates
    i = (double*)malloc(sizeof(double) * numHiddenUnits);
    f = (double*)malloc(sizeof(double) * numHiddenUnits);
    g = (double*)malloc(sizeof(double) * numHiddenUnits);
    o = (double*)malloc(sizeof(double) * numHiddenUnits);

    //  i = extractdata(sigmoid(dlarray(Wi * x + Ri * hPr + bi)));
    memset(temp1,0,sizeof(double)*numHiddenUnits);
    memset(temp2,0,sizeof(double)*numHiddenUnits);
    matrixVectorMult(Wi, x, temp1, numHiddenUnits, inputSize);
    matrixVectorMult(Ri, hPr, temp2, numHiddenUnits, numHiddenUnits);
    for(int n=0;n<numHiddenUnits;n++) {
        i[n] = sigmoid(temp1[n] + temp2[n] + bi[n]);
    }

     
    //  f = extractdata(sigmoid(dlarray(Wf * x + Rf * hPr + bf)));
    memset(temp1,0,sizeof(double)*numHiddenUnits);
    memset(temp2,0,sizeof(double)*numHiddenUnits);
    matrixVectorMult(Wf, x, temp1,numHiddenUnits, inputSize);
    matrixVectorMult(Rf, hPr, temp2,numHiddenUnits, numHiddenUnits);
    for(int n=0;n<numHiddenUnits;n++) {
        f[n] = sigmoid(temp1[n] + temp2[n] + bf[n]);
     }

    //  g = tanh(Wg * x + Rg * hPr + bg);
    memset(temp1,0,sizeof(double)*numHiddenUnits);
    memset(temp2,0,sizeof(double)*numHiddenUnits);
    matrixVectorMult(Wg, x, temp1,numHiddenUnits, inputSize);
    matrixVectorMult(Rg, hPr, temp2,numHiddenUnits, numHiddenUnits);
    for(int n=0;n<numHiddenUnits;n++) {
         g[n] = tanh(temp1[n] + temp2[n] + bg[n]);
     }

    //  o = extractdata(sigmoid(dlarray(Wo * x + Ro * hPr + bo)));
    memset(temp1,0,sizeof(double)*numHiddenUnits);
    memset(temp2,0,sizeof(double)*numHiddenUnits);
    matrixVectorMult(Wo, x, temp1,numHiddenUnits, inputSize);
    matrixVectorMult(Ro, hPr, temp2,numHiddenUnits, numHiddenUnits);    
    for(int n=0;n<numHiddenUnits;n++) {
         o[n] = sigmoid(temp1[n] + temp2[n] + bo[n]);
     }

    //  c = f.*cPr + i.*g;
    //  h = o .* tanh(c);
    //  %update for the next time step
    //  hPr = h;
    //  cPr = c;
    for(int n=0;n<numHiddenUnits;n++) {
         cPr[n] = f[n]*cPr[n] + i[n]*g[n];
         hPr[n] = o[n]*tanh(cPr[n]);
     }
 


      //  printf(" printing present cell state :- \n");
    //for (int n=0;n<numHiddenUnits;n++)
    //{
      //  printf("%lf ", cPr[n]);
    //}
    //printf("\n");

      //  printf(" printing presnet hidden state :- \n");
    //for (int n=0;n<numHiddenUnits;n++)
    //{
     //   printf("%lf ", hPr[n]);
    //}
    //printf("\n \n");

    free(temp1);
    free(temp2);
    free(i);
    free(f);
    free(g);
    free(o);
}


/////////////////////////////////////////////////

int verifyMatrices(double *H1,double *H2,int R,int C)
{
   int mismatches = 0;

   for(int r=0;r<R;r++) {
      for(int c=0;c<C;c++) {
         if(fabs(H1[r*C+c]-H2[r*C+c]) > 0.0001) {
             // printf("%f:%f\n",H1[r*C+c],H2[r*C+c]);
             // printf(" H1[%d][%d]:%f,H2[%d][%d]:%f \n",r,c,H1[r*C+c],r,c,H2[r*C+c]);
            mismatches += 1;
         }
      }
   }
   if(mismatches != 0) {
      printf("Test Failed mismatches:%d\n",mismatches);

   }
   else {
      printf("Test Passed\n");
   }
   return mismatches;
}


int main()
{
   printf(" Variant3 (doing computation left to right (cond0) and right to left (cond1)) of Method 3 :-  GPU blockwise implementation  method :- \n");
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

   inputSize = 8;
   numHiddenUnits = 8;
   timeSteps = 2;
   int tileSize = TILE_WIDTH;
   int smemSize = SMEM_SIZE;
 
   printf(" inputSize = %d , numHiddenUnits = %d , timeSteps = %d , tileSize = %d , smemSize = %d \n", inputSize, numHiddenUnits, timeSteps, tileSize, smemSize);
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
   double *Y, *Y1;

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
    Y1 = (double*)malloc(sizeof(double) * numHiddenUnits * timeSteps);

    initializeMatrix(W, 4*numHiddenUnits, inputSize);
    initializeMatrix(R, 4*numHiddenUnits, numHiddenUnits);
    initializeMatrix(bias, 4*numHiddenUnits, 1);
    initializeMatrix(X, inputSize, timeSteps);
    // initializeMatrix(Y, numHiddenUnits, timeSteps);
    memset(Y, 0, sizeof(numHiddenUnits * timeSteps));
    memset(Y1, 0, sizeof(numHiddenUnits * timeSteps));

    // Y = (double*)malloc(sizeof(double) * numHiddenUnits * timeSteps);
    // memset(Y, 0, sizeof(double) * numHiddenUnits * timeSteps);

    GpuLstmLayer gpulstmlayer;
    createGpuLSTMLayer(&gpulstmlayer,numHiddenUnits,inputSize,W,R,bias);

    double total_time = 0.0;
 
    //double startTime , endTime;
    struct timeval t1, t2;
    printf(" for the elementwise gpu code :- \n");
 
    for (int i=0;i<timeSteps;i++)
    {
     
      gettimeofday(&t1, 0);
      GpuLSTMForwardStepBlockReuse(&X[i*inputSize],&gpulstmlayer);
      cudaMemcpy(&Y[i*numHiddenUnits],gpulstmlayer.hiddenState,sizeof(double)*numHiddenUnits, cudaMemcpyDeviceToHost);
      gettimeofday(&t2, 0);
     
      double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
           
      total_time += time;
      printf(" time for this step is %lf milliseconds \n ", time);
      printf("time after %d time step is %lf milliseconds \n", i, total_time );
    }
    printf("\n");


    printf("total time taken by gpu blockwise program (variant3 - doing computation left to right (cond0) and right to left (cond1)) is %lf milliseconds for input size of %d , numHiddenUnits of %d and timeSteps of %d \n", total_time, inputSize, numHiddenUnits, timeSteps );
    // writeMatrixIntoFile(fpY,  Y, timeSteps, inputSize);
    // fclose(fpY);
 
    printf("\n");
    printf(" for the coventional cpu code :- \n ");
 
    LstmLayer lstmlayer;
    createLSTMLayer(&lstmlayer,numHiddenUnits,inputSize,W,R,bias);
 
    total_time = 0.0;
 
    for (int i=0;i<timeSteps;i++)
    {
     
      gettimeofday(&t1, 0);
      CPULSTMForwardStepReuse(&X[i*inputSize],&lstmlayer);
      memcpy(&Y1[i*numHiddenUnits],lstmlayer.hiddenState,sizeof(double)*numHiddenUnits);
      gettimeofday(&t2, 0);
     
      double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
           
      total_time += time;
      printf(" time for this step is %lf milliseconds \n ", time);
      printf("time after %d time step is %lf milliseconds \n", i, total_time );
    }
    printf("\n");
 
    printf("total time taken by covnetional cpu program is %lf milliseconds for input size of %d , numHiddenUnits of %d and timeSteps of %d \n", total_time, inputSize, numHiddenUnits, timeSteps );
 
    printf("\n");
 
    int misMatches = verifyMatrices(Y,Y1,timeSteps,numHiddenUnits);
    printf(" no. of misMatches = %d \n", misMatches);

    printf("print Y (gpu) :- \n");
    for (int i=0;i<timeSteps;i++)
    {
        for (int j=0;j<numHiddenUnits;j++)
        {
            printf("%lf ",Y[i*numHiddenUnits+j]);
        }
        printf("\n");
    }
    printf("\n");
 
    printf("print Y1 (cpu) :- \n");
    for (int i=0;i<timeSteps;i++)
    {
        for (int j=0;j<numHiddenUnits;j++)
        {
            printf("%lf ",Y1[i*numHiddenUnits+j]);
        }
        printf("\n");
    }
    printf("\n");
    

    freeLSTMLayer(&lstmlayer);
    freeGpuLSTMLayer( &gpulstmlayer);
    free(W);
    free(R);
    free(bias);
    free(X);
    free(Y);
    free(Y1);
}