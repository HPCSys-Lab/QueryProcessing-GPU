/*
 * HostManager.c
 *
 *  Created on: 07/12/2017
 *      Author: roussian
 */

#include "HostManager.cuh"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "../include/helper_cuda.h"
#include "FileManager.cuh"
#include "ParallelPrunningDaat.cuh"
#include "Structs.cuh"

//#define DOC_QUANTITY_IN_MEMORY 64

void setThresholdForANDQuery(float *dInitialThreshold, float* h_dUBList, int iTermNumber){
	printf("Verify if the query is AND... ");
	for (int i = 0; i < iTermNumber; ++i) {
		*dInitialThreshold += h_dUBList[i];
	}

	*dInitialThreshold = *dInitialThreshold * 0.9 ;
	printf("OK\n");
}

void setThresholdForANDQueryOnInvertedList(float *dInitialThreshold, float* h_dUBList,
										   int *query, int iTermNumberByQuery){
	printf("Verify if the query is AND... ");

	for (int i = 0; i < iTermNumberByQuery; ++i) {
		*dInitialThreshold += h_dUBList[ query[i] ];
	}

	*dInitialThreshold = *dInitialThreshold * 0.70;

	printf("OK\n");

}

void initializeSingleExperimentQuery(int* iTermNumber, float* dAverageDocLength,  int* iTopK,
						  	  	  	 int*** h_iDocIdList, int*** h_iLenghtList,  unsigned short int*** h_iFreqList,
						  	  	  	 float** h_dIdfList, float** h_dUBList, int** h_iDocNumberList, int* iPostingTotalNumber,
						  	  	  	 int* iMaxNumberInList, int iExperimentNumber){


	readQuery(iTermNumber, dAverageDocLength, iTopK, h_iDocIdList, h_iLenghtList,
			  h_iFreqList, h_dIdfList, h_dUBList, h_iDocNumberList,iExperimentNumber);

	for (int i = 0; i < *iTermNumber; ++i) {

		*iPostingTotalNumber +=  (*h_iDocNumberList)[i];

		if(*iMaxNumberInList < (*h_iDocNumberList)[i]){
			*iMaxNumberInList = (*h_iDocNumberList)[i];
		}
	}

	printf("#Terms: %hu\n",*iTermNumber);
	printf("Max: %i\n",*iMaxNumberInList);
	printf("Total: %i\n",*iPostingTotalNumber);

}

__host__ void initializeInvertedIndex(int* iTermNumberInVocabulary, float* dAverageDocLength, int*** h_iDocIdList,
							          int*** h_iLenghtList,  unsigned short int*** h_iFreqList,
							          float** h_dIdfList, float** h_dUBList, int** h_iDocNumberList,
							          unsigned long long* docTotalNumber){

	#ifdef DEBUG
		printf("Initializing inverted list in host memory... ");
	#endif

	readInvertedList(iTermNumberInVocabulary, dAverageDocLength, h_iDocIdList, h_iLenghtList,h_iFreqList,
			  	  	 h_dIdfList, h_dUBList, h_iDocNumberList, docTotalNumber);

	#ifdef DEBUG
		printf("OK!\n");
	#endif
//	for (int i = 0; i < *iTermNumber; ++i) {
//
//		*iPostingTotalNumber +=  (*h_iDocNumberList)[i];
//
//		if(*iMaxNumberInList < (*h_iDocNumberList)[i]){
//			*iMaxNumberInList = (*h_iDocNumberList)[i];
//		}
//	}
	#ifdef DEBUG
		printf("#Terms in Inverted List: %hu\n",*iTermNumberInVocabulary);
	#endif
//	printf("Max: %i\n",*iMaxNumberInList);
//	printf("Total: %i\n",*iPostingTotalNumber);

}

__host__ void initializeQueryBatch(int ***h_iQueryBatches, int *h_iQueryNumber, int **h_iTermNumberList){

	#ifdef DEBUG
		printf("Get query batch to host memory... ");
	#endif

	readQueryBatch(h_iQueryBatches, h_iQueryNumber, h_iTermNumberList);

	#ifdef DEBUG
		printf("OK!\n");
	#endif
}

__host__ void memoryAllocationOfInvertedIndexInGPU(unsigned long long iDocTotalNumber, int iTermNumber,
												   int **d_iDocIdList, int **d_iDocLenghtList,
												   unsigned short int **d_iFreqDocList,
												   float **d_dUBlist, float **d_dIdfList,
												   int **d_iDocNumberByTermList){

	#ifdef DEBUG
		printf("Allocating Pointer to Pointer and coping yours values in/to Device Memory... ");
	#endif

	unsigned long long int nbytes_short = iDocTotalNumber * sizeof(unsigned short int);
	unsigned long long int nbytes = iDocTotalNumber * sizeof(int);

	checkCudaErrors(cudaMalloc((void**) &(*d_iDocIdList), nbytes));
	checkCudaErrors(cudaMalloc((void**) &(*d_iDocLenghtList), nbytes));
	checkCudaErrors(cudaMalloc((void**) &(*d_iFreqDocList), nbytes_short));

	nbytes = iTermNumber * sizeof(float);
	checkCudaErrors(cudaMalloc((void **) &(*d_dUBlist), nbytes));
	checkCudaErrors(cudaMalloc((void **) &(*d_dIdfList), nbytes));

	nbytes = iTermNumber * sizeof(int);
	checkCudaErrors(cudaMalloc((void **) &(*d_iDocNumberByTermList), nbytes));

//	nbytes = iTopk * sizeof(int) * topkListNumber;
//	checkCudaErrors(cudaMalloc((void **) &(*d_iTopkDocList), nbytes));
//	checkCudaErrors(cudaMemset(*d_iTopkDocList, -1, nbytes));

//	nbytes = iTopk * sizeof(float) * topkListNumber;
//	checkCudaErrors(cudaMalloc((void **)&(*d_dTopkScoreList), nbytes));
//	checkCudaErrors(cudaMemset(*d_dTopkScoreList, 0.0, nbytes));
	#ifdef DEBUG
		printf("OK!\n");
	#endif
}


__host__ void memoryAllocationOfSingleQueryInGPU(int iDocTotalNumber, int iTermNumber,
												 int topkListNumber, int iTopk,
												 int **d_iDocIdList, int **d_iDocLenghtList,
												 unsigned short int **d_iFreqDocList,
												 float **d_dUBlist, float **d_dIdfList,
												 int **d_iDocNumberByTermList, int **d_iTopkDocList,
												 float **d_dTopkScoreList){
	#ifdef DEBUG
		printf("Allocating Pointer to Pointer and coping yours values in/to Device Memory... ");
	#endif

	int nbytes_short = iDocTotalNumber * sizeof(unsigned short int);
	long long nbytes = iDocTotalNumber * sizeof(int);

	checkCudaErrors(cudaMalloc((void**) &(*d_iDocIdList), nbytes));
	checkCudaErrors(cudaMalloc((void**) &(*d_iDocLenghtList), nbytes));
	checkCudaErrors(cudaMalloc((void**) &(*d_iFreqDocList), nbytes_short));

	nbytes = iTermNumber * sizeof(float);
	checkCudaErrors(cudaMalloc((void **) &(*d_dUBlist), nbytes));
	checkCudaErrors(cudaMalloc((void **) &(*d_dIdfList), nbytes));

	nbytes = iTermNumber * sizeof(int);
	checkCudaErrors(cudaMalloc((void **) &(*d_iDocNumberByTermList), nbytes));

	nbytes = iTopk * sizeof(int) * topkListNumber;
	checkCudaErrors(cudaMalloc((void **) &(*d_iTopkDocList), nbytes));
	checkCudaErrors(cudaMemset(*d_iTopkDocList, -1, nbytes));

	nbytes = iTopk * sizeof(float) * topkListNumber;
	checkCudaErrors(cudaMalloc((void **)&(*d_dTopkScoreList), nbytes));
	checkCudaErrors(cudaMemset(*d_dTopkScoreList, 0.0, nbytes));

	#ifdef DEBUG
		printf("OK!\n");
	#endif
}

__host__ void freeAllocationOfInvertedListInGPU(int *d_iDocIdList, int *d_iDocLenghtList,
											   unsigned short int *d_iFreqDocList,
											   float *d_dUBlist, float *d_dIdfList,
											   int *d_iDocNumberByTermList){
	#ifdef DEBUG
		printf("Free Memory in GPU... ");
	#endif
	checkCudaErrors(cudaFree(d_iDocIdList));
	checkCudaErrors(cudaFree(d_iFreqDocList));
	checkCudaErrors(cudaFree(d_iDocLenghtList));
	checkCudaErrors(cudaFree(d_dUBlist));
	checkCudaErrors(cudaFree(d_dIdfList));
	checkCudaErrors(cudaFree(d_iDocNumberByTermList));
	#ifdef DEBUG
		printf("OK!\n");
	#endif
}


__host__ void freeAllocationOfSingleQueryInGPU(int *d_iDocIdList, int *d_iDocLenghtList,
											   unsigned short int *d_iFreqDocList,
											   float *d_dUBlist, float *d_dIdfList,
											   int *d_iDocNumberByTermList, int *d_iTopkDocList,
											   float *d_dTopkScoreList){
	#ifdef DEBUG
	   printf("Free Memory in GPU... ");
	#endif
	checkCudaErrors(cudaFree(d_iDocIdList));
	checkCudaErrors(cudaFree(d_iFreqDocList));
	checkCudaErrors(cudaFree(d_iDocLenghtList));
	checkCudaErrors(cudaFree(d_dUBlist));
	checkCudaErrors(cudaFree(d_dIdfList));
	checkCudaErrors(cudaFree(d_iDocNumberByTermList));
	checkCudaErrors(cudaFree(d_iTopkDocList));
	checkCudaErrors(cudaFree(d_dTopkScoreList));
	#ifdef DEBUG
		printf("OK!\n");
	#endif
}

__host__ void memoryCopyOfSingleQuery_To_Device(unsigned long long iDocTotalNumber, int iTermNumber,
											    int* h_iDocNumberByTermList,
											    int *d_iDocIdList, int *d_iDocLenghtList,
											    unsigned short int *d_iFreqDocList,float *d_dUBlist, float *d_dIdfList,
											    int *d_iDocNumberByTermList, int **h_iDocIdList,
											    int **h_iDocLenghtList, unsigned short int **h_iFreqList,
												float *h_dUBlist, float *h_dIdfList){
	#ifdef DEBUG
		printf("Memory Copy of Host Memory To Device Memory... ");
	#endif

	unsigned long long int nbytes_short;// = iDocTotalNumber * sizeof(unsigned short int);
	unsigned long long int nbytes;// = iDocTotalNumber * sizeof(int);
	int position = 0;

	for (int i = 0; i < iTermNumber; ++i) {
		nbytes = h_iDocNumberByTermList[i] * sizeof(int);
		checkCudaErrors(cudaMemcpyAsync(&d_iDocIdList[position], h_iDocIdList[i], nbytes, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(&d_iDocLenghtList[position], h_iDocLenghtList[i], nbytes, cudaMemcpyHostToDevice));

		nbytes_short = h_iDocNumberByTermList[i] *sizeof(unsigned short int);
		checkCudaErrors(cudaMemcpyAsync(&d_iFreqDocList[position], h_iFreqList[i], nbytes_short, cudaMemcpyHostToDevice));

		position += h_iDocNumberByTermList[i];
	}

	nbytes = iTermNumber * sizeof(float);
	checkCudaErrors(cudaMemcpyAsync(d_dUBlist, h_dUBlist, nbytes, cudaMemcpyHostToDevice, 0));
	checkCudaErrors(cudaMemcpyAsync(d_dIdfList, h_dIdfList, nbytes, cudaMemcpyHostToDevice, 0));

	nbytes = iTermNumber * sizeof(int);
	checkCudaErrors(cudaMemcpyAsync(d_iDocNumberByTermList, h_iDocNumberByTermList, nbytes, cudaMemcpyHostToDevice, 0));
//	checkCudaErrors(cudaMemcpyToSymbol(iDocNumberByTermListConstant, h_iDocNumberByTermList, nbytes));
//	checkCudaErrors(cudaDeviceSynchronize());

	#ifdef DEBUG
		printf("OK!\n");
	#endif
}

__host__ void callToBatchKernels_byBlock(dim3 blocksByGrid, dim3 threadsByBlock, int iMergeNumberByBlock, int iTopk,
								float dAverageDocumentLength, float dInitialThreshold,
								short int* iTermNumberInQuery,
								float *d_dUBList, float *d_dIdfList, int *d_iDocIdList,
								unsigned short int *d_iFreqDocList, int *d_iDocLenghtList,
								int *d_iTopkDocList, float *d_dTopkScoreList,
								int *iQueryTerms,long long* d_ptrPostingPositions,
								int* d_ptrQueryPositions, int *d_iDocNumberByTermList){
	#ifdef DEBUG
		printf("Launching Kernels by Block: WAND... ");
	#endif
//    int nbytes;
    //--------------------Creation CUDA Event Handles------------------------
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	float gpu_time = 0.0f;
    //-------------------------------------------------------------------------

   checkCudaErrors(cudaEventRecord(start, 0));

//	matchWandParallel_VARIABLE_Batch_Block<<<blocksByGrid, threadsByBlock>>>(d_iDocIdList, d_iFreqDocList,
//	        				  	  	  	  	  	  	  	  	  	  	  	  d_dUBList, d_dIdfList, d_iDocLenghtList,
//	        				  	  	  	  	  	  	  	  	  	  	  	  iTermNumberInQuery, d_iTopkDocList, d_dTopkScoreList,
//	        				  	  	  	  	  	  	  	  	  	  	  	  dAverageDocumentLength,iBlockNumberRound,
//	        				  	  	  	  	  	  	  	  	  	  	  	  iGlobalNumberRound,iTopk,dInitialThreshold,
//	        				  	  	  	  	  	  	  	  	  	  	  	  iQueryTerms, d_ptrPostingPositions,
//	        				  	  	  	  	  	  	  	  	  	  	  	  d_ptrQueryPositions, d_iDocNumberByTermList);

   matchWandParallel_VARIABLE_Batch_Block_2<<<blocksByGrid, threadsByBlock>>>(d_iDocIdList, d_iFreqDocList,
																			   d_dUBList, d_dIdfList, d_iDocLenghtList,
																			   iTermNumberInQuery, d_iTopkDocList, d_dTopkScoreList,
																			   dAverageDocumentLength,
																			   iTopk,dInitialThreshold,
																			   iQueryTerms, d_ptrPostingPositions,
																			   d_ptrQueryPositions, d_iDocNumberByTermList);



	#ifdef DEBUG
		printf("Ok \n ");
	#endif

//
//    printf("OK!\n");

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));


	#ifdef DEBUG
		printf("OK!\n");
	#endif

	checkCudaErrors(cudaDeviceSynchronize());
	printf("Coping results from GPU to CPU... ");
	int * h_iTopkDocList = (int*) malloc(iTopk * sizeof(int)*500);
	float * h_dTopkScoreList = (float*) malloc(iTopk * sizeof(float)*500);

	int nbytes = iTopk * sizeof(int)*500;
	checkCudaErrors(cudaMemcpy(h_iTopkDocList,d_iTopkDocList, nbytes, cudaMemcpyDeviceToHost));

	nbytes = iTopk * sizeof(float)*500;
	checkCudaErrors(cudaMemcpy(h_dTopkScoreList,d_dTopkScoreList, nbytes, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());

	for (int j = 0; j < 500; ++j) {
		printf("\n--- %d Query -----\n",j);
		for (int i = 0; i < iTopk; ++i) {
			printf("--- %d %.2f----",h_iTopkDocList[j*iTopk + i],h_dTopkScoreList[j*iTopk + i]);
		}
	}

    printf("Time spent executing by the GPU: %.2f ms\n", gpu_time);
    printf("Batch %d, %d, %d, %.4f \n",TOP_K,SHAREDTHESHOLD,threadsByBlock.x, gpu_time);
}

__host__ void __inline__ callToBatchKernels(dim3 blocksByGrid, dim3 threadsByBlock, int iMergeNumberByBlock, int iTopk,
								float dAverageDocumentLength, float dInitialThreshold,
								int iBlockNumberRound, int iGlobalNumberRound, int iTermNumberInQuery,
								float *d_dUBList, float *d_dIdfList,
								int *d_iDocIdList, unsigned short int *d_iFreqDocList, int *d_iDocLenghtList,
								int *d_iTopkDocList, float *d_dTopkScoreList,
								int idQuery,
								cudaStream_t stream, int *iQueryTerms, long long* d_ptrPostingPositions,
								int* d_ptrQueryPositions, int *d_iDocNumberByTermList){

	#ifdef DEBUG
		printf("Launching Kernels: WAND... ");
	    //--------------------Creation CUDA Event Handles------------------------
		cudaEvent_t start, stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		float gpu_time = 0.0f;
		    checkCudaErrors(cudaEventRecord(start, stream));
	    //-------------------------------------------------------------------------

	#endif




	matchWandParallel_BATCH_2<<<blocksByGrid, threadsByBlock, 0, stream>>>(d_iDocIdList, d_iFreqDocList,
	        				  	  	  	  	  	  	  	  	  	  	  	  d_dUBList, d_dIdfList, d_iDocLenghtList,
	        				  	  	  	  	  	  	  	  	  	  	  	  iTermNumberInQuery, d_iTopkDocList, d_dTopkScoreList,
	        				  	  	  	  	  	  	  	  	  	  	  	  dAverageDocumentLength,iBlockNumberRound,
	        				  	  	  	  	  	  	  	  	  	  	  	  iGlobalNumberRound,iTopk,dInitialThreshold,
	        				  	  	  	  	  	  	  	  	  	  	  	  iQueryTerms, d_ptrPostingPositions,
	        				  	  	  	  	  	  	  	  	  	  	  	  d_ptrQueryPositions, idQuery, d_iDocNumberByTermList);


	int iTotalElementos = blocksByGrid.x * iTopk;
	int iProcessedListNumberbyBlock = iMergeNumberByBlock + 1;
	int iSkipTopKBetweenThreadBlocks = iProcessedListNumberbyBlock;
	int iSkipTopKBetweenMerges = 1;
	int exponent = 0;
	int exponent_b = 1;

	blocksByGrid.x = ceilf( blocksByGrid.x/(iProcessedListNumberbyBlock) );
//    checkCudaErrors(cudaDeviceSynchronize());
//    evaluateAccuracyInGPU_Test(iTopk, copy_block , d_iTopkDocList,round);


	#ifdef DEBUG
		checkCudaErrors(cudaDeviceSynchronize());
		printf("Ok\n");
		printf("Launching Kernels: Merge... ");
	#endif

    while(blocksByGrid.x >= 1){
//    	checkCudaErrors(cudaDeviceSynchronize());
        mergeTopkLists_v3<<<blocksByGrid, iTopk, 0,stream>>>(d_dTopkScoreList, d_iTopkDocList,
      													 iTopk, iMergeNumberByBlock,
      													 iSkipTopKBetweenMerges,
      													 iSkipTopKBetweenThreadBlocks,iTotalElementos);

    	blocksByGrid.x = ceilf (blocksByGrid.x/(iProcessedListNumberbyBlock) );
    	exponent ++;
    	exponent_b ++;
    	iSkipTopKBetweenMerges = pow(iProcessedListNumberbyBlock, exponent);
    	iSkipTopKBetweenThreadBlocks = pow(iProcessedListNumberbyBlock, exponent_b);
    }


	#ifdef DEBUG
    	checkCudaErrors(cudaDeviceSynchronize());
		printf("Ok \n ");
	#endif

//	printf("Coping results from GPU to CPU... ");
//
//	int nbytes = iTopk * sizeof(int);
//	int *h_iTopkDocList = (int*) malloc(sizeof(int) * iTopk);
//	checkCudaErrors(cudaMemcpy(h_iTopkDocList, d_iTopkDocList, nbytes, cudaMemcpyDeviceToHost));
//
//	float *h_dTopkScoreList = (float*) malloc(sizeof(float) * iTopk);
//	nbytes = iTopk * sizeof(float);
//    checkCudaErrors(cudaMemcpy(h_dTopkScoreList, d_dTopkScoreList, nbytes, cudaMemcpyDeviceToHost));
//
//    checkCudaErrors(cudaDeviceSynchronize());
//
////    for (int j = 0; j < 500; ++j) {
//    	printf("\n--- %d Query -----\n",idQuery);
//		for (int i = 0; i < iTopk; ++i) {
//			printf("--- %d %.2f----",h_iTopkDocList[i],h_dTopkScoreList[i]);
//		}
////	}
//
//    free(h_iTopkDocList);
//    free(h_dTopkScoreList);

	#ifdef DEBUG
	    checkCudaErrors(cudaEventRecord(stop));
	    checkCudaErrors(cudaEventSynchronize(stop));
	    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
	    checkCudaErrors(cudaEventDestroy(start));
	    checkCudaErrors(cudaEventDestroy(stop));

		printf("Time spent executing by the GPU: %.2f ms\n", gpu_time);
	#endif
}

__host__ void callToKernels_Teste(dim3 blocksByGrid, dim3 threadsByBlock, int iMergeNumberByBlock, int iTopk,
							float dAverageDocumentLength, float dInitialThreshold,
							int iBlockNumberRound, int iGlobalNumberRound, int iTermNumber,
							float *d_dUBList, float *d_dIdfList,
							int *d_iDocIdList, unsigned short int *d_iFreqDocList, int *d_iDocLenghtList,
							int *d_iTopkDocList, float *d_dTopkScoreList,
							int *h_iTopkDocList, float *h_dTopkScoreList, int *d_iDocNumberByTermList,
							int* d_iInitialPositionPostingList, int* d_extraPositions, int* d_docMaxList,
							int docIdNumberByBlock){

		#ifdef DEBUG
			printf("Launching Kernels: WAND Teste... ");
		#endif
		int nbytes;
		//--------------------Creation CUDA Event Handles------------------------
		cudaEvent_t start, stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		float gpu_time = 0.0f;
		//-------------------------------------------------------------------------

		checkCudaErrors(cudaEventRecord(start, 0));

		preProcessingWand<<<blocksByGrid, 1024, 0, 0>>>(d_iDocIdList, iTermNumber,
												      d_iDocNumberByTermList, d_iInitialPositionPostingList,
												      docIdNumberByBlock, d_extraPositions, d_docMaxList);

//		printf("Launching Kernels: WAND Teste... ");
//	    checkCudaErrors(cudaDeviceSynchronize());
		matchWandParallel_VARIABLE_3_Teste<<<blocksByGrid, threadsByBlock, 0, 0>>>(d_iDocIdList, d_iFreqDocList,
																			  d_dUBList, d_dIdfList,
																			  d_iDocLenghtList,
																			  iTermNumber, d_iTopkDocList, d_dTopkScoreList,
																			  dAverageDocumentLength, //iBlockNumberRound,
																			  iGlobalNumberRound,iTopk,dInitialThreshold,d_iDocNumberByTermList,
																			  d_extraPositions, d_docMaxList);


		int iTotalElementos = blocksByGrid.x * iTopk;
		int iProcessedListNumberbyBlock = iMergeNumberByBlock + 1;
		int iSkipTopKBetweenThreadBlocks = iProcessedListNumberbyBlock;
		int iSkipTopKBetweenMerges = 1;
		int exponent = 0;
		int exponent_b = 1;

		blocksByGrid.x = ceilf( blocksByGrid.x/(iProcessedListNumberbyBlock) );
		//    checkCudaErrors(cudaDeviceSynchronize());
		//    evaluateAccuracyInGPU_Test(iTopk, copy_block , d_iTopkDocList,round);

		#ifdef DEBUG
			checkCudaErrors(cudaDeviceSynchronize());
			printf("Ok\n");
			printf("Launching Kernels: Merge... ");
		#endif

		while(blocksByGrid.x >= 1){
		//    	checkCudaErrors(cudaDeviceSynchronize());
			mergeTopkLists_v3<<<blocksByGrid, iTopk, 0,0>>>(d_dTopkScoreList, d_iTopkDocList,
															 iTopk, iMergeNumberByBlock,
															 iSkipTopKBetweenMerges,
															 iSkipTopKBetweenThreadBlocks,iTotalElementos);

			blocksByGrid.x = ceilf (blocksByGrid.x/(iProcessedListNumberbyBlock) );
			exponent ++;
			exponent_b ++;
			iSkipTopKBetweenMerges = pow(iProcessedListNumberbyBlock, exponent);
			iSkipTopKBetweenThreadBlocks = pow(iProcessedListNumberbyBlock, exponent_b);
		}

		#ifdef DEBUG
			checkCudaErrors(cudaDeviceSynchronize());
		#endif

		checkCudaErrors(cudaEventRecord(stop, 0));
		checkCudaErrors(cudaEventSynchronize(stop));
		checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
		checkCudaErrors(cudaDeviceSynchronize());


		#ifdef DEBUG
			printf("Ok \n ");
			printf("Coping results from GPU to CPU... ");
		#endif

		nbytes = iTopk * sizeof(int);
		checkCudaErrors(cudaMemcpy(h_iTopkDocList, d_iTopkDocList, nbytes, cudaMemcpyDeviceToHost));

		nbytes = iTopk * sizeof(float);
		checkCudaErrors(cudaMemcpy(h_dTopkScoreList, d_dTopkScoreList, nbytes, cudaMemcpyDeviceToHost));
		//    checkCudaErrors(cudaDeviceSynchronize());

		#ifdef DEBUG
			printf("OK!\n");
		#endif

		checkCudaErrors(cudaEventDestroy(start));
		checkCudaErrors(cudaEventDestroy(stop));

		printf("Result %d, %d, %d, %d, %d, %.4f \n",TOP_K,SHAREDTHESHOLD,DOC_QUANTITY_IN_MEMORY,threadsByBlock.x, iGlobalNumberRound, gpu_time);

		//	#ifdef DEBUG
		printf("Time spent executing by the GPU: %.4f ms\n", gpu_time);
		//	#endif


}


__host__ void callToKernels(dim3 blocksByGrid, dim3 threadsByBlock, int iMergeNumberByBlock, int iTopk,
							float dAverageDocumentLength, float dInitialThreshold,
							int iBlockNumberRound, int iGlobalNumberRound, int iTermNumber,
							float *d_dUBList, float *d_dIdfList,
							int *d_iDocIdList, unsigned short int *d_iFreqDocList, int *d_iDocLenghtList,
							int *d_iTopkDocList, float *d_dTopkScoreList,
							int *h_iTopkDocList, float *h_dTopkScoreList, int *d_iDocNumberByTermList){
	#ifdef DEBUG
		printf("Launching Kernels: WAND... ");
	#endif
    int nbytes;
    //--------------------Creation CUDA Event Handles------------------------
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	float gpu_time = 0.0f;
    //-------------------------------------------------------------------------

	int iTotalElementos = blocksByGrid.x * iTopk;
	int iProcessedListNumberbyBlock = iMergeNumberByBlock;// + 1;
	iMergeNumberByBlock = (1<<iMergeNumberByBlock)-1;
//	iProcessedListNumberbyBlock = iProcessedListNumberbyBlock / 2;
	int iSkipTopKBetweenThreadBlocks = 1 << iProcessedListNumberbyBlock;
	int iSkipTopKBetweenMerges = 1;
//	int exponent = 0;
//	int exponent_b = 1;


    checkCudaErrors(cudaEventRecord(start, 0));

    matchWandParallel_VARIABLE_4_2<<<blocksByGrid, threadsByBlock, 0, 0>>>(d_iDocIdList, d_iFreqDocList,
	        				  	  	  	  	  	  	  	  	  	  	  	  d_dUBList, d_dIdfList,
	        				  	  	  	  	  	  	  	  	  	  	  	  d_iDocLenghtList,
	        				  	  	  	  	  	  	  	  	  	  	  	  iTermNumber, d_iTopkDocList, d_dTopkScoreList,
	        				  	  	  	  	  	  	  	  	  	  	  	  dAverageDocumentLength, //iBlockNumberRound,
	        				  	  	  	  	  	  	  	  	  	  	  	  iGlobalNumberRound,iTopk,dInitialThreshold,d_iDocNumberByTermList);


//	iTotalElementos = blocksByGrid.x * iTopk;
//	iProcessedListNumberbyBlock = iMergeNumberByBlock + 1;
//	iSkipTopKBetweenThreadBlocks = iProcessedListNumberbyBlock;
//	iSkipTopKBetweenMerges = 1;
//	exponent = 0;
//	exponent_b = 1;

    blocksByGrid.x = (((blocksByGrid.x & 1) == 1) && (blocksByGrid.x != 1)) ? blocksByGrid.x+1 :  blocksByGrid.x;
	blocksByGrid.x = blocksByGrid.x >> iProcessedListNumberbyBlock; //ceilf( blocksByGrid.x/(iProcessedListNumberbyBlock) );

	#ifdef DEBUG
		checkCudaErrors(cudaDeviceSynchronize());
		printf("Ok\n");
		printf("Launching Kernels: Merge... ");
	#endif

    while(blocksByGrid.x >= 1){
//    	checkCudaErrors(cudaDeviceSynchronize());
//    	printf("========>Blocks %d iSkipMerges %d iSkipeBlocks %d \n", blocksByGrid.x,iSkipTopKBetweenMerges,iSkipTopKBetweenThreadBlocks);
        mergeTopkLists_v3<<<blocksByGrid, iTopk, 0,0>>>(d_dTopkScoreList, d_iTopkDocList,
      													 iTopk, iMergeNumberByBlock,
      													 iSkipTopKBetweenMerges,
      													 iSkipTopKBetweenThreadBlocks,iTotalElementos);
//        if(blocksByGrid.x == 1) blocksByGrid.x = 0;
        blocksByGrid.x = (((blocksByGrid.x & 1) == 1) && (blocksByGrid.x != 1)) ? blocksByGrid.x+1 :  blocksByGrid.x;
        blocksByGrid.x =  blocksByGrid.x >> iProcessedListNumberbyBlock;//ceilf (blocksByGrid.x/(iProcessedListNumberbyBlock) );
//    	exponent ++;
//    	exponent_b ++;
    	iSkipTopKBetweenMerges = iSkipTopKBetweenMerges << 1;//pow(iProcessedListNumberbyBlock, exponent);
    	iSkipTopKBetweenThreadBlocks = iSkipTopKBetweenThreadBlocks << 1;//pow(iProcessedListNumberbyBlock, exponent_b);
//    	checkCudaErrors(cudaDeviceSynchronize());
    }

	#ifdef DEBUG
    	checkCudaErrors(cudaDeviceSynchronize());
	#endif

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));
    checkCudaErrors(cudaDeviceSynchronize());
	printf("Time spent executing by the GPU: %.4f ms\n", gpu_time);

    #ifdef DEBUG
		printf("Ok \n ");
		printf("Coping results from GPU to CPU... ");
	#endif

	nbytes = iTopk * sizeof(int);
	checkCudaErrors(cudaMemcpy(h_iTopkDocList, d_iTopkDocList, nbytes, cudaMemcpyDeviceToHost));

	nbytes = iTopk * sizeof(float);
    checkCudaErrors(cudaMemcpy(h_dTopkScoreList, d_dTopkScoreList, nbytes, cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaDeviceSynchronize());

    #ifdef DEBUG
		printf("OK!\n");
	#endif

	for (int i = 0; i < iTopk; ++i) {
		printf("--- %d ",h_iTopkDocList[i]);
	}

	for (int i = 0; i < iTopk; ++i) {
		printf("--- %.2f ",h_dTopkScoreList[i]);
	}
	printf("\n");
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    printf("Result %d, %d, %d, %d, %d, %.4f \n",TOP_K,SHAREDTHESHOLD,DOC_QUANTITY_IN_MEMORY,threadsByBlock.x, iGlobalNumberRound, gpu_time);

}

__host__ void querySingleProcessingHost_Teste(int iTopk, int iBlockSize, int iBlockNumberRound,
											  int iGlobalNumberRound, int iMergeNumberByBlock,
											  int iQueryType, int iExperimentNumber){

	cudaSetDevice(0);
	//	printf("TOPK: %d\n", TOP_K);
	//	printf("DOC_QUANTITY_IN_MEMORY: %d\n", DOC_QUANTITY_IN_MEMORY);

//const int* iDocIdList,
//const short int iTermNumber,
//const int* d_iDocNumberByTermList,
//const int* iInitialPositionPostingList,
//const int docIdNumberByBlock,
//int* extraPositions, int* docMaxList

		int iPostingTotalNumber = 0;
		int iMaxNumberInList = 0;
		int iTermNumber;
		int docTotalNumber = 0;

		//----------Host Variables-------------
		int** h_iDocIdList;
		int** h_iDocLenghtList;
		unsigned short int** h_iFreqList;

		float* h_dUBList;
		float* h_dIdfList;

		int* h_iDocNumberByTermList;
		int* h_iInitialPositionPostingList;

		int* h_iTopkDocList = (int*) malloc(iTopk * sizeof(int));
		float* h_dTopkScoreList = (float*) malloc(iTopk * sizeof(float));

		float dAverageDocumentLength;
		//-------------------------------------

		//----------Device Variables-------------
		float dInitialThreshold = 0;
		float* d_dUBList;
		float* d_dIdfList;

		int* d_iDocIdList;
		int* d_iDocLenghtList;
		unsigned short int* d_iFreqDocList;

		int* d_iDocNumberByTermList;

		int* d_iTopkDocList;
		float* d_dTopkScoreList;

		int* d_iInitialPositionPostingList;
		int* d_extraPositions;
		int* d_docMaxList;
		int docIdNumberByBlock;
		//-------------------------------------
		#ifdef DEBUG
			printf("Initializing variables in host memory...\n");
		#endif
		initializeSingleExperimentQuery(&iTermNumber, &dAverageDocumentLength, &iTopk, &h_iDocIdList, &h_iDocLenghtList,
								 &h_iFreqList, &h_dIdfList, &h_dUBList, &h_iDocNumberByTermList,
								 &iPostingTotalNumber, &iMaxNumberInList,iExperimentNumber);
		//-------------------------------------

		//------------Kernel Launch Configuration---------------------------------
		printf("Kernel Launch Configuration...\n");
		int iBlockNumber= (int) ceil((float)iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * iGlobalNumberRound));
	//	int iBlockNumber= (int) ceil((float)iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * iBlockNumberRound * iGlobalNumberRound));
		dim3 threadsByBlock = dim3(iBlockSize, 1);
		dim3 blocksByGrid   = dim3(iBlockNumber, 1);
		printf("#threads by block: %i, #blocks: %i ...\n",threadsByBlock.x, blocksByGrid.x) ;
		//-------------------------------------------------------------------------

		h_iInitialPositionPostingList = (int*) malloc(sizeof(int*) * iTermNumber);
		int positionInList = 0;
		for (int idTerm = 0; idTerm < iTermNumber; ++idTerm) {
			h_iInitialPositionPostingList[idTerm] = positionInList;
			positionInList += h_iDocNumberByTermList[idTerm];
		}
		int nb = sizeof(int) * iTermNumber;
		checkCudaErrors(cudaMalloc((void**) &d_iInitialPositionPostingList, nb));
		checkCudaErrors(cudaMemcpyAsync(d_iInitialPositionPostingList, h_iInitialPositionPostingList, nb, cudaMemcpyHostToDevice, 0));

		nb = sizeof(int) * iBlockNumber * iTermNumber;
		checkCudaErrors(cudaMalloc((void**) &d_extraPositions, nb));
		nb = sizeof(int) * iBlockNumber;
		checkCudaErrors(cudaMalloc((void**) &d_docMaxList, nb));

		docIdNumberByBlock = DOC_QUANTITY_IN_MEMORY*iGlobalNumberRound;
		//------------------------------------

	//	float** h_dUBLocal;
	//	h_dUBLocal = (float**) malloc( sizeof(float*)*iTermNumber);
	//
	//	float score = 0.0;
	//	float maxScore = 0.0;
	//	float k_1 = 1.2;
	//	float b = 0.75;
	//	for (int term = 0; term < iTermNumber; ++term) {
	//		maxScore = 0.0;
	//		h_dUBLocal[term] = (float*) malloc(iBlockNumber * sizeof(float));
	//
	//		for (int partition = 0; partition < iBlockNumber; ++partition) {
	//
	//			int position= partition * DOC_QUANTITY_IN_MEMORY * iGlobalNumberRound;
	//			int count = 0;
	//			while((count < DOC_QUANTITY_IN_MEMORY * iGlobalNumberRound) && (position < h_iDocNumberByTermList[term])) {
	//				score = (k_1 * h_iFreqList[term][position]) / ( h_iFreqList[term][position] + (k_1 * ((1 - b) + (b * h_iDocLenghtList[term][position]) / dAverageDocumentLength)));
	//				if(score > maxScore){
	//					maxScore = score;
	//				}
	//				count++;
	//				position++;
	//			}
	//
	//			h_dUBLocal[term][partition] = 1.1 * maxScore * h_dIdfList[term];
	//		}
	//	}
	//
	//	float* d_dUBLocal;
	//	int nbytes = sizeof(float) * iTermNumber * iBlockNumber;
	//	checkCudaErrors(cudaMalloc((void**) &(d_dUBLocal), nbytes));
	//
	//	int position = 0;
	//	for (int iTerm = 0; iTerm < iTermNumber; ++iTerm) {
	//		nbytes = iBlockNumber * sizeof(float);
	//		checkCudaErrors(cudaMemcpyAsync(&d_dUBLocal[position], h_dUBLocal[iTerm], nbytes, cudaMemcpyHostToDevice));
	//		position += iBlockNumber;
	//	}


		//-----------------------------------



		//-----------Initializing the initial value of Threshold if it is necessery ---------------------
		if(iQueryType){
			setThresholdForANDQuery(&dInitialThreshold, h_dUBList, iTermNumber);
		}
		//-----------------------------------------------------------------------------------------------

		for (int i = 0; i < iTermNumber; ++i) {
			docTotalNumber += h_iDocNumberByTermList[i];
		}

		//------------Allocating and Coping Pointer to Pointer in Device Memory--------------------------------------
		memoryAllocationOfSingleQueryInGPU(docTotalNumber, iTermNumber, iBlockNumber, iTopk,
										   &d_iDocIdList, &d_iDocLenghtList, &d_iFreqDocList,
										   &d_dUBList, &d_dIdfList, &d_iDocNumberByTermList, &d_iTopkDocList,
										   &d_dTopkScoreList);

		memoryCopyOfSingleQuery_To_Device(docTotalNumber, iTermNumber,h_iDocNumberByTermList,
			    						  d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
			    						  d_dUBList, d_dIdfList, d_iDocNumberByTermList, h_iDocIdList,
			    						  h_iDocLenghtList, h_iFreqList, h_dUBList, h_dIdfList);
		//----------------------------------------------------------------------------------------------------------

		//--------------------------Call to Kernels----------------------------
		callToKernels_Teste(blocksByGrid, threadsByBlock, iMergeNumberByBlock, iTopk,
					  dAverageDocumentLength, dInitialThreshold, iBlockNumberRound,
					  iGlobalNumberRound, iTermNumber,
					  d_dUBList, d_dIdfList,d_iDocIdList, d_iFreqDocList, d_iDocLenghtList,
					  d_iTopkDocList, d_dTopkScoreList, h_iTopkDocList, h_dTopkScoreList,d_iDocNumberByTermList,
					  d_iInitialPositionPostingList, d_extraPositions, d_docMaxList,
					  docIdNumberByBlock);

		//----------------------------------------------------------------------------------------------------------

		//-----------------------Free GPU Memory------------------------------

		freeAllocationOfSingleQueryInGPU(d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
										 d_dUBList, d_dIdfList, d_iDocNumberByTermList, d_iTopkDocList,
										 d_dTopkScoreList);
		//----------------------------------------------------------------------------------------------------------

		checkCudaErrors(cudaFree(d_extraPositions));
		checkCudaErrors(cudaFree(d_docMaxList));
		//------------------Release resources in CPU-------------------------------
		printf("Releasing resources in CPU... ");

		for (int i = 0; i < iTermNumber; ++i) {
			free(h_iDocIdList[i]);
			free(h_iDocLenghtList[i]);
			free(h_iFreqList[i]);
	//		free(h_dUBLocal[i]);
		}
		free(h_iInitialPositionPostingList);
		free(h_iDocIdList);
		free(h_iDocLenghtList);
		free(h_iFreqList);
		free(h_dIdfList);
		free(h_dUBList);
		free(h_iDocNumberByTermList);
		free(h_iTopkDocList);
		free(h_dTopkScoreList);
		printf("OK!\n");
		//-------------------------------------------------------------------------
		printf("Finish!\n");


}

__host__ void querySingleProcessingHost(int iTopk, int iBlockSize, int iBlockNumberRound,
										int iGlobalNumberRound, int iMergeNumberByBlock,
										int iQueryType, int iExperimentNumber){
//	printf("TOPK: %d\n", TOP_K);
//	printf("DOC_QUANTITY_IN_MEMORY: %d\n", DOC_QUANTITY_IN_MEMORY);

	cudaSetDevice(0);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	printf("Device Name: %s\n",prop.name);

	int iPostingTotalNumber = 0;
	int iMaxNumberInList = 0;
	int iTermNumber;
	int docTotalNumber = 0;

	//----------Host Variables-------------
	int** h_iDocIdList;
	int** h_iDocLenghtList;
	unsigned short int** h_iFreqList;

	float* h_dUBList;
	float* h_dIdfList;

	int* h_iDocNumberByTermList;

	int* h_iTopkDocList = (int*) malloc(iTopk * sizeof(int));
	float* h_dTopkScoreList = (float*) malloc(iTopk * sizeof(float));

	float dAverageDocumentLength;
	//-------------------------------------

	//----------Device Variables-------------
	float dInitialThreshold = 0;
	float* d_dUBList;
	float* d_dIdfList;

	int* d_iDocIdList;
	int* d_iDocLenghtList;
	unsigned short int* d_iFreqDocList;

	int* d_iDocNumberByTermList;
	int* d_iTopkDocList;
	float* d_dTopkScoreList;
	//-------------------------------------
	#ifdef DEBUG
		printf("Initializing variables in host memory...\n");
	#endif
	initializeSingleExperimentQuery(&iTermNumber, &dAverageDocumentLength, &iTopk, &h_iDocIdList, &h_iDocLenghtList,
							 &h_iFreqList, &h_dIdfList, &h_dUBList, &h_iDocNumberByTermList,
							 &iPostingTotalNumber, &iMaxNumberInList,iExperimentNumber);
	//-------------------------------------

	//------------Kernel Launch Configuration---------------------------------
	printf("Kernel Launch Configuration...\n");
	int iBlockNumber= (int) ceil((float)iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * iGlobalNumberRound));
//	int iBlockNumber= (int) ceil((float)iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * iBlockNumberRound * iGlobalNumberRound));
	dim3 threadsByBlock = dim3(iBlockSize, 1);
	dim3 blocksByGrid   = dim3(iBlockNumber, 1);
	printf("#threads by block: %i, #blocks: %i ...\n",threadsByBlock.x, blocksByGrid.x) ;
	//-------------------------------------------------------------------------

	//------------------------------------

//	float** h_dUBLocal;
//	h_dUBLocal = (float**) malloc( sizeof(float*)*iTermNumber);
//
//	float score = 0.0;
//	float maxScore = 0.0;
//	float k_1 = 1.2;
//	float b = 0.75;
//	for (int term = 0; term < iTermNumber; ++term) {
//		maxScore = 0.0;
//		h_dUBLocal[term] = (float*) malloc(iBlockNumber * sizeof(float));
//
//		for (int partition = 0; partition < iBlockNumber; ++partition) {
//
//			int position= partition * DOC_QUANTITY_IN_MEMORY * iGlobalNumberRound;
//			int count = 0;
//			while((count < DOC_QUANTITY_IN_MEMORY * iGlobalNumberRound) && (position < h_iDocNumberByTermList[term])) {
//				score = (k_1 * h_iFreqList[term][position]) / ( h_iFreqList[term][position] + (k_1 * ((1 - b) + (b * h_iDocLenghtList[term][position]) / dAverageDocumentLength)));
//				if(score > maxScore){
//					maxScore = score;
//				}
//				count++;
//				position++;
//			}
//
//			h_dUBLocal[term][partition] = 1.1 * maxScore * h_dIdfList[term];
//		}
//	}
//
//	float* d_dUBLocal;
//	int nbytes = sizeof(float) * iTermNumber * iBlockNumber;
//	checkCudaErrors(cudaMalloc((void**) &(d_dUBLocal), nbytes));
//
//	int position = 0;
//	for (int iTerm = 0; iTerm < iTermNumber; ++iTerm) {
//		nbytes = iBlockNumber * sizeof(float);
//		checkCudaErrors(cudaMemcpyAsync(&d_dUBLocal[position], h_dUBLocal[iTerm], nbytes, cudaMemcpyHostToDevice));
//		position += iBlockNumber;
//	}
//

	//-----------------------------------



	//-----------Initializing the initial value of Threshold if it is necessery ---------------------
	if(iQueryType){
		setThresholdForANDQuery(&dInitialThreshold, h_dUBList, iTermNumber);
	}
	//-----------------------------------------------------------------------------------------------

	for (int i = 0; i < iTermNumber; ++i) {
		docTotalNumber += h_iDocNumberByTermList[i];
	}

	//------------Allocating and Coping Pointer to Pointer in Device Memory--------------------------------------
	memoryAllocationOfSingleQueryInGPU(docTotalNumber, iTermNumber, iBlockNumber, iTopk,
									   &d_iDocIdList, &d_iDocLenghtList, &d_iFreqDocList,
									   &d_dUBList, &d_dIdfList, &d_iDocNumberByTermList, &d_iTopkDocList,
									   &d_dTopkScoreList);

	memoryCopyOfSingleQuery_To_Device(docTotalNumber, iTermNumber,h_iDocNumberByTermList,
		    						  d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
		    						  d_dUBList, d_dIdfList, d_iDocNumberByTermList, h_iDocIdList,
		    						  h_iDocLenghtList, h_iFreqList, h_dUBList, h_dIdfList);
	//----------------------------------------------------------------------------------------------------------

	//--------------------------Call to Kernels----------------------------
	callToKernels(blocksByGrid, threadsByBlock, iMergeNumberByBlock, iTopk,
				  dAverageDocumentLength, dInitialThreshold, iBlockNumberRound,
				  iGlobalNumberRound, iTermNumber,
				  d_dUBList, d_dIdfList,d_iDocIdList, d_iFreqDocList, d_iDocLenghtList,
				  d_iTopkDocList, d_dTopkScoreList, h_iTopkDocList, h_dTopkScoreList,d_iDocNumberByTermList);

	//----------------------------------------------------------------------------------------------------------

	//-----------------------Free GPU Memory------------------------------

	freeAllocationOfSingleQueryInGPU(d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
									 d_dUBList, d_dIdfList, d_iDocNumberByTermList, d_iTopkDocList,
									 d_dTopkScoreList);
	//----------------------------------------------------------------------------------------------------------

//	checkCudaErrors(cudaFree(d_dUBLocal));
	//------------------Release resources in CPU-------------------------------
	printf("Releasing resources in CPU... ");

	for (int i = 0; i < iTermNumber; ++i) {
		free(h_iDocIdList[i]);
		free(h_iDocLenghtList[i]);
		free(h_iFreqList[i]);
//		free(h_dUBLocal[i]);
	}
//	free(h_dUBLocal);
	free(h_iDocIdList);
	free(h_iDocLenghtList);
	free(h_iFreqList);
	free(h_dIdfList);
	free(h_dUBList);
	free(h_iDocNumberByTermList);
	free(h_iTopkDocList);
	free(h_dTopkScoreList);
	printf("OK!\n");
	//-------------------------------------------------------------------------
	printf("Finish!\n");
}

__host__ void queryBatchProcessingHost_Mix(int iTopk, int iBlockSize, int iBlockNumberRound,
										int iGlobalNumberRound, int iMergeNumberByBlock,
									int iQueryType){



//	int iPostingTotalNumber = 0;
//	int iMaxNumberInList = 0;
	int iTermNumberInVocabulary;
	unsigned long long docTotalNumber = 0;

	//----------Host Variables-------------
	int **h_iDocIdList, **h_iDocLenghtList;
	unsigned short int** h_iFreqList;

	float *h_dUBList, *h_dIdfList;

	int* h_iDocNumberByTermList;

	int* h_iTopkDocList = (int*) malloc(iTopk * sizeof(int));
	float* h_dTopkScoreList = (float*) malloc(iTopk * sizeof(float));

	float dAverageDocumentLength;

	int *h_iTermNumberByQuery; //Número de termo por query do batch de query
	int **h_iQueryBatches; //Batch de Queries
	int iQueryTotalNumberByBatch = 500; //Número de query por batch
	//-------------------------------------

	//---------Mapped Variables in Host and Device------------------
	int *m_iTermNumberByQuery; //Número de termo por query do batch de query - Os indices representam as queries
	int *m_iQueryBatches; //Batch de Queries - Um conjunto de termos - Os termos pertence a um conjunto de queries
	int *m_ptrQueryPositions; //A posição das queries na lista de batch de queries; essa contagem é a realizada pela contagem dos termos

	long long* m_ptrInitPostingList; //Considera que há uma única lista no acelerador  - Posição inicial das listas de postings dos termos

	int** m_iTopkDocList; // Topk
	float** m_dTopkScoreList; // Topk
	//---------------------------------------------------------------

	//----------Device Variables------------------------------------


	float dInitialThreshold = 0, *d_dUBList, *d_dIdfList;

	int *d_iDocIdList, *d_iDocLenghtList;
	unsigned short int* d_iFreqDocList;

	int* d_iDocNumberByTermList;
	int* d_iTopkDocList;
	float* d_dTopkScoreList;

	int *d_iTermNumberByQuery, *d_iQueryBatches, *d_ptrQueryPositions;
	long long *d_ptrInitPostingList;

	//-------------------------------------

	initializeInvertedIndex(&iTermNumberInVocabulary, &dAverageDocumentLength, &h_iDocIdList,
							&h_iDocLenghtList, &h_iFreqList, &h_dIdfList, &h_dUBList,
							&h_iDocNumberByTermList, &docTotalNumber);

	//-------------------GPU Pre-Configuration---------------------
	cudaSetDevice(0);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	//-------------------------------------------------------------

	//------------Allocating and Coping Pointer to Pointer in Device Memory--------------------------------------
	memoryAllocationOfInvertedIndexInGPU(docTotalNumber, iTermNumberInVocabulary,&d_iDocIdList,
										 &d_iDocLenghtList, &d_iFreqDocList,&d_dUBList,
										 &d_dIdfList, &d_iDocNumberByTermList);

	memoryCopyOfSingleQuery_To_Device(docTotalNumber, iTermNumberInVocabulary,h_iDocNumberByTermList,
									  d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
									  d_dUBList, d_dIdfList, d_iDocNumberByTermList, h_iDocIdList,
									  h_iDocLenghtList, h_iFreqList, h_dUBList, h_dIdfList);
	//----------------------------------------------------------------------------------------------------------


	initializeQueryBatch(&h_iQueryBatches, &iQueryTotalNumberByBatch, &h_iTermNumberByQuery);
//	iQueryTotalNumberByBatch=500;
	//Obtém a soma do número de termos de todas as queries (Esse valor pode ser pre-computado)
	int iTermTotalNumberOfBatch = 0;
	for (int i = 0; i < iQueryTotalNumberByBatch; ++i) {
		iTermTotalNumberOfBatch += h_iTermNumberByQuery[i];
	}
//	checkCudaErrors(cudaMalloc((void**)&m_iTermNumberByQuery, sizeof(int)*iQueryTotalNumberByBatch));
//	checkCudaErrors(cudaMalloc((void**)&m_ptrQueryPositions, sizeof(int)*iQueryTotalNumberByBatch));
//	checkCudaErrors(cudaMalloc((void**)&m_ptrInitPostingList, sizeof(int)*iQueryTotalNumberByBatch));
//	checkCudaErrors(cudaMalloc((void**)&m_iTermNumberByQuery, sizeof(int)*iQueryTotalNumberByBatch));
//
	checkCudaErrors(cudaHostAlloc(&m_iTermNumberByQuery, sizeof(int)*iQueryTotalNumberByBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_ptrQueryPositions, sizeof(int)*iQueryTotalNumberByBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_ptrInitPostingList, sizeof(long long)*iTermNumberInVocabulary, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_iQueryBatches, sizeof(int)*iTermTotalNumberOfBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	long long position = 0;
	for (int i = 0; i < iQueryTotalNumberByBatch; i++) {
		m_ptrQueryPositions[i] = position;
		m_iTermNumberByQuery[i] = h_iTermNumberByQuery[i];
		for (int term = 0; term < h_iTermNumberByQuery[i]; term++) {
			m_iQueryBatches[position] = h_iQueryBatches[i][term];//Alinha as queries para o mapeamento com a GPU
			position++;
		}
	}

	position = 0;
	for (int i = 0; i < iTermNumberInVocabulary; ++i) {//Obtém a posição inicial de todas as listas invertidas
		m_ptrInitPostingList[i] = position;
		position += h_iDocNumberByTermList[i];
	}

	checkCudaErrors(cudaHostGetDevicePointer(&d_iTermNumberByQuery, m_iTermNumberByQuery, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_ptrQueryPositions, m_ptrQueryPositions, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_iQueryBatches,m_iQueryBatches, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_ptrInitPostingList, m_ptrInitPostingList, 0 ) );

	//-------------------------------------

	cudaStream_t *streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * iQueryTotalNumberByBatch);

	int nbytes = sizeof(int**) * iQueryTotalNumberByBatch;
	checkCudaErrors(cudaHostAlloc((void **)&(m_iTopkDocList), nbytes, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	nbytes = sizeof(float**) * iQueryTotalNumberByBatch;
	checkCudaErrors(cudaHostAlloc((void **)&(m_dTopkScoreList), nbytes, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	int *queryListOrder = (int*) malloc(sizeof(int)*iQueryTotalNumberByBatch);
	int *sizePostingList = (int*) malloc(sizeof(int)*iQueryTotalNumberByBatch);

	int iMaxNumberInList;
	int* d_temp_iTopkDocList;
	float* d_temp_dTopkScoreList;
	for (int idQuery = 0; idQuery < iQueryTotalNumberByBatch; ++idQuery) {
		iMaxNumberInList = 0;
		for (int idTerm = 0; idTerm < h_iTermNumberByQuery[idQuery]; idTerm++) {
			if(iMaxNumberInList < h_iDocNumberByTermList[ h_iQueryBatches[idQuery][idTerm] ] )
				iMaxNumberInList = h_iDocNumberByTermList[ h_iQueryBatches[idQuery][idTerm] ];
		}

		sizePostingList[idQuery] = iMaxNumberInList;
		queryListOrder[idQuery] =  idQuery;

		int auxq=0, auxtamanho=0;

		int i = idQuery;
		while(i > 0){
			if(sizePostingList[i-1] < sizePostingList[i]){
				auxtamanho = sizePostingList[i-1];
				auxq = queryListOrder[i-1];

				sizePostingList[i-1] = sizePostingList[i];
				queryListOrder[i-1] = queryListOrder[i];

				sizePostingList[i] = auxtamanho;
				queryListOrder[i] = auxq;
			}
			else
				break;
			i--;
		}

		nbytes = iTopk * sizeof(int) * ((int) ceil((float) iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * 1 * iGlobalNumberRound)));
		checkCudaErrors(cudaMalloc((void**)&d_temp_iTopkDocList, nbytes)  );
		checkCudaErrors(cudaMemset(d_temp_iTopkDocList, -1, nbytes));

		nbytes = iTopk * sizeof(float)  * ((int) ceil((float) iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * 1 * iGlobalNumberRound)));
		checkCudaErrors(cudaMalloc((void**)&d_temp_dTopkScoreList, nbytes)  );
		checkCudaErrors(cudaMemset(d_temp_dTopkScoreList, 0.0, nbytes));

		m_iTopkDocList[idQuery]=d_temp_iTopkDocList;
		m_dTopkScoreList[idQuery]=d_temp_dTopkScoreList;

		d_temp_iTopkDocList = NULL;
		d_temp_dTopkScoreList = NULL;

		checkCudaErrors( cudaStreamCreate(&streams[idQuery]) );
	}
//	free(sizePostingList);
	//Processamento das Queries em Paralelo:

	double start;
	double end;

	int idQuery;
//	int* d_iTopkDocList;
//	float* d_dTopkScoreList;

//	int iMaxNumberInList;
	int iBlockNumber;
	dim3 threadsByBlock;
	dim3 blocksByGrid;

	int iTotalElementos; //= blocksByGrid.x * iTopk;
	int iProcessedListNumberbyBlock;// = iMergeNumberByBlock + 1;
	int iSkipTopKBetweenThreadBlocks;// = iProcessedListNumberbyBlock;
	int iSkipTopKBetweenMerges;// = 1;
	int exponent;// = 0;
	int exponent_b;// = 1;

	int* d_OrderQueryList;
	int n = 32;
	nbytes = n * sizeof(int);
	checkCudaErrors(cudaMalloc((void**)&d_OrderQueryList, nbytes)  );
	checkCudaErrors(cudaMemcpy(d_OrderQueryList, queryListOrder, nbytes, cudaMemcpyHostToDevice));

	start = omp_get_wtime();

	threadsByBlock = dim3(iBlockSize, 1);
	blocksByGrid   = dim3(n, 1);

	d_iTopkDocList = m_iTopkDocList[0];
	d_dTopkScoreList = m_dTopkScoreList[0];

	matchWandParallel_VARIABLE_Batch_Block_Test<<<blocksByGrid, threadsByBlock,0,
																streams[queryListOrder[0]]>>>(d_iDocIdList, d_iFreqDocList,
																			   d_dUBList, d_dIdfList, d_iDocLenghtList,
																			   (short int*)m_iTermNumberByQuery, d_iTopkDocList,
																			   d_dTopkScoreList,
																			   dAverageDocumentLength,
																			   iTopk,dInitialThreshold,
																			   d_iQueryBatches, d_ptrInitPostingList,
																			   d_ptrQueryPositions, d_iDocNumberByTermList,
																			   d_OrderQueryList);



	for(int i=n; i < 500;i++){
		idQuery = queryListOrder[i];

		iMaxNumberInList = sizePostingList[i];
		iBlockNumber= (int) ceil((float) iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * 1 * iGlobalNumberRound));
		threadsByBlock = dim3(iBlockSize, 1);
		blocksByGrid   = dim3(iBlockNumber, 1);

		sizePostingList[i] = iBlockNumber;

		d_iTopkDocList = m_iTopkDocList[idQuery];
		d_dTopkScoreList = m_dTopkScoreList[idQuery];

		matchWandParallel_BATCH_2<<<blocksByGrid,threadsByBlock, 0, streams[idQuery]>>>
												 (d_iDocIdList, d_iFreqDocList,
												 d_dUBList, d_dIdfList, d_iDocLenghtList,
												 m_iTermNumberByQuery[idQuery],
												 d_iTopkDocList, d_dTopkScoreList,
												 dAverageDocumentLength,iBlockNumberRound,
												 iGlobalNumberRound,iTopk,dInitialThreshold,
												 d_iQueryBatches, d_ptrInitPostingList,
												 d_ptrQueryPositions, idQuery, d_iDocNumberByTermList);

	}

	for(int i=n; i < 500;i++){
		idQuery = queryListOrder[i];
		d_iTopkDocList = m_iTopkDocList[idQuery];
		d_dTopkScoreList = m_dTopkScoreList[idQuery];

		iProcessedListNumberbyBlock = iMergeNumberByBlock + 1;
		iSkipTopKBetweenThreadBlocks = iProcessedListNumberbyBlock;
		iSkipTopKBetweenMerges = 1;
		exponent = 0;
		exponent_b = 1;

		iBlockNumber= ceilf( sizePostingList[i]/(iProcessedListNumberbyBlock) );
		threadsByBlock = dim3(iBlockSize, 1);
		blocksByGrid   = dim3(iBlockNumber, 1);
		iTotalElementos = blocksByGrid.x * iTopk;

		while(blocksByGrid.x >= 1){
			mergeTopkLists_v3<<<blocksByGrid, iTopk, 0,streams[idQuery]>>>(d_dTopkScoreList, d_iTopkDocList,
															 iTopk, iMergeNumberByBlock,
															 iSkipTopKBetweenMerges,
															 iSkipTopKBetweenThreadBlocks,iTotalElementos);

			blocksByGrid.x = ceilf (blocksByGrid.x/(iProcessedListNumberbyBlock) );
			exponent ++;
			exponent_b ++;
			iSkipTopKBetweenMerges = pow(iProcessedListNumberbyBlock, exponent);
			iSkipTopKBetweenThreadBlocks = pow(iProcessedListNumberbyBlock, exponent_b);
		}

//	    checkCudaErrors(cudaDeviceSynchronize());
//			int nbytes = iTopk * sizeof(int);
//			int *h_iTopkDocList = (int*) malloc(sizeof(int) * iTopk);
//			checkCudaErrors(cudaMemcpy(h_iTopkDocList, d_iTopkDocList, nbytes, cudaMemcpyDeviceToHost));
//
//			float *h_dTopkScoreList = (float*) malloc(sizeof(float) * iTopk);
//			nbytes = iTopk * sizeof(float);
//		    checkCudaErrors(cudaMemcpy(h_dTopkScoreList, d_dTopkScoreList, nbytes, cudaMemcpyDeviceToHost));
//
//		    checkCudaErrors(cudaDeviceSynchronize());
//
//		//    for (int j = 0; j < 500; ++j) {
//		    	printf("\n--- %d Query -----\n",idQuery);
//				for (int i = 0; i < iTopk; ++i) {
//					printf("--- %d %.2f----",h_iTopkDocList[i],h_dTopkScoreList[i]);
//				}
//		//	}
//
//		    free(h_iTopkDocList);
//		    free(h_dTopkScoreList);


	}
	checkCudaErrors(cudaDeviceSynchronize());

//    for(int idQuery=0; idQuery < iQueryTotalNumberByBatch;idQuery++){
//		checkCudaErrors(cudaStreamSynchronize(streams[idQuery]));
//	}

	end = omp_get_wtime();
	printf("Batch - Work took %f s\n", (end - start));
	printf("Batch %d, %d, %d, %d, %.4f \n",TOP_K,SHAREDTHESHOLD,DOC_QUANTITY_IN_MEMORY,iGlobalNumberRound, (end - start));

	free(queryListOrder);
	free(sizePostingList);
	for (int idQuery = 0; idQuery < iQueryTotalNumberByBatch; ++idQuery) {
		d_iTopkDocList = m_iTopkDocList[idQuery];
		d_dTopkScoreList = m_dTopkScoreList[idQuery];

		checkCudaErrors(cudaFree(d_iTopkDocList));
		checkCudaErrors(cudaFree(d_dTopkScoreList));
	}

	checkCudaErrors(cudaFree(d_OrderQueryList));
	checkCudaErrors(cudaFreeHost(m_iTopkDocList));
	checkCudaErrors(cudaFreeHost(m_dTopkScoreList));



	//-----------------------Destroy Stream Objects------------------------------
	for (int i = 0; i < iQueryTotalNumberByBatch; ++i)
		cudaStreamDestroy(streams[i]);
	//---------------------------------------------------------------------------

	//-----------------------Free GPU Memory------------------------------
	checkCudaErrors(cudaFreeHost(m_iTermNumberByQuery));
	checkCudaErrors(cudaFreeHost(m_iQueryBatches));
	checkCudaErrors(cudaFreeHost(m_ptrQueryPositions));
	checkCudaErrors(cudaFreeHost(m_ptrInitPostingList));

	freeAllocationOfInvertedListInGPU(d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
									  d_dUBList, d_dIdfList, d_iDocNumberByTermList);
	//----------------------------------------------------------------------------------------------------------

	//------------------Release resources in CPU-------------------------------
	#ifdef DEBUG
		printf("Releasing resources in CPU... ");
	#endif

	for (int i = 0; i < iTermNumberInVocabulary; ++i) {
		free(h_iDocIdList[i]);
		free(h_iDocLenghtList[i]);
		free(h_iFreqList[i]);
	}

	for (int i = 0; i < iQueryTotalNumberByBatch; ++i) {
		free(h_iQueryBatches[i]);
	}

//	free(streams);
	free(h_iDocIdList);
	free(h_iDocLenghtList);
	free(h_iFreqList);
	free(h_iQueryBatches);
	free(h_dIdfList);
	free(h_dUBList);
	free(h_iTermNumberByQuery);
	free(h_iDocNumberByTermList);
	free(h_iTopkDocList);
	free(h_dTopkScoreList);
	//-------------------------------------------------------------------------
	#ifdef DEBUG
		printf("OK!\n");
		printf("Finish!\n");
	#endif


}


__host__ void queryBatchProcessingHost(int iTopk, int iBlockSize, int iBlockNumberRound,
										int iGlobalNumberRound, int iMergeNumberByBlock,
										int iQueryType){

//	int iPostingTotalNumber = 0;
//	int iMaxNumberInList = 0;
	int iTermNumberInVocabulary;
	unsigned long long docTotalNumber = 0;

	//----------Host Variables-------------
	int **h_iDocIdList, **h_iDocLenghtList;
	unsigned short int** h_iFreqList;

	float *h_dUBList, *h_dIdfList;

	int* h_iDocNumberByTermList;

	int* h_iTopkDocList = (int*) malloc(iTopk * sizeof(int));
	float* h_dTopkScoreList = (float*) malloc(iTopk * sizeof(float));

	float dAverageDocumentLength;

	int *h_iTermNumberByQuery; //Número de termo por query do batch de query
	int **h_iQueryBatches; //Batch de Queries
	int iQueryTotalNumberByBatch = 500; //Número de query por batch
	//-------------------------------------

	//---------Mapped Variables in Host and Device------------------
	int *m_iTermNumberByQuery; //Número de termo por query do batch de query - Os indices representam as queries
	int *m_iQueryBatches; //Batch de Queries - Um conjunto de termos - Os termos pertence a um conjunto de queries
	int *m_ptrQueryPositions; //A posição das queries na lista de batch de queries; essa contagem é a realizada pela contagem dos termos

	long long* m_ptrInitPostingList; //Considera que há uma única lista no acelerador  - Posição inicial das listas de postings dos termos

	int** m_iTopkDocList; // Topk
	float** m_dTopkScoreList; // Topk
	//---------------------------------------------------------------

	//----------Device Variables------------------------------------


	float dInitialThreshold = 0, *d_dUBList, *d_dIdfList;

	int *d_iDocIdList, *d_iDocLenghtList;
	unsigned short int* d_iFreqDocList;

	int* d_iDocNumberByTermList;
	int* d_iTopkDocList;
	float* d_dTopkScoreList;

	int *d_iTermNumberByQuery, *d_iQueryBatches, *d_ptrQueryPositions;
	long long *d_ptrInitPostingList;

	//-------------------------------------

	initializeInvertedIndex(&iTermNumberInVocabulary, &dAverageDocumentLength, &h_iDocIdList,
							&h_iDocLenghtList, &h_iFreqList, &h_dIdfList, &h_dUBList,
							&h_iDocNumberByTermList, &docTotalNumber);

	//-------------------GPU Pre-Configuration---------------------
	cudaSetDevice(0);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	//-------------------------------------------------------------

	//------------Allocating and Coping Pointer to Pointer in Device Memory--------------------------------------
	memoryAllocationOfInvertedIndexInGPU(docTotalNumber, iTermNumberInVocabulary,&d_iDocIdList,
										 &d_iDocLenghtList, &d_iFreqDocList,&d_dUBList,
									     &d_dIdfList, &d_iDocNumberByTermList);

	memoryCopyOfSingleQuery_To_Device(docTotalNumber, iTermNumberInVocabulary,h_iDocNumberByTermList,
		    						  d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
		    						  d_dUBList, d_dIdfList, d_iDocNumberByTermList, h_iDocIdList,
		    						  h_iDocLenghtList, h_iFreqList, h_dUBList, h_dIdfList);
	//----------------------------------------------------------------------------------------------------------


	initializeQueryBatch(&h_iQueryBatches, &iQueryTotalNumberByBatch, &h_iTermNumberByQuery);
//	iQueryTotalNumberByBatch=500;
	//Obtém a soma do número de termos de todas as queries (Esse valor pode ser pre-computado)
	int iTermTotalNumberOfBatch = 0;
	for (int i = 0; i < iQueryTotalNumberByBatch; ++i) {
		iTermTotalNumberOfBatch += h_iTermNumberByQuery[i];
	}
//	checkCudaErrors(cudaMalloc((void**)&m_iTermNumberByQuery, sizeof(int)*iQueryTotalNumberByBatch));
//	checkCudaErrors(cudaMalloc((void**)&m_ptrQueryPositions, sizeof(int)*iQueryTotalNumberByBatch));
//	checkCudaErrors(cudaMalloc((void**)&m_ptrInitPostingList, sizeof(int)*iQueryTotalNumberByBatch));
//	checkCudaErrors(cudaMalloc((void**)&m_iTermNumberByQuery, sizeof(int)*iQueryTotalNumberByBatch));
//
	checkCudaErrors(cudaHostAlloc(&m_iTermNumberByQuery, sizeof(int)*iQueryTotalNumberByBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_ptrQueryPositions, sizeof(int)*iQueryTotalNumberByBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_ptrInitPostingList, sizeof(long long)*iTermNumberInVocabulary, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_iQueryBatches, sizeof(int)*iTermTotalNumberOfBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	long long position = 0;
	for (int i = 0; i < iQueryTotalNumberByBatch; i++) {
		m_ptrQueryPositions[i] = position;
		m_iTermNumberByQuery[i] = h_iTermNumberByQuery[i];
		for (int term = 0; term < h_iTermNumberByQuery[i]; term++) {
			m_iQueryBatches[position] = h_iQueryBatches[i][term];//Alinha as queries para o mapeamento com a GPU
			position++;
		}
	}

	position = 0;
	for (int i = 0; i < iTermNumberInVocabulary; ++i) {//Obtém a posição inicial de todas as listas invertidas
		m_ptrInitPostingList[i] = position;
		position += h_iDocNumberByTermList[i];
	}

	checkCudaErrors(cudaHostGetDevicePointer(&d_iTermNumberByQuery, m_iTermNumberByQuery, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_ptrQueryPositions, m_ptrQueryPositions, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_iQueryBatches,m_iQueryBatches, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_ptrInitPostingList, m_ptrInitPostingList, 0 ) );

	//-------------------------------------

	cudaStream_t *streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * iQueryTotalNumberByBatch);

	int nbytes = sizeof(int**) * iQueryTotalNumberByBatch;
	checkCudaErrors(cudaHostAlloc((void **)&(m_iTopkDocList), nbytes, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	nbytes = sizeof(float**) * iQueryTotalNumberByBatch;
	checkCudaErrors(cudaHostAlloc((void **)&(m_dTopkScoreList), nbytes, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	int *queryListOrder = (int*) malloc(sizeof(int)*iQueryTotalNumberByBatch);
	int *sizePostingList = (int*) malloc(sizeof(int)*iQueryTotalNumberByBatch);

	int iMaxNumberInList;
	int* d_temp_iTopkDocList;
	float* d_temp_dTopkScoreList;
	for (int idQuery = 0; idQuery < iQueryTotalNumberByBatch; ++idQuery) {
		iMaxNumberInList = 0;
		for (int idTerm = 0; idTerm < h_iTermNumberByQuery[idQuery]; idTerm++) {
				if(iMaxNumberInList < h_iDocNumberByTermList[ h_iQueryBatches[idQuery][idTerm] ] )
					iMaxNumberInList = h_iDocNumberByTermList[ h_iQueryBatches[idQuery][idTerm] ];
		}

		sizePostingList[idQuery] = iMaxNumberInList;
		queryListOrder[idQuery] =  idQuery;

		int auxq=0, auxtamanho=0;

		int i = idQuery;
		while(i > 0){
			if(sizePostingList[i-1] < sizePostingList[i]){
				auxtamanho = sizePostingList[i-1];
				auxq = queryListOrder[i-1];

				sizePostingList[i-1] = sizePostingList[i];
				queryListOrder[i-1] = queryListOrder[i];

				sizePostingList[i] = auxtamanho;
				queryListOrder[i] = auxq;
			}
			else
				break;
			i--;
		}

		nbytes = iTopk * sizeof(int) * ((int) ceil((float) iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * 1 * iGlobalNumberRound)));
		checkCudaErrors(cudaMalloc((void**)&d_temp_iTopkDocList, nbytes)  );
		checkCudaErrors(cudaMemset(d_temp_iTopkDocList, -1, nbytes));

		nbytes = iTopk * sizeof(float)  * ((int) ceil((float) iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * 1 * iGlobalNumberRound)));
		checkCudaErrors(cudaMalloc((void**)&d_temp_dTopkScoreList, nbytes)  );
		checkCudaErrors(cudaMemset(d_temp_dTopkScoreList, 0.0, nbytes));

		m_iTopkDocList[idQuery]=d_temp_iTopkDocList;
		m_dTopkScoreList[idQuery]=d_temp_dTopkScoreList;

		d_temp_iTopkDocList = NULL;
		d_temp_dTopkScoreList = NULL;

		checkCudaErrors( cudaStreamCreate(&streams[idQuery]) );
	}
//	free(sizePostingList);
	//Processamento das Queries em Paralelo:

	double start;
	double end;
//	int idQuery;

	start = omp_get_wtime();
//#pragma omp parallel num_threads(1)  //private(idQuery)
//{


//	#pragma omp single
//	{
//		int idThread = omp_get_thread_num();
//		int portion = iQueryTotalNumberByBatch/omp_get_num_threads();
//		int pointStart = portion * idThread;

//		#pragma omp single
//		{
//			start = omp_get_wtime();
//		}
//	   #pragma omp taskgroup
//		{
//			#pragma omp parallel for num_threads(8)
			for(int i=0; i < 500;i++){
//			for (int idQuery = pointStart; idQuery < pointStart+portion ; ++idQuery) {
//				#pragma omp task //firstprivate(idQuery)
//				{
					int idQuery = queryListOrder[i];
//					if(idQuery != 8)
//						continue;
					int* d_iTopkDocList;
					float* d_dTopkScoreList;
					#ifdef DEBUG
						printf("idThreadCPU: %d\n",omp_get_thread_num());
						printf("idQuery: %d\n",idQuery);
						printf("idTerm: ");

						for (int i = 0; i < h_iTermNumberByQuery[idQuery]; ++i) {
							printf("%d ",h_iQueryBatches[idQuery][i]);
						}

						printf("\n");
					#endif

					#ifdef DEBUG
						int totalDoc = 0;
					#endif

					int iMaxNumberInList= sizePostingList[i];
//					for (int idTerm = 0; idTerm < h_iTermNumberByQuery[idQuery]; idTerm++) {
//						if(iMaxNumberInList < h_iDocNumberByTermList[ h_iQueryBatches[idQuery][idTerm] ] )
//							iMaxNumberInList = h_iDocNumberByTermList[ h_iQueryBatches[idQuery][idTerm] ];
//
//						#ifdef DEBUG
//							totalDoc +=h_iDocNumberByTermList[ h_iQueryBatches[idQuery][idTerm]];
//						#endif
//					}
					#ifdef DEBUG
						printf("Total of Docs: %d - Max of Doc: %d \n",totalDoc, iMaxNumberInList);
					#endif
					//------------Kernel Launch Configuration---------------------------------
					#ifdef DEBUG
						printf("Kernel Launch Configuration... \n");
					#endif

					int iBlockNumber= (int) ceil((float) iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * 1 * iGlobalNumberRound));
					dim3 threadsByBlock = dim3(iBlockSize, 1);
					dim3 blocksByGrid   = dim3(iBlockNumber, 1);

					#ifdef DEBUG
						printf("#threads by block: %i, #blocks: %i ... OK\n",threadsByBlock.x, blocksByGrid.x) ;
					#endif
					//-------------------------------------------------------------------------

					//------------Allocation memory of Top-k Lists---------------------------------
					#ifdef DEBUG
						printf("Allocation memory of Top-k Lists... ");
					#endif

					d_iTopkDocList = m_iTopkDocList[idQuery];
					d_dTopkScoreList = m_dTopkScoreList[idQuery];

					#ifdef DEBUG
						printf("OK\n");
					#endif
					//-------------------------------------------------------------------------

					//-----------Initializing the initial value of Threshold if it is necessary ---------------------
			//		if(iQueryType){
			//			setThresholdForANDQueryOnInvertedList(&dInitialThreshold, h_dUBList,
			//												  h_iQueryBatches[idQuery], h_iTermNumberByQuery[idQuery]);
			//		}
					//-----------------------------------------------------------------------------------------------

					//--------------------------Call to Kernels----------------------------
					callToBatchKernels(blocksByGrid, threadsByBlock, iMergeNumberByBlock, iTopk,
								  dAverageDocumentLength, dInitialThreshold, iBlockNumberRound,
								  iGlobalNumberRound, m_iTermNumberByQuery[idQuery], d_dUBList, d_dIdfList,
								  d_iDocIdList, d_iFreqDocList, d_iDocLenghtList,d_iTopkDocList,
								  d_dTopkScoreList, idQuery,
								  streams[idQuery], d_iQueryBatches,d_ptrInitPostingList,d_ptrQueryPositions,
								  d_iDocNumberByTermList);

					//----------------------------------------------------------------------------------------------------------
				}
//			}//Task
//
//		}//TaskGroup-1

//		#pragma omp taskgroup
//		{
//			for (int idQuery = pointStart; idQuery < pointStart+portion ; ++idQuery) {
//			#pragma omp for nowait
			for(int idQuery=0; idQuery < iQueryTotalNumberByBatch;idQuery++){
//				#pragma omp task //firstprivate(idQuery)
//				{
					cudaStreamSynchronize(streams[idQuery]);
//				}
			}
//		}
//	}//Single
//}

	end = omp_get_wtime();
	printf("Batch - Work took %f s\n", (end - start));
	printf("Batch %d, %d, %d, %d, %.4f \n",TOP_K,SHAREDTHESHOLD,DOC_QUANTITY_IN_MEMORY,iGlobalNumberRound, (end - start));

	free(queryListOrder);
	free(sizePostingList);
	for (int idQuery = 0; idQuery < iQueryTotalNumberByBatch; ++idQuery) {
		d_iTopkDocList = m_iTopkDocList[idQuery];
		d_dTopkScoreList = m_dTopkScoreList[idQuery];

		checkCudaErrors(cudaFree(d_iTopkDocList));
		checkCudaErrors(cudaFree(d_dTopkScoreList));
	}

	checkCudaErrors(cudaFreeHost(m_iTopkDocList));
	checkCudaErrors(cudaFreeHost(m_dTopkScoreList));



	//-----------------------Destroy Stream Objects------------------------------
	for (int i = 0; i < iQueryTotalNumberByBatch; ++i)
		cudaStreamDestroy(streams[i]);
	//---------------------------------------------------------------------------

	//-----------------------Free GPU Memory------------------------------
	checkCudaErrors(cudaFreeHost(m_iTermNumberByQuery));
	checkCudaErrors(cudaFreeHost(m_iQueryBatches));
	checkCudaErrors(cudaFreeHost(m_ptrQueryPositions));
	checkCudaErrors(cudaFreeHost(m_ptrInitPostingList));

	freeAllocationOfInvertedListInGPU(d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
									  d_dUBList, d_dIdfList, d_iDocNumberByTermList);
	//----------------------------------------------------------------------------------------------------------

	//------------------Release resources in CPU-------------------------------
	#ifdef DEBUG
		printf("Releasing resources in CPU... ");
	#endif

	for (int i = 0; i < iTermNumberInVocabulary; ++i) {
		free(h_iDocIdList[i]);
		free(h_iDocLenghtList[i]);
		free(h_iFreqList[i]);
	}

	for (int i = 0; i < iQueryTotalNumberByBatch; ++i) {
		free(h_iQueryBatches[i]);
	}

//	free(streams);
	free(h_iDocIdList);
	free(h_iDocLenghtList);
	free(h_iFreqList);
	free(h_iQueryBatches);
	free(h_dIdfList);
	free(h_dUBList);
	free(h_iTermNumberByQuery);
	free(h_iDocNumberByTermList);
	free(h_iTopkDocList);
	free(h_dTopkScoreList);
	//-------------------------------------------------------------------------
	#ifdef DEBUG
		printf("OK!\n");
		printf("Finish!\n");
	#endif
}

__host__ void queryBatchProcessingHost_2(int iTopk, int iBlockSize, int iBlockNumberRound,
										int iGlobalNumberRound, int iMergeNumberByBlock,
										int iQueryType){

//	int iPostingTotalNumber = 0;
//	int iMaxNumberInList = 0;
	int iTermNumberInVocabulary;
	unsigned long long docTotalNumber = 0;

	//----------Host Variables-------------
	int **h_iDocIdList, **h_iDocLenghtList;
	unsigned short int** h_iFreqList;

	float *h_dUBList, *h_dIdfList;

	int* h_iDocNumberByTermList;

	int* h_iTopkDocList = (int*) malloc(iTopk * sizeof(int));
	float* h_dTopkScoreList = (float*) malloc(iTopk * sizeof(float));

	float dAverageDocumentLength;

	int *h_iTermNumberByQuery; //Número de termo por query do batch de query
	int **h_iQueryBatches; //Batch de Queries
	int iQueryTotalNumberByBatch = 500; //Número de query por batch
	//-------------------------------------

	//---------Mapped Variables in Host and Device------------------
	int *m_iTermNumberByQuery; //Número de termo por query do batch de query - Os indices representam as queries
	int *m_iQueryBatches; //Batch de Queries - Um conjunto de termos - Os termos pertence a um conjunto de queries
	int *m_ptrQueryPositions; //A posição das queries na lista de batch de queries; essa contagem é a realizada pela contagem dos termos

	long long* m_ptrInitPostingList; //Considera que há uma única lista no acelerador  - Posição inicial das listas de postings dos termos

	int** m_iTopkDocList; // Topk
	float** m_dTopkScoreList; // Topk
	//---------------------------------------------------------------

	//----------Device Variables------------------------------------


	float dInitialThreshold = 0, *d_dUBList, *d_dIdfList;

	int *d_iDocIdList, *d_iDocLenghtList;
	unsigned short int* d_iFreqDocList;

	int* d_iDocNumberByTermList;
	int* d_iTopkDocList;
	float* d_dTopkScoreList;

	int *d_iTermNumberByQuery, *d_iQueryBatches, *d_ptrQueryPositions;
	long long *d_ptrInitPostingList;

	//-------------------------------------

	initializeInvertedIndex(&iTermNumberInVocabulary, &dAverageDocumentLength, &h_iDocIdList,
							&h_iDocLenghtList, &h_iFreqList, &h_dIdfList, &h_dUBList,
							&h_iDocNumberByTermList, &docTotalNumber);

	//-------------------GPU Pre-Configuration---------------------
	cudaSetDevice(0);
	cudaSetDeviceFlags(cudaDeviceMapHost);

	//-------------------------------------------------------------

	//------------Allocating and Coping Pointer to Pointer in Device Memory--------------------------------------
	memoryAllocationOfInvertedIndexInGPU(docTotalNumber, iTermNumberInVocabulary,&d_iDocIdList,
										 &d_iDocLenghtList, &d_iFreqDocList,&d_dUBList,
									     &d_dIdfList, &d_iDocNumberByTermList);

	memoryCopyOfSingleQuery_To_Device(docTotalNumber, iTermNumberInVocabulary,h_iDocNumberByTermList,
		    						  d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
		    						  d_dUBList, d_dIdfList, d_iDocNumberByTermList, h_iDocIdList,
		    						  h_iDocLenghtList, h_iFreqList, h_dUBList, h_dIdfList);
	//----------------------------------------------------------------------------------------------------------


	initializeQueryBatch(&h_iQueryBatches, &iQueryTotalNumberByBatch, &h_iTermNumberByQuery);
//	iQueryTotalNumberByBatch=500;
	//Obtém a soma do número de termos de todas as queries (Esse valor pode ser pre-computado)
	int iTermTotalNumberOfBatch = 0;
	for (int i = 0; i < iQueryTotalNumberByBatch; ++i) {
		iTermTotalNumberOfBatch += h_iTermNumberByQuery[i];
	}
//	checkCudaErrors(cudaMalloc((void**)&m_iTermNumberByQuery, sizeof(int)*iQueryTotalNumberByBatch));
//	checkCudaErrors(cudaMalloc((void**)&m_ptrQueryPositions, sizeof(int)*iQueryTotalNumberByBatch));
//	checkCudaErrors(cudaMalloc((void**)&m_ptrInitPostingList, sizeof(int)*iQueryTotalNumberByBatch));
//	checkCudaErrors(cudaMalloc((void**)&m_iTermNumberByQuery, sizeof(int)*iQueryTotalNumberByBatch));
//
	checkCudaErrors(cudaHostAlloc(&m_iTermNumberByQuery, sizeof(int)*iQueryTotalNumberByBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_ptrQueryPositions, sizeof(int)*iQueryTotalNumberByBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_ptrInitPostingList, sizeof(long long)*iTermNumberInVocabulary, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_iQueryBatches, sizeof(int)*iTermTotalNumberOfBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	long long position = 0;
	for (int i = 0; i < iQueryTotalNumberByBatch; i++) {
		m_ptrQueryPositions[i] = position;
		m_iTermNumberByQuery[i] = h_iTermNumberByQuery[i];
		for (int term = 0; term < h_iTermNumberByQuery[i]; term++) {
			m_iQueryBatches[position] = h_iQueryBatches[i][term];//Alinha as queries para o mapeamento com a GPU
			position++;
		}
	}

	position = 0;
	for (int i = 0; i < iTermNumberInVocabulary; ++i) {//Obtém a posição inicial de todas as listas invertidas
		m_ptrInitPostingList[i] = position;
		position += h_iDocNumberByTermList[i];
	}

	checkCudaErrors(cudaHostGetDevicePointer(&d_iTermNumberByQuery, m_iTermNumberByQuery, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_ptrQueryPositions, m_ptrQueryPositions, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_iQueryBatches,m_iQueryBatches, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_ptrInitPostingList, m_ptrInitPostingList, 0 ) );

	//-------------------------------------

	cudaStream_t *streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * iQueryTotalNumberByBatch);

	int nbytes = sizeof(int**) * iQueryTotalNumberByBatch;
	checkCudaErrors(cudaHostAlloc((void **)&(m_iTopkDocList), nbytes, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	nbytes = sizeof(float**) * iQueryTotalNumberByBatch;
	checkCudaErrors(cudaHostAlloc((void **)&(m_dTopkScoreList), nbytes, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	int *queryListOrder = (int*) malloc(sizeof(int)*iQueryTotalNumberByBatch);
	int *sizePostingList = (int*) malloc(sizeof(int)*iQueryTotalNumberByBatch);

	int iMaxNumberInList;
	int* d_temp_iTopkDocList;
	float* d_temp_dTopkScoreList;
	for (int idQuery = 0; idQuery < iQueryTotalNumberByBatch; ++idQuery) {
		iMaxNumberInList = 0;
		for (int idTerm = 0; idTerm < h_iTermNumberByQuery[idQuery]; idTerm++) {
			if(iMaxNumberInList < h_iDocNumberByTermList[ h_iQueryBatches[idQuery][idTerm] ] )
				iMaxNumberInList = h_iDocNumberByTermList[ h_iQueryBatches[idQuery][idTerm] ];
		}

		sizePostingList[idQuery] = iMaxNumberInList;
		queryListOrder[idQuery] =  idQuery;

		int auxq=0, auxtamanho=0;

		int i = idQuery;
		while(i > 0){
			if(sizePostingList[i-1] < sizePostingList[i]){
				auxtamanho = sizePostingList[i-1];
				auxq = queryListOrder[i-1];

				sizePostingList[i-1] = sizePostingList[i];
				queryListOrder[i-1] = queryListOrder[i];

				sizePostingList[i] = auxtamanho;
				queryListOrder[i] = auxq;
			}
			else
				break;
			i--;
		}

		nbytes = iTopk * sizeof(int) * ((int) ceil((float) iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * 1 * iGlobalNumberRound)));
		checkCudaErrors(cudaMalloc((void**)&d_temp_iTopkDocList, nbytes)  );
		checkCudaErrors(cudaMemset(d_temp_iTopkDocList, -1, nbytes));

		nbytes = iTopk * sizeof(float)  * ((int) ceil((float) iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * 1 * iGlobalNumberRound)));
		checkCudaErrors(cudaMalloc((void**)&d_temp_dTopkScoreList, nbytes)  );
		checkCudaErrors(cudaMemset(d_temp_dTopkScoreList, 0.0, nbytes));

		m_iTopkDocList[idQuery]=d_temp_iTopkDocList;
		m_dTopkScoreList[idQuery]=d_temp_dTopkScoreList;

		d_temp_iTopkDocList = NULL;
		d_temp_dTopkScoreList = NULL;

		checkCudaErrors( cudaStreamCreate(&streams[idQuery]) );
	}
//	free(sizePostingList);
	//Processamento das Queries em Paralelo:

	double start;
	double end;

	int idQuery;
//	int* d_iTopkDocList;
//	float* d_dTopkScoreList;

//	int iMaxNumberInList;
	int iBlockNumber;
	dim3 threadsByBlock;
	dim3 blocksByGrid;

	int iTotalElementos; //= blocksByGrid.x * iTopk;
	int iProcessedListNumberbyBlock;// = iMergeNumberByBlock + 1;
	int iSkipTopKBetweenThreadBlocks;// = iProcessedListNumberbyBlock;
	int iSkipTopKBetweenMerges;// = 1;
	int exponent;// = 0;
	int exponent_b;// = 1;

	start = omp_get_wtime();

	for(int i=0; i < iQueryTotalNumberByBatch;i++){
		idQuery = queryListOrder[i];

		iMaxNumberInList = sizePostingList[i];
		iBlockNumber= (int) ceil((float) iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * 1 * iGlobalNumberRound));
		threadsByBlock = dim3(iBlockSize, 1);
		blocksByGrid   = dim3(iBlockNumber, 1);

		sizePostingList[i] = iBlockNumber;

		d_iTopkDocList = m_iTopkDocList[idQuery];
		d_dTopkScoreList = m_dTopkScoreList[idQuery];

		matchWandParallel_BATCH_2<<<blocksByGrid,threadsByBlock, 0, streams[idQuery]>>>
												 (d_iDocIdList, d_iFreqDocList,
												 d_dUBList, d_dIdfList, d_iDocLenghtList,
												 m_iTermNumberByQuery[idQuery],
												 d_iTopkDocList, d_dTopkScoreList,
												 dAverageDocumentLength,iBlockNumberRound,
												 iGlobalNumberRound,iTopk,dInitialThreshold,
												 d_iQueryBatches, d_ptrInitPostingList,
												 d_ptrQueryPositions, idQuery, d_iDocNumberByTermList);

	}

	for(int i=0; i < iQueryTotalNumberByBatch;i++){
		idQuery = queryListOrder[i];
		d_iTopkDocList = m_iTopkDocList[idQuery];
		d_dTopkScoreList = m_dTopkScoreList[idQuery];

		iProcessedListNumberbyBlock = iMergeNumberByBlock + 1;
		iSkipTopKBetweenThreadBlocks = iProcessedListNumberbyBlock;
		iSkipTopKBetweenMerges = 1;
		exponent = 0;
		exponent_b = 1;

		iBlockNumber= ceilf( sizePostingList[i]/(iProcessedListNumberbyBlock) );
		threadsByBlock = dim3(iBlockSize, 1);
		blocksByGrid   = dim3(iBlockNumber, 1);
		iTotalElementos = blocksByGrid.x * iTopk;

		while(blocksByGrid.x >= 1){
			mergeTopkLists_v3<<<blocksByGrid, iTopk, 0,streams[idQuery]>>>(d_dTopkScoreList, d_iTopkDocList,
															 iTopk, iMergeNumberByBlock,
															 iSkipTopKBetweenMerges,
															 iSkipTopKBetweenThreadBlocks,iTotalElementos);

			blocksByGrid.x = ceilf (blocksByGrid.x/(iProcessedListNumberbyBlock) );
			exponent ++;
			exponent_b ++;
			iSkipTopKBetweenMerges = pow(iProcessedListNumberbyBlock, exponent);
			iSkipTopKBetweenThreadBlocks = pow(iProcessedListNumberbyBlock, exponent_b);
		}

//	    checkCudaErrors(cudaDeviceSynchronize());
//			int nbytes = iTopk * sizeof(int);
//			int *h_iTopkDocList = (int*) malloc(sizeof(int) * iTopk);
//			checkCudaErrors(cudaMemcpy(h_iTopkDocList, d_iTopkDocList, nbytes, cudaMemcpyDeviceToHost));
//
//			float *h_dTopkScoreList = (float*) malloc(sizeof(float) * iTopk);
//			nbytes = iTopk * sizeof(float);
//		    checkCudaErrors(cudaMemcpy(h_dTopkScoreList, d_dTopkScoreList, nbytes, cudaMemcpyDeviceToHost));
//
//		    checkCudaErrors(cudaDeviceSynchronize());
//
//		//    for (int j = 0; j < 500; ++j) {
//		    	printf("\n--- %d Query -----\n",idQuery);
//				for (int i = 0; i < iTopk; ++i) {
//					printf("--- %d %.2f----",h_iTopkDocList[i],h_dTopkScoreList[i]);
//				}
//		//	}
//
//		    free(h_iTopkDocList);
//		    free(h_dTopkScoreList);


	}
    checkCudaErrors(cudaDeviceSynchronize());

//    for(int idQuery=0; idQuery < iQueryTotalNumberByBatch;idQuery++){
//		checkCudaErrors(cudaStreamSynchronize(streams[idQuery]));
//	}

	end = omp_get_wtime();
	printf("Batch - Work took %f s\n", (end - start));
	printf("Batch %d, %d, %d, %d, %.4f \n",TOP_K,SHAREDTHESHOLD,DOC_QUANTITY_IN_MEMORY,iGlobalNumberRound, (end - start));

	free(queryListOrder);
	free(sizePostingList);
	for (int idQuery = 0; idQuery < iQueryTotalNumberByBatch; ++idQuery) {
		d_iTopkDocList = m_iTopkDocList[idQuery];
		d_dTopkScoreList = m_dTopkScoreList[idQuery];

		checkCudaErrors(cudaFree(d_iTopkDocList));
		checkCudaErrors(cudaFree(d_dTopkScoreList));
	}

	checkCudaErrors(cudaFreeHost(m_iTopkDocList));
	checkCudaErrors(cudaFreeHost(m_dTopkScoreList));



	//-----------------------Destroy Stream Objects------------------------------
	for (int i = 0; i < iQueryTotalNumberByBatch; ++i)
		cudaStreamDestroy(streams[i]);
	//---------------------------------------------------------------------------

	//-----------------------Free GPU Memory------------------------------
	checkCudaErrors(cudaFreeHost(m_iTermNumberByQuery));
	checkCudaErrors(cudaFreeHost(m_iQueryBatches));
	checkCudaErrors(cudaFreeHost(m_ptrQueryPositions));
	checkCudaErrors(cudaFreeHost(m_ptrInitPostingList));

	freeAllocationOfInvertedListInGPU(d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
									  d_dUBList, d_dIdfList, d_iDocNumberByTermList);
	//----------------------------------------------------------------------------------------------------------

	//------------------Release resources in CPU-------------------------------
	#ifdef DEBUG
		printf("Releasing resources in CPU... ");
	#endif

	for (int i = 0; i < iTermNumberInVocabulary; ++i) {
		free(h_iDocIdList[i]);
		free(h_iDocLenghtList[i]);
		free(h_iFreqList[i]);
	}

	for (int i = 0; i < iQueryTotalNumberByBatch; ++i) {
		free(h_iQueryBatches[i]);
	}

//	free(streams);
	free(h_iDocIdList);
	free(h_iDocLenghtList);
	free(h_iFreqList);
	free(h_iQueryBatches);
	free(h_dIdfList);
	free(h_dUBList);
	free(h_iTermNumberByQuery);
	free(h_iDocNumberByTermList);
	free(h_iTopkDocList);
	free(h_dTopkScoreList);
	//-------------------------------------------------------------------------
	#ifdef DEBUG
		printf("OK!\n");
		printf("Finish!\n");
	#endif
}



__host__ void queryBatchProcessingHost_ByBlock(int iTopk, int iBlockSize, int iBlockNumberRound,
											   int iGlobalNumberRound, int iMergeNumberByBlock,
											   int iQueryType, int iBatchSize){

//	int iPostingTotalNumber = 0;
//	int iMaxNumberInList = 0;
	int iTermNumberInVocabulary;
	unsigned long long docTotalNumber = 0;

	//----------Host Variables-------------
	int **h_iDocIdList, **h_iDocLenghtList;
	unsigned short int** h_iFreqList;


	float *h_dUBList, *h_dIdfList;

	int* h_iDocNumberByTermList;

//	int* h_iTopkDocList = (int*) malloc(iTopk * sizeof(int) * iBatchSize);
//	float* h_dTopkScoreList = (float*) malloc(iTopk * sizeof(float) * iBatchSize);

	float dAverageDocumentLength;

	int *h_iTermNumberByQuery; //Número de termo por query do batch de query
	int **h_iQueryBatches; //Batch de Queries contém as queries
	int iQueryTotalNumberByBatch=500; //Número total de query
	//-------------------------------------

	//---------Mapped Variables in Host and Device------------------
	short int *m_iTermNumberByQuery; //Número de termo por query do batch de query - Os indices representam as queries
	int *m_iQueryBatches; //Batch de Queries - Um conjunto de termos - Os termos pertence a um conjunto de queries
	int *m_ptrQueryPositions; //A posição das queries na lista de batch de queries; essa contagem é a realizada pela contagem dos termos

	long long* m_ptrInitPostingList; //Contém as listas de postings de todos os termos do vocabulário - Considera que há uma única lista no acelerador

//	int* m_iTopkDocList;
//	float* m_dTopkScoreList;
	//---------------------------------------------------------------

	//----------Device Variables------------------------------------


	float dInitialThreshold = 0, *d_dUBList, *d_dIdfList;

	int *d_iDocIdList, *d_iDocLenghtList;
	unsigned short int* d_iFreqDocList;

	int* d_iDocNumberByTermList;
	int* d_iTopkDocList;
	float* d_dTopkScoreList;

	int *d_iTermNumberByQuery, *d_iQueryBatches, *d_ptrQueryPositions;
	long long *d_ptrInitPostingList;

	//-------------------------------------

	initializeInvertedIndex(&iTermNumberInVocabulary, &dAverageDocumentLength, &h_iDocIdList,
							&h_iDocLenghtList, &h_iFreqList, &h_dIdfList, &h_dUBList,
							&h_iDocNumberByTermList, &docTotalNumber);

	//-------------------GPU Pre-Configuration---------------------
	cudaSetDevice(0);
	cudaSetDeviceFlags(cudaDeviceMapHost);
	//-------------------------------------------------------------

	//------------Allocating and Coping Pointer to Pointer in Device Memory--------------------------------------
	memoryAllocationOfInvertedIndexInGPU(docTotalNumber, iTermNumberInVocabulary,&d_iDocIdList,
										 &d_iDocLenghtList, &d_iFreqDocList,&d_dUBList,
									     &d_dIdfList, &d_iDocNumberByTermList);

	memoryCopyOfSingleQuery_To_Device(docTotalNumber, iTermNumberInVocabulary,h_iDocNumberByTermList,
		    						  d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
		    						  d_dUBList, d_dIdfList, d_iDocNumberByTermList, h_iDocIdList,
		    						  h_iDocLenghtList, h_iFreqList, h_dUBList, h_dIdfList);
	//----------------------------------------------------------------------------------------------------------


	initializeQueryBatch(&h_iQueryBatches, &iQueryTotalNumberByBatch, &h_iTermNumberByQuery);
	//Obtém o número total de termos nas queries (Esse valor pode ser pre-computado)
	int iTermTotalNumberOfBatch = 0;
	for (int i = 0; i < iQueryTotalNumberByBatch; ++i) {
		iTermTotalNumberOfBatch += h_iTermNumberByQuery[i];
	}

	checkCudaErrors(cudaHostAlloc(&m_iTermNumberByQuery, sizeof(short int)*iQueryTotalNumberByBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_ptrQueryPositions, sizeof(int)*iQueryTotalNumberByBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_ptrInitPostingList, sizeof(long long)*iTermNumberInVocabulary, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&m_iQueryBatches, sizeof(int)*iTermTotalNumberOfBatch, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	long long position = 0;
	for (int i = 0; i < iQueryTotalNumberByBatch; i++) {
			printf("\n--- %d Query ",i);
			m_ptrQueryPositions[i] = position;
			m_iTermNumberByQuery[i] = h_iTermNumberByQuery[i];
		for (int term = 0; term < h_iTermNumberByQuery[i]; term++) {
			printf("%d ",h_iQueryBatches[i][term]);
			m_iQueryBatches[position] = h_iQueryBatches[i][term];

			position++;
		}
	}

	position = 0;
	for (int i = 0; i < iTermNumberInVocabulary; ++i) {

//		#ifdef DEBUG
//			if (i == 4558 || i == 2515)
//				printf("STOP");
//		#endif
		m_ptrInitPostingList[i] = position;
		position += h_iDocNumberByTermList[i];
	}

	checkCudaErrors(cudaHostGetDevicePointer(&d_iTermNumberByQuery, m_iTermNumberByQuery, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_ptrQueryPositions, m_ptrQueryPositions, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_iQueryBatches,m_iQueryBatches, 0 ) );
	checkCudaErrors(cudaHostGetDevicePointer(&d_ptrInitPostingList, m_ptrInitPostingList, 0 ) );

	//------------Allocation memory of Top-k Lists---------------------------------
	#ifdef DEBUG
		printf("Allocation memory of Top-k Lists... ");
	#endif


	int nbytes = iTopk * sizeof(int) * iBatchSize;
	checkCudaErrors(cudaMalloc((void**)&d_iTopkDocList, nbytes)  );
	checkCudaErrors(cudaMemset(d_iTopkDocList, -1, nbytes));

	nbytes = iTopk * sizeof(float) * iBatchSize;
	checkCudaErrors(cudaMalloc((void**)&d_dTopkScoreList, nbytes)  );
	checkCudaErrors(cudaMemset(d_dTopkScoreList, 0.0, nbytes));

//	int nbytes = iTopk * sizeof(int) * iBatchSize;
//	checkCudaErrors(cudaHostAlloc((void **) &m_iTopkDocList, nbytes,cudaHostAllocWriteCombined | cudaHostAllocMapped));
//	memset(m_iTopkDocList, -1, nbytes);
//
//	nbytes = iTopk * sizeof(float) * iBatchSize;
//	checkCudaErrors(cudaHostAlloc((void **) &m_dTopkScoreList, nbytes, cudaHostAllocWriteCombined | cudaHostAllocMapped));
//	memset(m_dTopkScoreList, 0.0, nbytes);
//
//	checkCudaErrors(cudaHostGetDevicePointer(&d_iTopkDocList, m_iTopkDocList, 0 ) );
//	checkCudaErrors(cudaHostGetDevicePointer(&d_dTopkScoreList,m_dTopkScoreList, 0 ) );

	#ifdef DEBUG
		printf("OK\n");
	#endif

	//-------------------------------------------------------------------------

	//-------------------------------------

//	cudaStream_t *streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * iQueryTotalNumberByBatch);


	//Processamento das Queries em Paralelo:
	//(1) Processar as 1000 queries em um único round (todas estarão no Buffer do acelerador)
	//1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
//	for (int idQuery = 0; idQuery < h_iQueryNumber; ++idQuery) {


//		checkCudaErrors( cudaStreamCreate(&streams[idQuery]) );

//		iMaxNumberInList = 0;
//		for (int idTerm = 0; idTerm < h_iTermNumberByQuery[idQuery]; idTerm++) {
//			if(iMaxNumberInList < h_iDocNumberByTermList[ m_iQueryBatches[ m_ptrQueryPositions[idQuery] +idTerm] ] )
//				iMaxNumberInList = h_iDocNumberByTermList[ m_iQueryBatches[ m_ptrQueryPositions[idQuery] +idTerm] ];
//		}

		//------------Kernel Launch Configuration---------------------------------
		#ifdef DEBUG
			printf("Kernel Launch Configuration... \n");
		#endif

		int iBlockNumber = iBatchSize;//(int) ceil((float)iMaxNumberInList/(DOC_QUANTITY_IN_MEMORY * iBlockNumberRound * iGlobalNumberRound));
		dim3 threadsByBlock = dim3(iBlockSize, 1);
		dim3 blocksByGrid   = dim3(iBlockNumber, 1);

		#ifdef DEBUG
			printf("#threads by block: %i, #blocks: %i ... OK\n",threadsByBlock.x, blocksByGrid.x) ;
		#endif
		//-------------------------------------------------------------------------


		//-----------Initializing the initial value of Threshold if it is necessary ---------------------
//		if(iQueryType){
//			setThresholdForANDQueryOnInvertedList(&dInitialThreshold, h_dUBList,
//												  h_iQueryBatches[idQuery], h_iTermNumberByQuery[idQuery]);
//		}
		//-----------------------------------------------------------------------------------------------

		//--------------------------Call to Kernels----------------------------
//		callToBatchKernels_byBlock(blocksByGrid, threadsByBlock, iMergeNumberByBlock, iTopk,
//								   dAverageDocumentLength, dInitialThreshold, iBlockNumberRound,
//								   iGlobalNumberRound, m_iTermNumberByQuery, d_dUBList, d_dIdfList,
//								   d_iDocIdList, d_iFreqDocList, d_iDocLenghtList,d_iTopkDocList,
//								   d_dTopkScoreList, d_iQueryBatches,d_ptrPostingPositions,
//								   d_ptrQueryPositions, d_iDocNumberByTermList);

		callToBatchKernels_byBlock(blocksByGrid, threadsByBlock, iMergeNumberByBlock, iTopk,
							  dAverageDocumentLength, dInitialThreshold,
							  m_iTermNumberByQuery, d_dUBList, d_dIdfList,
							  d_iDocIdList, d_iFreqDocList, d_iDocLenghtList,d_iTopkDocList,
							  d_dTopkScoreList, d_iQueryBatches,d_ptrInitPostingList,d_ptrQueryPositions,
							  d_iDocNumberByTermList);

//		dim3 blocksByGrid, dim3 threadsByBlock, int iMergeNumberByBlock, int iTopk,
//										float dAverageDocumentLength, float dInitialThreshold,
//										short int* iTermNumberInQuery,
//										float *d_dUBList, float *d_dIdfList, int *d_iDocIdList,
//										unsigned short int *d_iFreqDocList, int *d_iDocLenghtList,
//										int *d_iTopkDocList, float *d_dTopkScoreList,
//										int *iQueryTerms,long long* d_ptrPostingPositions,
//										int* d_ptrQueryPositions, int *d_iDocNumberByTermList)

		//----------------------------------------------------------------------------------------------------------
//	}


//   checkCudaErrors(cudaFreeHost(m_iTopkDocList));
//   checkCudaErrors(cudaFreeHost(m_dTopkScoreList));

//	//-----------------------Destroy Stream Objects------------------------------
//	for (int i = 0; i < h_iQueryNumber; ++i)
//		cudaStreamDestroy(streams[i]);
//	//---------------------------------------------------------------------------

	//-----------------------Free GPU Memory------------------------------
	checkCudaErrors(cudaFreeHost(m_iTermNumberByQuery));
	checkCudaErrors(cudaFreeHost(m_iQueryBatches));
	checkCudaErrors(cudaFreeHost(m_ptrQueryPositions));
	checkCudaErrors(cudaFreeHost(m_ptrInitPostingList));

	checkCudaErrors(cudaFree(d_iTopkDocList));
	checkCudaErrors(cudaFree(d_dTopkScoreList));

	freeAllocationOfInvertedListInGPU(d_iDocIdList, d_iDocLenghtList, d_iFreqDocList,
									  d_dUBList, d_dIdfList, d_iDocNumberByTermList);
	//----------------------------------------------------------------------------------------------------------

	//------------------Release resources in CPU-------------------------------
	#ifdef DEBUG
		printf("Releasing resources in CPU... ");
	#endif
	for (int i = 0; i < iTermNumberInVocabulary; ++i) {
		free(h_iDocIdList[i]);
		free(h_iDocLenghtList[i]);
		free(h_iFreqList[i]);
	}

	for (int i = 0; i < iQueryTotalNumberByBatch; ++i) {
		free(h_iQueryBatches[i]);
	}

//	free(streams);
	free(h_iQueryBatches);
	free(h_iDocIdList);
	free(h_iDocLenghtList);
	free(h_iFreqList);
	free(h_dIdfList);
	free(h_dUBList);
	free(h_iTermNumberByQuery);
	free(h_iDocNumberByTermList);

//	free(h_iTopkDocList);
//	free(h_dTopkScoreList);

	//-------------------------------------------------------------------------
	#ifdef DEBUG
		printf("OK!\n");
	#endif

	printf("Finish!\n");
}




