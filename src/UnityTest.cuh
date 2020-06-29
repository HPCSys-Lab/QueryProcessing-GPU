/*
 * UnityTest.cuh
 *
 *  Created on: 04/08/2018
 *      Author: roussian
 */

#ifndef UNITYTEST_CUH_
#define UNITYTEST_CUH_
#include "Structs.cuh"

__device__ float checkMinHeapProperty(documentTopkList heap, float newScore, int docCurrent, int topk);

__device__ float checkSorting(documentTopkList heap, float *dTopkScoreListGlobal, int *iTopkDocListGlobal, int topk);

__device__ float checkMerge_Sorting_Documents(documentTopkList sortingList,int iSkipTopkBetweenMerges,
	  	  	  	  	  int iSkipTopkBetweenBlocks, int topk);

#endif /* UNITYTEST_CUH_ */
