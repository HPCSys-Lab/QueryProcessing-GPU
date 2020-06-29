/*
 * UnityTest.c
 *
 *  Created on: 04/08/2018
 *      Author: roussian
 */


#include <stdio.h>
#include <stdlib.h>
#include "UnityTest.cuh"

__device__ float checkMinHeapProperty(documentTopkList heap, float newScore, int docCurrent,
									  int topk){

	int index = (topk - 1) - heap.padding - threadIdx.x;
	int parent = index;
	float result=0.0; //0 - Min heap is correct  1 - Min heap is not correct

	for (; index > 0; index-=blockDim.x ) {
		do{
			parent = (parent-1)/2;
			if(heap.score[index] < heap.score[parent]){
				result = heap.score[index];
				printf("[HEAP] position - %d | parent's position - %d | newScore - %.2f | docCurrent - %d | padding -%d \n", index, parent, newScore, docCurrent,heap.padding);
				break;
			}
		}while(parent > 0);

		if(result != 0.0)
			break;
	}

	if(result == 0.0){
		index = threadIdx.x;
		int max = (topk - 1) - heap.padding;
		for (; index < max; index+=blockDim.x ) {
			for (int i=index+1; i < max; i++ ) {
				if(heap.id[index] == heap.id[i]){
					printf("[HEAP] Duplicated document in heap %d | score %.2f | initial position %d | duplicated position %d | blockId %d | docCurrent %d\n",
										 heap.id[index], heap.score[index], index, i, blockIdx.x, docCurrent);
					result = heap.id[index];
				}
			}
		}
	}

	return result;
}

__device__ float checkSorting(documentTopkList heap, float *dTopkScoreListGlobal, int *iTopkDocListGlobal, int topk){

	int globalIndex = blockIdx.x * topk + heap.padding;
	int result=0;
	int index,maxIndex;

	if(THREAD_MASTER){
		if(heap.id[0] != iTopkDocListGlobal[globalIndex]){
			printf("[SORTING] First doc is not correct in topk: global index %d | blockIdx.x %d | doc %d | global doc %d !\n",
					blockIdx.x * topk, blockIdx.x,heap.id[0],iTopkDocListGlobal[globalIndex]);
			result = 1;
			return result;
		}
	}

	int gindex = blockIdx.x * topk + threadIdx.x;
	maxIndex = blockIdx.x * topk + heap.padding;
	for (int i = gindex; i < maxIndex; i+=blockDim.x) {
		if(iTopkDocListGlobal[i] != -1 ){
			printf("[SORTING] Error in padding: blockId %d | index %d | document in position %d\n", blockIdx.x, i,iTopkDocListGlobal[i]);
			result = iTopkDocListGlobal[i];
			return result;
		}
	}

    gindex = blockIdx.x * topk + heap.padding + threadIdx.x;
	maxIndex = (blockIdx.x+1) * topk;
	for (int i = gindex; i < maxIndex; i+=blockDim.x) {
		if(iTopkDocListGlobal[i] == -1 ){
			printf("[SORTING] Docs is empty: blockId %d | index %d | document in position %d\n", blockIdx.x, i,iTopkDocListGlobal[i]);
			result = iTopkDocListGlobal[i];
			return result;
		}
	}

	maxIndex = topk * (blockIdx.x + 1);
	index = topk * blockIdx.x + heap.padding + threadIdx.x;
	for (; index < maxIndex; index+=blockDim.x ) {
		for (int i=index+1; i < maxIndex; i++ ) {
			if(iTopkDocListGlobal[index] == iTopkDocListGlobal[i]){
				printf("[SORTING] Duplicate document in global topk list %d - score %.2f \n", iTopkDocListGlobal[index], dTopkScoreListGlobal[index]);
				result = iTopkDocListGlobal[index];
				return result;
			}
		}
	}

	index=threadIdx.x;
	globalIndex = blockIdx.x * topk;
	maxIndex = topk * (blockIdx.x + 1);
	for (; index < topk-heap.padding; index += blockDim.x) {
		int count = 0;
		for (int i = globalIndex; i < maxIndex; ++i) {
			if(heap.id[index] == iTopkDocListGlobal[i]){
				count++;
				if(heap.score[index] != dTopkScoreListGlobal[i]){
					printf("[SORTING] document's score is wrong! doc %d | score %.2f | local index %d | global index %d\n",
							heap.id[index], heap.score[index], index, i);
					result = heap.id[index];
					return result;
				}
				break;
			}
		}

		if(count == 0){
			printf("[SORTING] Document disappeared: doc %d | score %.2f | local index %d\n",
					heap.id[index], heap.score[index], index);
			result = heap.id[index];
			return result;
		}
	}

	index = blockIdx.x * topk + heap.padding + threadIdx.x;
	maxIndex = topk * (blockIdx.x + 1);
	for (; index < maxIndex; index+=blockDim.x ) {
		for (int i=index+1; i < maxIndex; i++ ) {
			if(dTopkScoreListGlobal[index] > dTopkScoreListGlobal[i]){
				printf("[SORTING] BlockIdx %d | Documents are not sorting!!! doc %d (%.2f) is greater than doc %d (%.2f)\n",
						blockIdx.x,iTopkDocListGlobal[index],dTopkScoreListGlobal[index],iTopkDocListGlobal[i],dTopkScoreListGlobal[i]);
				result = iTopkDocListGlobal[index] ;
				return result;
			}
		}
	}

	return result;
}

__device__ float checkMerge_Sorting_Documents(documentTopkList sortingList,int iSkipMerges,
	  	  	  	  	  	  	  	  	  	  	  int iSkipBlocks, int topk){

	float result = 0.0;
	float score;

	for (int i = threadIdx.x; i < topk; i+=blockDim.x) {
		int doc = sortingList.id[i];

		if((doc != -1 && sortingList.score[i] == 0.0) || (doc == -1 && sortingList.score[i] != 0.0) )
			printf("[MERGE] BlockId %d | SkipBlocks %d | SkipMerges %d | Document is inconsistent: doc %d (%.2f - %d)",iSkipBlocks,
					blockIdx.x, iSkipMerges, doc, sortingList.score[i], i);

		if(doc != -1)
			for (int j = i+1; j < topk; ++j) {
				if(sortingList.id[j] != -1 && sortingList.id[j] == doc){
					printf("[MERGE] BlockId %d | SkipBlocks %d | SkipMerges %d | Duplicated Document: doc %d (%.2f - %d) - doc %d (%.2f - %d)\n",
							blockIdx.x, iSkipBlocks, iSkipMerges, sortingList.id[i], sortingList.score[i], i, sortingList.id[j], sortingList.score[j], j);
					result = sortingList.id[i];
					return result;
				}
			}
	}

	for (int i = threadIdx.x; i < topk; i+=blockDim.x) {
		score = sortingList.score[i];
		if(score != 0.0)
			for (int j = i+1; j < topk; ++j) {
				if(sortingList.score[j] != 0.0 && score > sortingList.score[j]){
					printf("[MERGE] BlockId %d | SkipBlocks %d | SkipMerges %d | Documents are not sorting!!! doc %d (%.2f - %d) is greater than doc %d (%.2f - %d)\n",
							blockIdx.x, iSkipBlocks, iSkipMerges, sortingList.id[i], sortingList.score[i], i, sortingList.id[j], sortingList.score[j], j);
					result = sortingList.id[i];
					return result;
				}
			}
	}

	return result;
}
