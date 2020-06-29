/*
 * DeviceFunctions.cu
 *
 *  Created on: 08/12/2017
 *      Author: roussian
 */
#include "DeviceFunctions.cuh"
#include "Structs.cuh"
#include <stdio.h>


 __device__  float scoreTf_Idf(int tf, int dDocLength, float idf,
							  float averageDocumentLength, float keyFrequency){
	float k_1 = 1.2;
	float b = 0.75;
	float robertsonTf = (k_1 * tf) / ( tf + (k_1 * ((1 - b) + (b * dDocLength) / averageDocumentLength)));

	return keyFrequency * robertsonTf * idf;
}


__device__  float atomicAddD(double* address, double val){

	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN !=NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}


__device__  double atomicExchD(double volatile *address, double volatile val){

	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;

		old = atomicExch(address_as_ull, __double_as_longlong(val));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN !=NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}


__device__  double atomicMaxD(double volatile *address, double volatile val){

	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;

		old = atomicMax(address_as_ull, __double_as_longlong(val));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN !=NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}


 __device__  void sortingTerms_2(finger* fingers,
								unsigned int *iOrderedTermListShared,
								const short int iTermNumber){

		if(threadIdx.x < iTermNumber){
			int ownValue = fingers[threadIdx.x].docId;
			int otherThreadValue = 0;
			int position = 0;
			for (int round = 0; round < iTermNumber; ++round) {
				otherThreadValue = __shfl_sync(0xFFFFFFFF,ownValue, round);
				if((otherThreadValue < ownValue) ||
						(otherThreadValue == ownValue && round > threadIdx.x)){
					position++;
				}
			}
			iOrderedTermListShared[position] = threadIdx.x;//O id da thread representa o termo que será apontado.
		}
}

__device__  void selectTermPivot_No_SharedMemory(pivot* pivot,
								   unsigned int *iOrderedTermListShared,
								   finger* fingers,
								   const float *dUBlist,
								   int iTermNumber,
								   float threshold){
	float sumUB = 0.0;
	int iPivotPosition = 0;

	do {
		sumUB += dUBlist[iOrderedTermListShared[iPivotPosition]];
		iPivotPosition++;
	} while ( (iPivotPosition < iTermNumber)
				&& (sumUB < threshold)
				&& (fingers[iOrderedTermListShared[iPivotPosition]].docId != NO_MORE_DOC));
	iPivotPosition--;

// 	if(fingers[0].position == NO_VALID_POSITION && fingers[1].position == NO_VALID_POSITION)
// 		printf("Oi");

	if( (sumUB < threshold) || iPivotPosition >= iTermNumber || (fingers[iOrderedTermListShared[iPivotPosition]].docId == NO_MORE_DOC)){
		pivot->positionInOrderedList = NO_PIVOT_TERM;
		pivot->idTerm = NO_PIVOT_TERM;
// 		if(sumUB > threshold)
// 			printf("--%d---",blockIdx.x);
	}
	else{
// 	 	if(fingers[iOrderedTermListShared[iPivotPosition]].position >= DOC_QUANTITY_IN_MEMORY)
// 	 		printf("Eita");

		pivot->positionInOrderedList = iPivotPosition;
		pivot->idTerm = iOrderedTermListShared[iPivotPosition];
	}
}

__device__  void selectTermPivot_2(pivot* pivot,
 								   unsigned int *iOrderedTermListShared,
 								   finger* fingers,
 								   const float *dUBlist,
 								   int iTermNumber,
 								   float threshold){
 	float sumUB = 0.0;
 	int iPivotPosition = 0;

 	do {
 		sumUB += dUBlist[iOrderedTermListShared[iPivotPosition]];
 		iPivotPosition++;
 	} while ( (iPivotPosition < iTermNumber) && (sumUB < threshold) && (fingers[iOrderedTermListShared[iPivotPosition]].docId != NO_MORE_DOC));
		iPivotPosition--;
// 	if(fingers[0].position == NO_VALID_POSITION && fingers[1].position == NO_VALID_POSITION)
// 		printf("Oi");

 	if( (sumUB < threshold) || iPivotPosition >= iTermNumber || (fingers[iOrderedTermListShared[iPivotPosition]].position >= DOC_QUANTITY_IN_MEMORY)
 			|| (fingers[iOrderedTermListShared[iPivotPosition]].docId == NO_MORE_DOC)){
 		pivot->positionInOrderedList = NO_PIVOT_TERM;
 		pivot->idTerm = NO_PIVOT_TERM;
// 		if(sumUB > threshold)
// 			printf("--%d---",blockIdx.x);
 	}
 	else{
// 	 	if(fingers[iOrderedTermListShared[iPivotPosition]].position >= DOC_QUANTITY_IN_MEMORY)
// 	 		printf("Eita");

 		pivot->positionInOrderedList = iPivotPosition;
 		pivot->idTerm = iOrderedTermListShared[iPivotPosition];
 	}
}

  __device__ void advancePivoTermFinger_2(pivot pivot,
 									    finger *fingers,
 									    postingList *postingLists){

 	int docPivot =  fingers[pivot.idTerm].docId;

 	if(fingers[threadIdx.x].docId ==  docPivot){

 		fingers[threadIdx.x].position++;

 		if(fingers[threadIdx.x].position >= postingLists[threadIdx.x].maxIndex ||
 				fingers[threadIdx.x].position >= NO_VALID_POSITION){//Não Válido
 			fingers[threadIdx.x].docId = NO_MORE_DOC;
 			fingers[threadIdx.x].position = NO_VALID_POSITION;
 		}else{
 			fingers[threadIdx.x].docId = postingLists[threadIdx.x].docId[fingers[threadIdx.x].position];
 		}
 	}
}

/*
 * Avança em paralelo os ponteiros das listas invertidas anterior ao termo pivô;
 * Cada thread obtem o seu elemento de acordo com a posição do atual docId
 * e verifica se o elemento que lhe pertence é o primeiro maior docId.
 */
__device__ void advanceDocIdOfPredecessorTerm_3(postingList *postingList,
											    unsigned int *iOrderedTermListShared,
											    finger *fingers, pivot pivot,
											    const int iBlockRoundNumber){
	int threadPosition;
	int idTerm;

	int docId;
	int docFingerOld;
//	int warpNumber = blockDim.x >> 5;

	int warpId = threadIdx.x >> 5;
	int idThreadInWarp = threadIdx.x - (warpId << 5);

	while(warpId < pivot.positionInOrderedList && warpId < TERM_NUMBER){

		idTerm = iOrderedTermListShared[warpId];
		docFingerOld = fingers[idTerm].docId;

		if(fingers[idTerm].docId == fingers[pivot.idTerm].docId){
//			if(warpId == 0)//Se entrar, está errado algo!
//				printf("Erro: Primeiro elemento igual ao pivo idTerm %d doc %d\n",idTerm,fingers[pivot.idTerm].docId);
			break;
		}

		threadPosition = idThreadInWarp + fingers[idTerm].position + 1;
		docId = 0;//(iPositionThread < iBlockRoundNumber * blockDim.x) ? postingList[iTermId].docId[iPositionThread] : 0;

		while(threadPosition < postingList[warpId].maxIndex
				&& docId < fingers[pivot.idTerm].docId){

			docId = postingList[idTerm].docId[threadPosition];
			threadPosition += warpSize;
		}
		threadPosition -= warpSize;

		if((docId >= fingers[pivot.idTerm].docId)
					&& (postingList[idTerm].docId[threadPosition - 1] < fingers[pivot.idTerm].docId)){
			fingers[idTerm].docId = docId;
			fingers[idTerm].position = threadPosition;
		}

		if(docFingerOld == fingers[idTerm].docId && ((threadIdx.x & 0x1f) == 0) ){
			fingers[idTerm].docId = NO_MORE_DOC;
			fingers[idTerm].position = NO_VALID_POSITION;
		}
		warpId += (blockDim.x >> 5);
	}
}

 __device__ void advancePivoTermFinger_4(pivot pivot,
 									    finger *fingers,
 									    postingList *postingLists,
 									    unsigned short int iElementQuantityByBlock,
 									    int threadIdInWarp){

 	int docPivot =  fingers[pivot.idTerm].docId;

 	if(fingers[threadIdInWarp].docId ==  docPivot){

 		fingers[threadIdInWarp].position++;

 		if(fingers[threadIdInWarp].position >= iElementQuantityByBlock){//Não Válido
 			fingers[threadIdInWarp].docId = NO_MORE_DOC;
 			fingers[threadIdInWarp].position = NO_VALID_POSITION;
 		}else{
 			fingers[threadIdInWarp].docId = postingLists[threadIdInWarp].docId[fingers[threadIdInWarp].position];

 			if(fingers[threadIdInWarp].docId == NO_MORE_DOC)
 				fingers[threadIdInWarp].position = NO_VALID_POSITION;
 		}
 	}
 }

  __device__ void advanceDocIdOfPredecessorTerm_4(postingList *postingList,
 											    unsigned int *iOrderedTermListShared,
 											    finger *fingers, pivot pivot, int docPivot,
 											    const unsigned short int iElementQuantityByBlock){
 	int threadPosition;
 	int idTerm;

 	int docId;
 	int docFingerOld;

 	int warpId = threadIdx.x >> 5;
 	int idThreadInWarp = threadIdx.x - (warpId << 5);

 	//Race
 	int positionFinger;

 	while(warpId < pivot.positionInOrderedList && warpId < TERM_NUMBER){

 		idTerm = iOrderedTermListShared[warpId];
 		positionFinger = fingers[idTerm].position;
 		docFingerOld = fingers[idTerm].docId;

 		if(fingers[idTerm].docId == docPivot){
 			break;
 		}

 		threadPosition = idThreadInWarp + positionFinger + 1;
 		docId = 0;//(iPositionThread < iBlockRoundNumber * blockDim.x) ? postingList[iTermId].docId[iPositionThread] : 0;

 		while(threadPosition < iElementQuantityByBlock
 				&& docId < docPivot){

 			docId = postingList[idTerm].docId[threadPosition];
 			threadPosition += warpSize;
 		}
 		threadPosition -= warpSize;

 		if((docId >= docPivot)
 					&& (postingList[idTerm].docId[threadPosition - 1] < docPivot)){
 			fingers[idTerm].docId = docId;
 			fingers[idTerm].position = threadPosition;
 		}

 		warpId += (blockDim.x >> 5);
 	}

 	__syncthreads();


 	if( ((threadIdx.x & 0x1f) == 0) && (threadIdx.x >> 5) < pivot.positionInOrderedList){

 		idTerm = iOrderedTermListShared[threadIdx.x >> 5];
 		if((threadIdx.x >> 5) < pivot.positionInOrderedList &&
 				docFingerOld == fingers[idTerm].docId ){

 			fingers[idTerm].docId = NO_MORE_DOC;
 			fingers[idTerm].position = NO_VALID_POSITION;
 		}
 	}
}

__device__ void fullScore_3(float *score, pivot pivot,
							unsigned int *iOrderedTermSharedList,
							finger *fingers,
							postingList *postingList,
							const float *dIdfList,
							const float dAverageDocLength){

	if(threadIdx.x < TERM_NUMBER){
		int termId = iOrderedTermSharedList[threadIdx.x];
		float scoreL = 0.0;

		if(fingers[termId].docId == fingers[pivot.idTerm].docId){

			scoreL = scoreTf_Idf(postingList[termId].freq[fingers[termId].position],
								postingList[termId].docLenght[fingers[termId].position],
								dIdfList[termId],dAverageDocLength,1.1);

		}

		float aux = 0;
		#pragma unroll 2
		for (int i = 0; i < TERM_NUMBER; ++i) {
			aux += __shfl_sync(0xFFFFFFFF,scoreL,i);
		}

		if(THREAD_MASTER)
			*score = aux;
	}
	__syncthreads();
}

 __device__ void fullScore_3_1(float *score, int docPivot,
 						  	 unsigned int *iOrderedTermSharedList,
 						  	 finger *fingers,
 						  	 postingList *postingList,
 						  	 const float *dIdfList,
 						  	 const float dAverageDocLength){

 		int termId = iOrderedTermSharedList[threadIdx.x];
 		float scoreL = 0.0;

 		if(fingers[termId].docId == docPivot){

 			scoreL = scoreTf_Idf(postingList[termId].freq[fingers[termId].position],
 								postingList[termId].docLenght[fingers[termId].position],
 								dIdfList[termId],dAverageDocLength,1.1);

 		}

 		float aux = 0;
 		for (int i = 0; i < TERM_NUMBER; ++i) {
 			aux += __shfl_sync(0xFFFFFFFF,scoreL,i);
 		}

 		if(THREAD_MASTER)
 			*score = aux;
}


 __device__ void insertValueInEmptyList_2(documentTopkList *documentTokList,
		 	 	 	 	 	 	 	 	 int newDocId, float newScore){
	documentTokList->id[0] = newDocId;
	documentTokList->score[0] = newScore;
	documentTokList->padding--;
}

 __device__  void insertValueInFullList_2(documentTopkList *topkList, int newDocId, float newScore){
//												documentTopkList* documentTemp){

	int position = (threadIdx.x < TOP_K) ? threadIdx.x : NO_VALID_POSITION;
	int nextPosition = position + 1;

	float nextScore, ownScore;
	int nextDocId;

//	#pragma unroll 4
	for (int i = 0; i < TOP_K/blockDim.x; ++i) {
		ownScore = (position == NO_VALID_POSITION) ? 0.0 : topkList->score[position];
		if(position != NO_VALID_POSITION){
			nextScore = (nextPosition < TOP_K) ? topkList->score[nextPosition] : ownScore;
			nextDocId = (nextPosition < TOP_K) ? topkList->id[nextPosition] : topkList->id[position];
		}

//		__syncthreads();

		if( (ownScore <= newScore) || ( (ownScore >= newScore) && (nextScore == 0.0)) ){
			if(nextScore <= newScore && nextPosition != TOP_K){
				topkList->score[position] = nextScore;
				topkList->id[position]  = nextDocId;
			}else{
				topkList->score[position] = newScore;
				topkList->id[position]  = newDocId;
			}
		}
		position += blockDim.x;
		nextPosition += blockDim.x;
	}
}


 __device__ void insertValueInNotFullList_2(documentTopkList *topkList,
													int newDocId, float newScore){
	int position = (threadIdx.x < TOP_K) ? (TOP_K - threadIdx.x - 1) : NO_VALID_POSITION;
	int previousPosition = (position == 0) ? position : position - 1;

	float previousScore, ownScore;
	int previousDocId;

//	if(THREAD_MASTER && topkList->score[1] >  newScore)
//		topkList->score[0] = newScore;

	#pragma unroll 4
	for (int i = 0; i < TOP_K/blockDim.x; ++i) {

		ownScore = (position == NO_VALID_POSITION) ? 0.0 : topkList->score[position];
		if(position != NO_VALID_POSITION){
			previousScore = (previousPosition >= 0) ? topkList->score[previousPosition] : ownScore;
			previousDocId = (previousPosition >= 0) ? topkList->id[previousPosition] : topkList->id[position];
		}

//		__syncthreads();

		if(ownScore >= newScore || (previousScore != 0.0 && ownScore == 0.0)){
			if(previousScore < newScore || position == 0){
				topkList->score[position] = newScore;
				topkList->id[position]  = newDocId;
			}else{
				topkList->score[position] = previousScore;
				topkList->id[position]  = previousDocId;
			}
		}

		position -= blockDim.x;
		previousPosition -= blockDim.x;
	}

	if(THREAD_MASTER)
		(topkList->padding)--;

//	__syncthreads();
}

 __device__ void insertValueInNotFullList_Heap(documentTopkList *topkList,
													int newDocId, float newScore){


	int insertPosition = TOP_K - topkList->padding;

	if(THREAD_MASTER)
		(topkList->padding)--;

//	if(insertPosition <= 2){//Inserção básica: os filhos da raiz não estão preenchidos
//		topkList->id[insertPosition] = newDocId;
//		topkList->score[insertPosition] = newScore;
//		return;
//	}

	int height =  log2f(insertPosition+1);

	//Uma thread para cada ancestral do nó positionInsert, com exceção da raiz
	if(threadIdx.x <= height){ //&& threadIdx.x != 0){
		int parentPosition,initialPosition = insertPosition;
		int elementNumberInLevel = (1 << height);

		int doc;
		float score;

		insertPosition = ( insertPosition - (elementNumberInLevel - 1) ) / ( elementNumberInLevel >> threadIdx.x );
		insertPosition += (1 << threadIdx.x) - 1;

		if(topkList->score[insertPosition] > newScore || initialPosition == insertPosition){

			parentPosition = (insertPosition - 1) >> 1;
			if(topkList->score[parentPosition] > newScore){
				doc = topkList->id[parentPosition];
				score = topkList->score[parentPosition];
			}else{//No level 1, sempre o pai é menor, pois é a raiz
				doc = newDocId;
				score = newScore;
			}

			topkList->id[insertPosition] = doc;
			topkList->score[insertPosition] =  score;
		}
	}
//	__syncthreads();
}


__device__ inline float insertValueInFullList_Heap(documentTopkList *topkList,
													int newDocId, float newScore){

//	if(blockIdx.x != 0) return 2;

	int height = HEIGHT_HEAP;
	//3º Passo = Se a propriedade do heap-min for desfeito, encontrar o caminho para inserir o novo elemento
	if(threadIdx.x <= height){

		//1º Passo = Extrai o menor valor (raiz) e aumenta a chave (substitui a raiz com o novo valor)
		topkList->id[0]    = newDocId;
		topkList->score[0] = newScore;

		//2º Passo = Identificar a raiz do subHeap que será alterado
		int subHeapRootIndex = 0;// = (topkList->score[1] < topkList->score[2]) ? 1 : 2;

//		if(topkList->score[subHeapRootIndex] < newScore){
			int level = 0;
			int doc;
			float score;

			int positionChild;

//			if(threadIdx.x == 0) subHeapRootIndex = 0;

			while (level < threadIdx.x  && topkList->score[subHeapRootIndex] <= newScore){
				subHeapRootIndex <<= 1;
				subHeapRootIndex++;
				if(subHeapRootIndex + 1 < TOP_K )
					if(topkList->score[subHeapRootIndex + 1] <= topkList->score[subHeapRootIndex])//Obtém o index do filho que tem o menor  valor
						subHeapRootIndex++;
				level++;
			}

			if(subHeapRootIndex >= TOP_K || topkList->score[subHeapRootIndex] > newScore){
				level--;
				subHeapRootIndex = -1;
			}

//			while(level < threadIdx.x && topkList->score[subHeapRootIndex] <= newScore){
//				subHeapRootIndex <<= 1;
//				subHeapRootIndex++;
////				if(subHeapRootIndex + 1 < TOP_K )
//					if(topkList->score[subHeapRootIndex + 1] <= topkList->score[subHeapRootIndex])//Obtém o index do filho que tem o menor  valor
//						subHeapRootIndex++;
//
//				if(topkList->score[subHeapRootIndex] < newScore)
//					level++;
//			}

//			if(level != threadIdx.x)
//				subHeapRootIndex = -1;

			positionChild = __shfl_down_sync(0xFFFFFFFF,subHeapRootIndex, 1);

			if(threadIdx.x ==  height)  //thread que está na folha
				positionChild = -1;

			if(level == threadIdx.x){

				if(positionChild != -1){// && ((threadIdx.x <  height)) {
					doc = topkList->id[positionChild];
					score = topkList->score[positionChild];
				}
				else{
					doc = newDocId;
					score = newScore;
				}

//				__syncwarp();

				topkList->id[subHeapRootIndex] = doc;
				topkList->score[subHeapRootIndex] = score;
			}
//		}
	}
//	else{
//		int subHeapRootIndex = (topkList->score[1] < topkList->score[2]) ? 1 : 2;
//
//		if(topkList->score[subHeapRootIndex] < topkList->score[0])
//			return topkList->score[subHeapRootIndex];
//	}

//	__syncthreads();
	return topkList->score[0];
}

 __device__ float managerMinValue_v4(documentTopkList *documentTopk,int newDocId, float newScore, int padding){

	if(padding == 0){
		insertValueInFullList_2(documentTopk,newDocId,newScore);
	}else if(padding == TOP_K){
		if(THREAD_MASTER) insertValueInEmptyList_2(documentTopk,newDocId, newScore);
	}else{
		insertValueInNotFullList_2(documentTopk, newDocId, newScore);
	}

	__syncthreads();

	return documentTopk->score[0];;
}


 __device__ float managerMinValue_v5(documentTopkList *documentTopk, int newDocId, float newScore, int padding){

	float min;

	if(padding == 0){
		insertValueInFullList_Heap(documentTopk,newDocId,newScore);
	}else if(padding == TOP_K){
		if(THREAD_MASTER) insertValueInEmptyList_2(documentTopk,newDocId, newScore);
		min = newScore;
	}else{
		insertValueInNotFullList_Heap(documentTopk, newDocId, newScore);
		min = documentTopk->score[0];
	}

	min = documentTopk->score[0];
	return min;
}

 __device__ void searchRangeOfDocs(const int* iDocIdList,
		  	  	  	  	  	  	  postingList *postingLists, int termId,
		  	  	  	  	  	  	  int iGlobalInitialPositionInList, limitDocId *limitDoc,
		  	  	  	  	  	  	  unsigned short int iElementQuantityByBlock,
		  	  	  	  	  	  	  int roundGlobalNumber,const int *iDocNumberByTermList){



	//	__shared__ int newPosition;
		int positionInListGlobal = 0;
		int globalIndex = iGlobalInitialPositionInList + threadIdx.x;

		if(THREAD_MASTER)//Race
		limitDoc->extraPosition[termId] = NO_MORE_DOC;

		__syncthreads();

		#pragma unroll 2
		for (int i = 0; i < termId; ++i) {
			positionInListGlobal += iDocNumberByTermList[i];
		}

		int docLocal = -1;
		while (docLocal < limitDoc->minDocId && globalIndex < iDocNumberByTermList[termId]){
			docLocal = iDocIdList[positionInListGlobal + globalIndex];
			globalIndex += blockDim.x;
		}
		globalIndex-= blockDim.x;
		long long int initialPosition;

		if(docLocal < limitDoc->minDocId)//Caso não encontre
			initialPosition = NO_VALID_POSITION;
		else
			initialPosition = globalIndex - iGlobalInitialPositionInList;

		int positionNeighbor;
		for (int i = 16; i >= 1; i /= 2) {
			positionNeighbor  = __shfl_down_sync(0xFFFFFFFF,initialPosition, i);

			if(positionNeighbor < initialPosition)
				initialPosition = positionNeighbor;
		}

	//	__syncthreads();

		if( ((threadIdx.x & 0x1f) == 0) && initialPosition != NO_MORE_DOC){
			atomicMin(&(limitDoc->extraPosition[termId]) , initialPosition);
		}
		__syncthreads();

		if( THREAD_MASTER && limitDoc->extraPosition[termId] == NO_MORE_DOC){
			limitDoc->extraPosition[termId] = 0;
		}

		if(THREAD_MASTER){
			globalIndex = iGlobalInitialPositionInList + iElementQuantityByBlock * roundGlobalNumber + threadIdx.x;
			globalIndex += limitDoc->extraPosition[termId];
	//
			if(globalIndex < iDocNumberByTermList[termId]){
				if(limitDoc->secondMaxDocId < iDocIdList[positionInListGlobal + globalIndex] -1)
					limitDoc->secondMaxDocId = iDocIdList[positionInListGlobal + globalIndex]-1;
			}
		}
	//
	//	if(THREAD_MASTER) limitDoc->extraPosition[termId] = newPosition;
}


__device__ void searchRangeOfDocs_batch(const int* iDocIdList,
		  	  	  	  	  	  	  	  	postingList *postingLists, int termId,
		  	  	  	  	  	  	  	  	int iGlobalInitialPositionInList, limitDocId *limitDoc,
		  	  	  	  	  	  	  	  	unsigned short int iElementQuantityByBlock,
		  	  	  	  	  	  	  	  	int roundGlobalNumber,int iPostingListSize,
		  	  	  	  	  	  	  	  	long long positionInListGlobal){

//	__shared__ int newPosition;
//		long long positionInListGlobal = ptrPostingPositions[termId];
	int globalIndex = iGlobalInitialPositionInList + threadIdx.x;

//	if(((threadIdx.x & 0x1f) == 0) && limitDoc->extraPosition[termId] == 0)//if(THREAD_MASTER)//Race
//		limitDoc->extraPosition[termId] = NO_MORE_DOC;

//	__syncthreads();

//		for (int i = 0; i < termId; ++i) {
//			positionInListGlobal += iDocNumberByTermList[i];
//		}

	int docLocal = -1;
	while (docLocal < limitDoc->minDocId && globalIndex < iPostingListSize){
		docLocal = iDocIdList[positionInListGlobal + globalIndex];
		globalIndex += blockDim.x;
	}
	globalIndex -= blockDim.x;

	long long int initialPosition;
	if(docLocal < limitDoc->minDocId)//Caso não encontre
		initialPosition = NO_VALID_POSITION;
	else
		initialPosition = globalIndex - iGlobalInitialPositionInList;

	int positionNeighbor;
	for (int i = 16; i >= 1; i /= 2) {
		positionNeighbor  = __shfl_down_sync(0xFFFFFFFF,initialPosition, i);

		if(positionNeighbor < initialPosition)
			initialPosition = positionNeighbor;
	}

//	__syncthreads();

	if( ((threadIdx.x & 0x1f) == 0) && initialPosition != NO_VALID_POSITION){
//		atomicMax(&(limitDoc->extraPosition[termId]) , initialPosition);
		atomicMin(&(limitDoc->extraPosition[termId]) , initialPosition);
	}

	__syncthreads();

	if( THREAD_MASTER ){//1066

		if (limitDoc->extraPosition[termId] == NO_MORE_DOC)
			limitDoc->extraPosition[termId] = 0;

		globalIndex = iGlobalInitialPositionInList + iElementQuantityByBlock * roundGlobalNumber + threadIdx.x;
		globalIndex += limitDoc->extraPosition[termId];

		if(globalIndex < iPostingListSize){
			if(limitDoc->secondMaxDocId < iDocIdList[positionInListGlobal + globalIndex] - 1)
				limitDoc->secondMaxDocId = iDocIdList[positionInListGlobal + globalIndex] - 1;
		}
	}

//	if(THREAD_MASTER){
//		globalIndex = iGlobalInitialPositionInList + iElementQuantityByBlock * roundGlobalNumber + threadIdx.x;
//		globalIndex += limitDoc->extraPosition[termId];
//
//		if(globalIndex < iPostingListSize){
//			if(limitDoc->secondMaxDocId < iDocIdList[positionInListGlobal + globalIndex] -1)
//				limitDoc->secondMaxDocId = iDocIdList[positionInListGlobal + globalIndex]-1;
//		}
//	}
//
//	if(THREAD_MASTER) limitDoc->extraPosition[termId] = newPosition;
}

 __device__ void searchMoreDocs(const int* iDocIdList,
							   const unsigned short int* iFreqList,  const int *iDocLengthList,
		  	  	  	  	  	   postingList *postingLists, int termId,
		  	  	  	  	  	   int iGlobalInitialPositionInList, limitDocId *limitDoc,
		  	  	  	  	  	   unsigned short int iElementQuantityByBlock,
		  	  	  	  	  	   finger *fingerT, int docCurrent,const int *iDocNumberByTermList){

	int globalIndex = iGlobalInitialPositionInList + limitDoc->extraPosition[termId] + iElementQuantityByBlock + threadIdx.x;
	int docLocal, positionInListGlobal=0;
	int localIndex;
	int docAmount = iDocNumberByTermList[termId];

	__shared__ int lowerAmount;
	lowerAmount =0;
//	#pragma unroll 2
	for (int i = 0; i < termId; ++i) {
		positionInListGlobal += iDocNumberByTermList[i];
	}
	docLocal = (globalIndex < docAmount) ? iDocIdList[positionInListGlobal + globalIndex] : NO_MORE_DOC;

//	if(blockIdx.x == 1230 && THREAD_MASTER)
//			printf("Atualizando Termo %d \n", termId);
//
//	if(docLocal < docCurrent){
//		if(THREAD_FINAL || ( (globalIndex + 1) == docAmount) || iDocIdList[positionInListGlobal + globalIndex + 1] > docCurrent){
//			lowerAmount = threadIdx.x;
//		}
//	}
//
	int isLowerLocal = (docLocal < docCurrent);
	#pragma unroll 16
	for (int i = 16; i >= 1; i /= 2) {
		isLowerLocal += __shfl_down_sync(0xFFFFFFFF,isLowerLocal, i);
	}

	if( ((threadIdx.x & 0x1f) == 0) && isLowerLocal != 0){
		atomicAdd(&lowerAmount, isLowerLocal);
	}

	__syncthreads();
//
//	isLowerLocal = lowerAmount;
	globalIndex += lowerAmount;
	for (localIndex = threadIdx.x; localIndex < iElementQuantityByBlock; localIndex += blockDim.x) {

		docLocal = (globalIndex < docAmount) ? iDocIdList[positionInListGlobal + globalIndex] : NO_MORE_DOC;

		if(docLocal > limitDoc->secondMaxDocId){
				postingLists[termId].docId[localIndex] = NO_MORE_DOC;
				fingerT->final = 1;
				break;
		}

		postingLists[termId].docId[localIndex] = docLocal;
		postingLists[termId].docLenght[localIndex] = iDocLengthList[positionInListGlobal + globalIndex];
		postingLists[termId].freq[localIndex] = iFreqList[positionInListGlobal + globalIndex];

		globalIndex += blockDim.x;
	}

//	__syncthreads();

	if(THREAD_MASTER){
		fingerT->docId = postingLists[termId].docId[0];
		fingerT->position = (fingerT->docId == NO_MORE_DOC) ? NO_VALID_POSITION : 0;
		if (fingerT->docId == NO_MORE_DOC) fingerT->position = 1;
		limitDoc->extraPosition[termId] += iElementQuantityByBlock + lowerAmount;//isLowerLocal + iElementQuantityByBlock;
	}
}


__device__ void searchMoreDocs_batch(const int* iDocIdList, const unsigned short int* iFreqList,
									 const int *iDocLengthList, postingList *postingLists, int termId,
									 int iGlobalInitialPositionInList, limitDocId *limitDoc,
									 unsigned short int iElementQuantityByBlock, finger *fingerT,
									 int docCurrent, int iPostingListSize, long long positionInGlobalList){

	int globalIndex = iGlobalInitialPositionInList + limitDoc->extraPosition[termId] + iElementQuantityByBlock + threadIdx.x;
	int docLocal;//, positionInListGlobal=0;
	int localIndex;
//	int docAmount = iPostingListSize;

	__shared__ int lowerAmount;
	lowerAmount =0;

//	for (int i = 0; i < termId; ++i) {
//		positionInListGlobal += iDocNumberByTermList[i];
//	}
	docLocal = (globalIndex < iPostingListSize) ? iDocIdList[positionInGlobalList + globalIndex] : NO_MORE_DOC;

//	if(blockIdx.x == 1230 && THREAD_MASTER)
//			printf("Atualizando Termo %d \n", termId);
//
//	if(docLocal < docCurrent){
//		if(THREAD_FINAL || ( (globalIndex + 1) == docAmount) || iDocIdList[positionInListGlobal + globalIndex + 1] > docCurrent){
//			lowerAmount = threadIdx.x;
//		}
//	}
//
	int isLowerLocal = (docLocal < docCurrent);
	#pragma unroll 16
	for (int i = 16; i >= 1; i /= 2) {
		isLowerLocal += __shfl_down_sync(0xFFFFFFFF,isLowerLocal, i);
	}

	if( ((threadIdx.x & 0x1f) == 0) && isLowerLocal != 0){
		atomicAdd(&lowerAmount, isLowerLocal);
	}

	__syncthreads();

//	isLowerLocal = lowerAmount;
	globalIndex += lowerAmount;
	for (localIndex = threadIdx.x; localIndex < iElementQuantityByBlock; localIndex += blockDim.x) {

		docLocal = (globalIndex < iPostingListSize) ? iDocIdList[positionInGlobalList + globalIndex] : NO_MORE_DOC;

		if(docLocal > limitDoc->secondMaxDocId){
			postingLists[termId].docId[localIndex] = NO_MORE_DOC;
			fingerT->final = 1;
			break;
		}

		postingLists[termId].docId[localIndex] = docLocal;
		postingLists[termId].docLenght[localIndex] = iDocLengthList[positionInGlobalList + globalIndex];
		postingLists[termId].freq[localIndex] = iFreqList[positionInGlobalList + globalIndex];

		globalIndex += blockDim.x;
	}

//	__syncthreads();

	if(THREAD_MASTER){
		fingerT->docId = postingLists[termId].docId[0];
		fingerT->position = (fingerT->docId == NO_MORE_DOC) ? NO_VALID_POSITION : 0;
		if (fingerT->docId == NO_MORE_DOC) fingerT->position = 1;
		limitDoc->extraPosition[termId] += iElementQuantityByBlock + lowerAmount;//isLowerLocal + iElementQuantityByBlock;
	}
}


__device__ void sortLocalTopkDocAndStoreInGlobal_BLOCK( float *dTopkScoreListGlobal,int *iTopkDocListGlobal, int iTopk,
												documentTopkList *localTopkDoc){

//	if(blockIdx.x == 4999 && threadIdx.x == 32)
//		printf("okay");

	float score_1, scoreAux, scoreNeighborAux; //score_3
	int position_1;//, position_2;//position_3;
	int threadIdInWarp = (threadIdx.x & 0x1f);
	int maxIndex = iTopk - localTopkDoc->padding;
	int globalIndex = iTopk * blockIdx.x;// + localTopkDoc->padding;

	for (int localIndex = threadIdx.x; localIndex < maxIndex; localIndex += blockDim.x) {
		position_1 = 0;
//		position_2 = 0;

		score_1 = localTopkDoc->score[localIndex];
//		score_2 = (localIndex + blockDim.x < maxIndex) ? localTopkDoc->score[localIndex + blockDim.x] : 0.0;

		for (int i = threadIdInWarp; i < maxIndex; i+=32) {
			scoreAux = localTopkDoc->score[i];

			for (int t = 0; t < 32; ++t) {
				scoreNeighborAux = 0.0;
				scoreNeighborAux = __shfl_sync(0xFFFFFFFF,scoreAux, t);

				if(threadIdInWarp == maxIndex - 1)
					scoreNeighborAux = 0.0;

				if(threadIdInWarp == 31 && i + 1 < maxIndex){
					scoreNeighborAux = localTopkDoc->score[i + 1];
				}

				if(score_1 < scoreNeighborAux || (score_1 == scoreNeighborAux && (i+t) < localIndex)){
					position_1++;
				}


//
//				if(scoreNeighborAux != 0.0){
//					if(score_1 < scoreNeighborAux || (score_1 == scoreNeighborAux && (i+t) < localIndex)){
//						position_1++;
//					}
//				}
			}
		}

		dTopkScoreListGlobal[globalIndex + position_1] = score_1;
		iTopkDocListGlobal[globalIndex + position_1] = localTopkDoc->id[localIndex];

		globalIndex += blockDim.x;

//		if(localIndex+blockDim.x < maxIndex){
//			globalIndex += blockDim.x;
//			dTopkScoreListGlobal[globalIndex + position_2] = score_2;
//			iTopkDocListGlobal[globalIndex + position_2] = localTopkDoc->id[localIndex + blockDim.x];
//		}

//		if(localIndex + localIndex + (blockDim.x << 1) < maxIndex){
//			globalIndex += blockDim.x;
//			dTopkScoreListGlobal[globalIndex + position_3] = score_3;
//			iTopkDocListGlobal[globalIndex + position_3] = localTopkDoc->id[localIndex + (blockDim.x << 1)];
//		}
	}

//	int length = TOP_K - localTopkDoc->padding;
//	int height = log2f(length);
//
//	if(threadIdx.x < height){
//		int maxIndex = length - 1;
//		int subHeapRootIndex, positionChild, level, doc;
//		float score;
//
//		for (int i = (length - 1); i > 0; i--) {
//
//			doc = localTopkDoc->id[i];
//			score = localTopkDoc->score[i];
//
//			localTopkDoc->id[i] = localTopkDoc->id[0];
//			localTopkDoc->score[i] = localTopkDoc->score[0];
//
//			maxIndex--;
//			if((1 << height) - 1 > maxIndex)
//				height--;
//
//			if(threadIdx.x > height) break;
//
//			level = 0;
//			subHeapRootIndex = 0;
//			while(level < threadIdx.x && subHeapRootIndex <= maxIndex && localTopkDoc->score[subHeapRootIndex] <= score){
//				subHeapRootIndex <<= 1;
//				subHeapRootIndex++;
//				if(subHeapRootIndex + 1 <= maxIndex )
//					if(localTopkDoc->score[subHeapRootIndex + 1] < localTopkDoc->score[subHeapRootIndex])//Obtém o index do filho que tem o menor  valor
//						subHeapRootIndex++;
//
//				level++;
//			}
//
//			if(level != threadIdx.x ||  subHeapRootIndex > maxIndex)
//				subHeapRootIndex = -1;
//
//			positionChild = __shfl_down(subHeapRootIndex, 1);
//
//			if(level == threadIdx.x && subHeapRootIndex != -1){
//
//				if(positionChild != -1 && (threadIdx.x != height)) {
//					doc = localTopkDoc->id[positionChild];
//					score = localTopkDoc->score[positionChild];
//				}
//
//				localTopkDoc->id[subHeapRootIndex] = doc;
//				localTopkDoc->score[subHeapRootIndex] = score;
//			}
//		}
//	}
//
//	length--;
//	int globalIndex =  iTopk * blockIdx.x + threadIdx.x - localTopkDoc->padding;
//	int localIndex = blockDim.x - 1 - threadIdx.x - length;
//
//	__syncthreads();
//
//	for (; localIndex >= 0; localIndex -= blockDim.x) {
//		iTopkDocListGlobal[globalIndex]   = localTopkDoc->id[localIndex];
//		dTopkScoreListGlobal[globalIndex] = localTopkDoc->score[localIndex];
//		globalIndex += blockDim.x;
//	}
}




__device__ void sortLocalTopkDocAndStoreInGlobal(float *dTopkScoreListGlobal,int *iTopkDocListGlobal, int iTopk,
												documentTopkList *localTopkDoc){
//	float score_1=0.0, scoreAux=0.0, scoreNeighborAux=0.0;//score_3 score_2=0.0,
//	int position_1=0;//, position_2=0;//,position_3;
//	int threadIdInWarp = (threadIdx.x & 0x1f);
//	int maxIndex = iTopk - localTopkDoc->padding;
//	int maxIndexThreads = ((maxIndex >> 5) <<5);
//	maxIndexThreads = (maxIndex > maxIndexThreads) ?  (maxIndexThreads + 32) : maxIndexThreads;
//	int globalIndex = iTopk * blockIdx.x + localTopkDoc->padding; //threadIdx.x + localTopkDoc->padding;
//	int activeThreads;

//	if(maxIndex >= 32)
//		activeThreads = 32;
//	else
//		activeThreads = maxIndex;
//
//
//	for (int localIndex = threadIdx.x; localIndex < maxIndexThreads; localIndex+=blockDim.x) {
//
//		if(localIndex < maxIndex){
//			score_1 = localTopkDoc->score[localIndex];
//			position_1 = 0;
//		}
//
//		for (int i = threadIdInWarp; i < maxIndexThreads; i+=32) {
//			if(i < maxIndex)
//				scoreAux = localTopkDoc->score[i];
//
//			for (int t = 0; t < activeThreads; ++t) {
//				scoreNeighborAux = __shfl_sync(0xFFFFFFFF,scoreAux, t);
//
//				if(score_1 > scoreNeighborAux || (score_1 == scoreNeighborAux &&  ((i >> 5) << 5) + t < localIndex )){// i > localIndex ) ){ //(i+t) < localIndex)){
//					position_1++;
//				}
//			}
//			activeThreads = maxIndex - (((i+32) >> 5) << 5); //Gato: valor de i precisa estar atualizado
//		}
//
//		if(localIndex < maxIndex){
//			dTopkScoreListGlobal[globalIndex + position_1] = score_1;
//			iTopkDocListGlobal[globalIndex + position_1] = localTopkDoc->id[localIndex];
//			if(maxIndex >= 32)
//					activeThreads = 32;
//				else
//					activeThreads = maxIndex;
//		}
//	}

	int globalIndex = iTopk * blockIdx.x + localTopkDoc->padding; //threadIdx.x + localTopkDoc->padding;
	float ownScore=0.0, scoreNeighborAux=0.0, scoreInList=0.0;
	int threadIdInWarp =  (threadIdx.x & 0x1f);
	int maxIndex =  iTopk - localTopkDoc->padding;
	long long position;
	for (int localIndex = threadIdx.x; localIndex < iTopk; localIndex+=blockDim.x) {
		position = 0;
		if(localIndex < maxIndex){
			ownScore = localTopkDoc->score[localIndex];
		}
		else {
			ownScore = NO_MORE_DOC;
		}

		for (int indexInlist = threadIdInWarp; indexInlist < iTopk; indexInlist+=32) {
			if(indexInlist < maxIndex){
				scoreInList = localTopkDoc->score[indexInlist];
			}else{
				scoreInList= NO_MORE_DOC;
			}

			for (int threadId = 0; threadId < 32; ++threadId) {

				if(threadId >= (maxIndex - ((indexInlist >> 5) << 5)))
					break;

				scoreNeighborAux = __shfl_sync(0xFFFFFFFF,scoreInList, threadId);

				if(ownScore > scoreNeighborAux || (ownScore == scoreNeighborAux && (((indexInlist >> 5) << 5) + threadId) < localIndex )){// i > localIndex ) ){ //(i+t) < localIndex)){
					position++;
				}
			}
		}
		if(localIndex < maxIndex){
			dTopkScoreListGlobal[globalIndex + position] = ownScore;
			iTopkDocListGlobal[globalIndex + position] = localTopkDoc->id[localIndex];
		}

	}

	__syncthreads();
}

