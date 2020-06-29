/*
 * ParallelPrunningDaat.cu
 *
 *  Created on: 08/12/2017
 *      Author: roussian
 */
#include "ParallelPrunningDaat.cuh"
#include "DeviceFunctions.cuh"
#include "UnityTest.cuh"
#include <stdio.h>


__device__ volatile float globalThreshold = 0.0;
__device__ volatile float globalThresholdBatch[500];
__device__ int globalCount=0;

__global__ void mergeTopkLists_v3(float *dTopkScoreList,
								  int *iTopkDocList,
								  int iTopk,
								  int iMergeNumber,
								  int iSkipTopkBetweenMerges,
								  int iSkipTopkBetweenBlocks,
								  int iTotalElementos){

//	if(blockIdx.x != 74)
//		return;

	//Peguei o doc idblock 4 skipBlock 32 skipMerges 16 na posição 18559 !
//	if(iSkipTopkBetweenBlocks == 4 && iSkipTopkBetweenMerges == 2)
//	if(blockIdx.x != 87)
//		return;

	__shared__ documentTopkList documentTopkSharedList_1;
	__shared__ documentTopkList documentTopkSharedList_2;
	__shared__ documentTopkList documentTopkSharedList_Partial;

	__shared__ short int halfNumberTopk;// = iTopk >> 1;
	__shared__ short int iTopkPosition;// = iTopk - 1; // Começa no índice 0 e vai iTopk - 1
	__shared__ short int halfPositionNumberTopk;// = iTopkPosition >> 1;

	if(THREAD_MASTER){
		halfNumberTopk = iTopk >> 1;
	}else if(THREAD_MASTER_2){
		iTopkPosition = iTopk - 1;
		halfPositionNumberTopk = iTopkPosition >> 1;
	}

	int warpIndex = threadIdx.x >> 5;
	int threadWarpId =  threadIdx.x - (warpIndex << 5); // (threadIdx.x & 0x1f); //threadIdx.x % 32;
	int isOdd = warpIndex & 1; //Verifica se o IdWarp é ímpar
	int numberThreadsInList = ((blockDim.x >> 6) << 5); // (#Block/Tamanho da Warp--2⁵) / 2; ---> isso pq metade do # de warps trabalham sobre uma lista
	warpIndex = warpIndex >> 1; //Isso pois as warps são divididas por impar e par. Então, se o idWarp é 5, então o novo id é 2

	__syncthreads();
//	   int proportion = iTopk / blockDim.x; //K é múltiplo do numero de threads por bloco

	int offset = iTopkPosition; //a Posição que cada thread irá inserir o seu elemento
	//A posição das threads nas listas --- half + (pos. da warp * #threads dentro da warp) + id
	int indexInMemShared = halfNumberTopk + ( warpIndex << 5) + threadWarpId;//(iTopk >> 1) + ((warpId >> 1) << 5) + threadWarpId;///half + (pos. da warp * #threads na warp) + id

	float score_1, score_2;
	float *ownScorePtr, *workListPtr;
	int *ownDocId;

	int position;
	int index_1, index_2, indexLocal;
//	int isEndPart;

	//Obtém a posição inicial que a thread irá inserir na lista final.
	offset -=  (iTopkPosition - indexInMemShared ) << 1;//A multiplicação por 2 é por causa das duas listas

//	if(blockIdx.x == 203)
//		printf("Oi!\n");
	__syncthreads();

	//As listas estão alinhadas em uma lista, por isso que o indice tem que seguir para
	//a próxima parte não processada referente ao bloco
	//Um merge pega 2 listas ou 1 lista + Resultado anterior
	index_1 = blockIdx.x * iTopk * iSkipTopkBetweenBlocks + threadIdx.x;//blockIdx.x * iTopk * (iMergeNumber + 1) * iSkipBetweenMerge + threadIdx.x;
	index_2 = index_1 + iTopk * iSkipTopkBetweenMerges;//index_1 + iTopk * iSkipTopkBetweenMerges;
//	isEndPart = 0;//index_2 > totalElements;

	indexLocal = threadIdx.x;
	//O número de threads por bloco pode ser menor que K
	while(indexLocal < iTopk){

		documentTopkSharedList_1.id[indexLocal] = iTopkDocList[index_1];
		documentTopkSharedList_1.score[indexLocal] = dTopkScoreList[index_1];

//		if(iTopkDocList[index_1] == 46517642)
//			printf("Peguei o doc idblock %d skipBlock %d skipMerges %d na posição %d !\n",
//					blockIdx.x, iSkipTopkBetweenBlocks,iSkipTopkBetweenMerges,index_1);

		index_1 += blockDim.x;
		indexLocal += blockDim.x;
	}



//	if(THREAD_MASTER && blockIdx.x == 0 && iSkipTopkBetweenBlocks >= 2048)
//		printf("idblock %d skipBlock %d skipMerges %d na posição inicial %d %d!\n",
//				blockIdx.x, iSkipTopkBetweenBlocks,iSkipTopkBetweenMerges,blockIdx.x * iTopk * iSkipTopkBetweenBlocks,
//				blockIdx.x * iTopk * iSkipTopkBetweenBlocks + iTopk * iSkipTopkBetweenMerges);

	__syncthreads();

//	if(THREAD_MASTER && blockIdx.x == 0){
//		printf("First List - ");
//		for (int i = 0; i < iTopk; ++i) {
//			printf(" %.2f ", documentTopkSharedList_1.score[i]);
//		}
//		printf("\n");
//	}

	for (int globalRound = 0; globalRound < iMergeNumber; ++globalRound) {
		//O número de threads por bloco pode ser menor que K
		indexLocal = threadIdx.x;
		while(indexLocal < iTopk){
			if(index_2 >= iTotalElementos || index_2 < 0){
				documentTopkSharedList_2.id[indexLocal] = 0;
				documentTopkSharedList_2.score[indexLocal] = 0;
			}else{
				documentTopkSharedList_2.id[indexLocal] = iTopkDocList[index_2];
				documentTopkSharedList_2.score[indexLocal] = dTopkScoreList[index_2];
//				if(iTopkDocList[index_2] == 46517642)
//					printf("Peguei o doc idblock %d skipBlock %d skipMerges %d na posição %d !\n",
//							blockIdx.x, iSkipTopkBetweenBlocks,iSkipTopkBetweenMerges,index_2);
			}
			index_2 += blockDim.x;
			indexLocal += blockDim.x;
		}

		__syncthreads();

		if(!isOdd){//As threads das Warps com ids par trabalham sobre os maiores elementos da mesma posição.

			do {//Esse bloco de instruções trabalha somente com dados que estão na memória compartilhada
				score_1 = documentTopkSharedList_1.score[indexInMemShared];
				score_2 = documentTopkSharedList_2.score[indexInMemShared];
				//Escolhe o maior elemento de uma mesma posição e a lista, a que tiver o menor elemento, que irá pecorrer.
				if(score_1 >= score_2){
					ownScorePtr = &score_1;
					ownDocId = &documentTopkSharedList_1.id[indexInMemShared];

//					if(*ownDocId == 46517642){// && blockIdx.x == 5
//						printf("1 - blockId %d threadId %d\n", blockIdx.x, threadIdx.x);
//					}

					workListPtr = documentTopkSharedList_2.score; //A lista de trabalho sempre é a lista do menor elemento
				}
				else{
					ownScorePtr = &score_2;
					ownDocId = &documentTopkSharedList_2.id[indexInMemShared];

//					if(*ownDocId == 46517642){// && blockIdx.x == 5){
//						printf("1.1 - blockId %d threadId %d\n", blockIdx.x, threadIdx.x);
//					}
					workListPtr = documentTopkSharedList_1.score;
				}

				if(score_1 != score_2){
					//Busca atualizar o offset, i.e., procura o 1º elemento maior
					position = indexInMemShared;//Define a posição início para fazer as comparações (//Se workscore está em A[i] e A[i] < B[i], então A[i] < B[i + (1,2,3...)])
					while( (position+1 < iTopk) && (*ownScorePtr > workListPtr[position+1]) ){
						offset++; //Ao encontrar um elemento menor, ele irá aumentar a posição que irá inserir o seu elemento
						position++;
					}
				}

				//Insere os maiores elementos das listas, i.e., os elementos mais a direita da lista dos top-k
				documentTopkSharedList_Partial.score[offset] = *ownScorePtr;
				documentTopkSharedList_Partial.id[offset] = *ownDocId;

				//Redefine as variáveis para inicializar outro bloco de dados que está na memória compartilhada
				indexInMemShared += numberThreadsInList;
				offset = iTopkPosition - ((iTopkPosition - indexInMemShared ) << 1); //Reinicia o offset

			} while (indexInMemShared < iTopk);

		}else{
			int count; //Quantos elementos irá buscar;
			float *ownScoreListPtr;

			do{//Esse bloco de instrução trabalha somente com dados que estão na memória compartilhada
				offset--;//É o menor elemento entre dois elementos (mesmo índice)

				score_1 = documentTopkSharedList_1.score[indexInMemShared];
				score_2 = documentTopkSharedList_2.score[indexInMemShared];
				//Escolhe o menor elemento de uma mesma posição e a lista, a que tiver o maior elemento, que irá pecorrer.
				if(score_1 < score_2){
					ownScorePtr = &score_1;
					ownDocId = &documentTopkSharedList_1.id[indexInMemShared];
//					if(*ownDocId == 46517642){
//						printf("2 -  blockId %d threadId %d\n", blockIdx.x, threadIdx.x);
//					}
					workListPtr = documentTopkSharedList_2.score;
					ownScoreListPtr = documentTopkSharedList_1.score;
				}
				else{//Entra igual(se for igual, entao o score_2 é selecionado) ou menor
					ownScorePtr = &score_2;
					ownDocId = &documentTopkSharedList_2.id[indexInMemShared];

//					if(*ownDocId == 46517642){
//						printf("2.1 - blockId %d threadId %d\n", blockIdx.x, threadIdx.x);
//					}
					workListPtr = documentTopkSharedList_1.score;
					ownScoreListPtr = documentTopkSharedList_2.score;
				}
				//Duas possibilidades podem ocorrer: (1) O elemento adquirido está entre os k maiores elementos
				//(2) o elemento não está entre os k maiore elementos
				if(*ownScorePtr >= workListPtr[halfPositionNumberTopk]){//Compara-se com o elemento que está na metade//if(*ownScorePtr > workListPtr[iTopkPosition >> 1]){//Compara-se com o elemento que está na metade

					position = indexInMemShared;// - 1;
					while(  (position - 1 > 0) && (*ownScorePtr < workListPtr[position-1]) ){
						offset--;
						position--;
					}

					documentTopkSharedList_Partial.score[offset] = *ownScorePtr;
					documentTopkSharedList_Partial.id[offset] = *ownDocId;

				}else{

					offset -= indexInMemShared - halfNumberTopk;//(iTopk >> 1); Subtrai da metade do número das posições e não do índice máx, pois já ouve uma subtração do conjunto dos maiores elementos
					count = halfPositionNumberTopk - offset; //Quantos elementos irá buscar;

					float *aux;
					int posWork, posOwn;
					int *docIdOwn, *docIdWork;
					if(ownScoreListPtr[iTopkPosition] >= workListPtr[halfPositionNumberTopk] ){
						ownScorePtr = &ownScoreListPtr[iTopkPosition];
						posOwn = iTopkPosition;
						posWork = halfPositionNumberTopk;
						if(ownScoreListPtr == documentTopkSharedList_2.score){
							docIdOwn = documentTopkSharedList_2.id;
							docIdWork = documentTopkSharedList_1.id;
						}else{
							docIdOwn = documentTopkSharedList_1.id;
							docIdWork = documentTopkSharedList_2.id;
						}
					}else{
						ownScorePtr = &workListPtr[halfPositionNumberTopk];
						aux = ownScoreListPtr;
						ownScoreListPtr = workListPtr;
						workListPtr = aux;
						posOwn = halfPositionNumberTopk;
						posWork = iTopkPosition;

						if(workListPtr == documentTopkSharedList_2.score){
							docIdWork = documentTopkSharedList_2.id;
							docIdOwn = documentTopkSharedList_1.id;
						}else{
							docIdWork = documentTopkSharedList_1.id;
							docIdOwn = documentTopkSharedList_2.id;
						}
					}

					while(count > 0){

						while((workListPtr[posWork] <= ownScoreListPtr[posOwn]) && (count > 0)){
							posOwn--;
							count--;
						}
//						posOwn++;

						if(count == 0){
							ownScorePtr = &ownScoreListPtr[posOwn];
							ownDocId = &docIdOwn[posOwn];
						}else{

							while((ownScoreListPtr[posOwn] <= workListPtr[posWork]) && count > 0){
								posWork--;
								count--;
							}
//							posWork++;
							if(count == 0){
								ownScorePtr = &workListPtr[posWork];
								ownDocId = &docIdWork[posWork];
							}
						}
					}

					documentTopkSharedList_Partial.score[offset] = *ownScorePtr;
					documentTopkSharedList_Partial.id[offset] = *ownDocId;
				}

				indexInMemShared += numberThreadsInList;
				offset = iTopkPosition - ((iTopkPosition - indexInMemShared ) << 1); //Reinicia o offset

			} while(indexInMemShared < iTopk);

		}//IF-ELSE ODD

		__syncthreads();

		indexLocal = threadIdx.x;
		while(indexLocal < iTopk){

			documentTopkSharedList_1.id[indexLocal] = documentTopkSharedList_Partial.id[indexLocal];
			documentTopkSharedList_1.score[indexLocal] = documentTopkSharedList_Partial.score[indexLocal];

			indexLocal += blockDim.x;
		}
		// -1 por causa do avanço realizado pelas threads para o próximo bloco de topk documentos no último loop
		index_2 += iTopk * (iSkipTopkBetweenMerges - 1);
		indexInMemShared = halfNumberTopk + ( warpIndex << 5) + threadWarpId;
		offset = iTopkPosition - ((iTopkPosition - indexInMemShared ) << 1); //Reinicia o offset

//		checkMerge_Sorting_Documents(documentTopkSharedList_Partial, iSkipTopkBetweenMerges, iSkipTopkBetweenBlocks, iTopk);
	}

	__syncthreads();

	index_1 = blockIdx.x * iTopk * iSkipTopkBetweenBlocks + threadIdx.x;
	indexLocal = threadIdx.x;
	while(indexLocal < iTopk){
//		if(isEndPart)
//			break;
//		if(documentTopkSharedList_Partial.id[indexLocal] == 46517642)
//			printf("Entregando o doc idblock %d skipBlock %d skipMerges %d em %d!\n",
//					blockIdx.x, iSkipTopkBetweenBlocks,iSkipTopkBetweenMerges,index_1);

		if(documentTopkSharedList_Partial.score[indexLocal] != 0.0){
			iTopkDocList[index_1] = documentTopkSharedList_Partial.id[indexLocal];
			dTopkScoreList[index_1] = documentTopkSharedList_Partial.score[indexLocal];
		}

		indexLocal += blockDim.x;
		index_1 += blockDim.x;
	}

//	__syncthreads();
//
//	if(THREAD_MASTER && blockIdx.x == 0){
//		printf("Final List - ");
//		for (int i = 0; i < iTopk; ++i) {
//			printf(" %.2f ", documentTopkSharedList_Partial.score[i]);
//		}
//		printf("\n");
//	}

}

__global__ void matchWandParallel_FIXED_2(const int* iDocIdList, const unsigned short int* iFreqList,
								  const float *dUBlist, const float *dIdfList,
								  const int *iDocLenghtList,
								  const short int iTermNumber, int *iTopkDocListGlobal,
								  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
								  const int iGlobalRoundNumber,// const int iBlockRoundNumber,
								  const short int iTopK,
								  const float iInitialThreshold,const int* d_iDocNumberByTermList){

		int count=0;

		__shared__ pivot sharedPivot;
		__shared__ finger fingers[TERM_NUMBER];

		__shared__ postingList postingLists[TERM_NUMBER];
		__shared__ documentTopkList documentTopk;

		__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];

		__shared__ int iGlobalInitialPosition;

		__shared__ float score;
		__shared__ bool isValidCandidate;

		int positionInitialInTermPostingList;
		float thresholdLocal = iInitialThreshold;
		int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
		int localIndex;

		if(THREAD_MASTER){
			iGlobalInitialPosition = blockDim.x  * blockIdx.x * iGlobalRoundNumber;
			documentTopk.padding = iTopK;
		}

		if(thresholdLocal < globalThreshold)
			thresholdLocal = globalThreshold;

		//Inicializa a lista de Score e Documentos dos Topk
		//Considero que o Top_K seja um número múltiplo do tamanho do bloco
		for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
			documentTopk.id[localIndex] = -1;
			documentTopk.score[localIndex] = 0.0;
		}

		__syncthreads();

		for (int globalRound = 0; globalRound < iGlobalRoundNumber; ++globalRound) {
			positionInitialInTermPostingList = 0;
			for (int termIndex = 0; termIndex < iTermNumber; ++termIndex) {
				localIndex = threadIdx.x;
				globalIndex = positionInitialInTermPostingList + iGlobalInitialPosition + localIndex;
				while(localIndex < DOC_QUANTITY_IN_MEMORY){//(globalIndex < d_iDocNumberByTermList[termIndex] && localIndex < DOC_QUANTITY_IN_MEMORY){

					if(globalIndex < d_iDocNumberByTermList[termIndex] + positionInitialInTermPostingList){
						postingLists[termIndex].docId[localIndex] =  iDocIdList[globalIndex];//[positionInitialInTermPostingList + globalIndex];
						postingLists[termIndex].freq[localIndex] = iFreqList[globalIndex];
						postingLists[termIndex].docLenght[localIndex] = iDocLenghtList[globalIndex];
					}
					else{
						postingLists[termIndex].docId[localIndex] = NO_MORE_DOC;
					}
					localIndex += blockDim.x;
					globalIndex += blockDim.x;
				}

				if(THREAD_MASTER){
					fingers[termIndex].docId = postingLists[termIndex].docId[0];
					fingers[termIndex].position = (fingers[termIndex].docId == NO_MORE_DOC) ? NO_VALID_POSITION : 0 ;
				}

				positionInitialInTermPostingList += d_iDocNumberByTermList[termIndex];
			}

			__syncthreads();

//			if(fingers[0].docId == 16563866)
//				printf("Oi!");

			//Sort the terms in non decreasing order of DID
			sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

//			__syncthreads();

			//Select term pivot
			if(THREAD_MASTER){
				selectTermPivot_2(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			}else if(THREAD_MASTER_2){
				score = 0.0;
			}

			__syncthreads();

			int padding;
			int threadIdInWarp = (threadIdx.x & 0x1f);
			int idWarp = ((blockDim.x >> 5) == 1 ) ? 1 :  threadIdx.x >> 5;

			while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){

				if(THREAD_MASTER)
					isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);

				count++;
				padding = documentTopk.padding;

				__syncthreads();

				if(isValidCandidate){
					//Avaliação Completa
					if(threadIdx.x < iTermNumber){
						fullScore_3_1(&score, fingers[sharedPivot.idTerm].docId, iOrderedTermSharedList,
									  fingers,postingLists, dIdfList, dAverageDocumentLength);
					}

					__syncthreads();

//					if(padding != 0 || thresholdLocal < score){
					if(thresholdLocal < score){
						thresholdLocal = managerMinValue_v5(&documentTopk, fingers[sharedPivot.idTerm].docId,score,padding);
//						thresholdLocal = documentTopk.score[0];
					}

					if(idWarp == 1 && threadIdInWarp < iTermNumber ){
						advancePivoTermFinger_4(sharedPivot,fingers, postingLists,DOC_QUANTITY_IN_MEMORY,threadIdInWarp);
					}
				}
				else{
					 advanceDocIdOfPredecessorTerm_4(postingLists,
												   iOrderedTermSharedList,
												   fingers,sharedPivot,fingers[sharedPivot.idTerm].docId,
												   DOC_QUANTITY_IN_MEMORY);
				}

//				if(fingers[0].docId == 16563866)
//					printf("Oi!");

				__syncthreads();

				sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

//				__syncthreads();

				//Select term sharedPivot
				if(THREAD_MASTER){
					selectTermPivot_2(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
				}else if(THREAD_MASTER_2){
					score = 0.0;
				}

				__syncthreads();
			}

			if(THREAD_MASTER){
				iGlobalInitialPosition += DOC_QUANTITY_IN_MEMORY;
			}

			if (SHAREDTHESHOLD == 1){//SHARED_READ
				if(THREAD_MASTER && documentTopk.padding == 0 && thresholdLocal > globalThreshold){
//					atomicMax(&globalThreshold,thresholdLocal);
					globalThreshold = thresholdLocal;
				}

				if(thresholdLocal < globalThreshold){
					thresholdLocal = globalThreshold;
				}
			}else if (SHAREDTHESHOLD == 2){ //TSHARED_WRITEREAD
				if(THREAD_MASTER && (documentTopk.padding < (iTopK >> 1)) && thresholdLocal > globalThreshold){
//					atomicMaxD(&globalThreshold,thresholdLocal);
					globalThreshold = thresholdLocal;
				}

				if((documentTopk.padding < (iTopK >> 1)))
					if(thresholdLocal < globalThreshold){
						thresholdLocal = globalThreshold;
					}
			}

			__syncthreads();
		}

		sortLocalTopkDocAndStoreInGlobal(dTopkScoreListGlobal,iTopkDocListGlobal,iTopK,&documentTopk);

		if(THREAD_MASTER)
				atomicAdd(&globalCount,count);

		if(THREAD_MASTER)
			printf("-----%d----", globalCount);

}

__global__ void matchWandParallel_VARIABLE_Batch_Block_3(const int* iDocIdList, const unsigned short int* iFreqList,
														const float *dUBlistGlobal, const float *dIdfListGlobal, const int *iDocLenghtList,
														const short int* iTermNumberByQuery, int *iTopkDocListGlobal,
														float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
														const short int iTopK, const float iInitialThreshold,
														const int* iQueryTerms, const long long* ptrInitPostingList,
														int* ptrQueryPositions,int *iDocNumberByTermListGlobal){





	__shared__ int queryPosition;
	__shared__ pivot sharedPivot;
	__shared__ finger fingers[TERM_NUMBER];

	__shared__ documentTopkList documentTopk;


	__shared__ postingList2 postings[TERM_NUMBER];
	__shared__ int positionInShared[TERM_NUMBER];

	__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];
	__shared__ long long iSharedPositionInitialInList[TERM_NUMBER];
	__shared__ float dUBlist[TERM_NUMBER];
	__shared__ float dIdfList[TERM_NUMBER];
	__shared__ float dAverageDocumentLength;

	__shared__ int iDocNumberByTermList[TERM_NUMBER];

	__shared__ float score;
	__shared__ bool isValidCandidate;
	__shared__ int docCurrent;
	__shared__ limitDocId limitDoc;
	__shared__ short int iTermNumber;

	int count;
	int padding;

 	float thresholdLocal;// = iInitialThreshold;
 	thresholdLocal = iInitialThreshold;

	int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
	int localIndex;

//	int count = 0;

	if(THREAD_MASTER){
		documentTopk.padding = iTopK;
		dAverageDocumentLength = dAverageDocumentLengthGlobal;
		limitDoc.secondMaxDocId = -1;
		iTermNumber = iTermNumberByQuery[blockIdx.x];
	}

	__syncthreads();

	if(threadIdx.x < iTermNumber){
		queryPosition = ptrQueryPositions[blockIdx.x];
		int idTerm = iQueryTerms[queryPosition + threadIdx.x];

		iDocNumberByTermList[threadIdx.x] = iDocNumberByTermListGlobal[idTerm];

		dUBlist[threadIdx.x] = dUBlistGlobal[idTerm]*1.0;//[threadIdx.x];
		dIdfList[threadIdx.x] = dIdfListGlobal[idTerm];//[threadIdx.x];

		iSharedPositionInitialInList[threadIdx.x] = ptrInitPostingList[idTerm];
		positionInShared[threadIdx.x] = -1;
	}

	//Inicializa a lista de Score e Documentos dos Topk
	//Considero que o Top_K seja um número múltiplo do tamanho do bloco
	#pragma unroll 4
	for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
		documentTopk.id[localIndex] = -1;
		documentTopk.score[localIndex] = 0.0;
	}

	//Define o max e o min
	if(threadIdx.x < iTermNumber){
		int docAmount = iDocNumberByTermList[threadIdx.x];
		globalIndex = 0;

		int maxDoc;

		if(THREAD_MASTER) limitDoc.minDocId = 0;

		globalIndex = docAmount;

		maxDoc =  iDocIdList[iSharedPositionInitialInList[threadIdx.x] +  globalIndex - 1];

		atomicMax(&(limitDoc.secondMaxDocId), maxDoc);

		fingers[threadIdx.x].docId = iDocIdList[iSharedPositionInitialInList[threadIdx.x]];
		fingers[threadIdx.x].position = iSharedPositionInitialInList[threadIdx.x];
	}

//	__syncthreads();

	int pos;
	for (int termId = 0; termId < iTermNumber; ++termId) {
		for (int localIndex = threadIdx.x; localIndex < DOC_QUANTITY_IN_MEMORY; localIndex+=blockDim.x) {
			pos = fingers[termId].position+localIndex+1;
			if(pos < iSharedPositionInitialInList[termId] + iDocNumberByTermList[termId]){
				postings[termId].docId[localIndex] = iDocIdList[pos];
			}else{
				postings[termId].docId[localIndex] = NO_MORE_DOC;
			}
		}
	}

	sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

	if(THREAD_MASTER){
		selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
		docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
		score = 0.0;
	}

	__syncthreads();

	while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){

		isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);
		count++;

//		if(count == 2559)
//			printf("Oi");

		if(isValidCandidate){
			if(threadIdx.x < iTermNumber){
		 		int termId = iOrderedTermSharedList[threadIdx.x];
		 		float scoreL = 0.0;
		 		if(fingers[termId].docId == fingers[sharedPivot.idTerm].docId){
		 			scoreL = scoreTf_Idf(iFreqList[fingers[termId].position],
										iDocLenghtList[fingers[termId].position],
										dIdfList[termId],dAverageDocumentLength,1.0);
		 		}

		 		float aux = 0;
		 		for (int i = 0; i < TERM_NUMBER; ++i) {
		 			aux += __shfl_sync(0xFFFFFFFF,scoreL,i);
		 		}

		 		if(THREAD_MASTER) score = aux;
//		 		atomicAdd(&score,scoreL);
			}

			padding = documentTopk.padding;

			__syncthreads();

/*				If the heap is not full
			the candidate is inserted into the heap. If the heap is full
			and the new score is larger than the minimum score in the
			heap, the new document is inserted into the heap, replacing
			the one with the minimum score.

*/
			if(padding != 0 || thresholdLocal < score ){
				thresholdLocal = managerMinValue_v5(&documentTopk, fingers[sharedPivot.idTerm].docId, score,padding);
			}

			for (int idTerm = 0; idTerm < iTermNumber; ++idTerm) {
				if(THREAD_MASTER && positionInShared[idTerm] == -1)
					fingers[idTerm].position++;

				if(docCurrent == fingers[idTerm].docId){
					fingers[idTerm].docId = NO_MORE_DOC;

					int docIdLocal, localIndex=0;
					for (localIndex = threadIdx.x + positionInShared[idTerm]; localIndex < DOC_QUANTITY_IN_MEMORY; localIndex+=blockDim.x) {
						docIdLocal = postings[idTerm].docId[localIndex];
						if(docIdLocal > docCurrent && docIdLocal != NO_MORE_DOC){
							if(localIndex == 0 || (postings[idTerm].docId[localIndex-1] <= docCurrent)){
								fingers[idTerm].docId = docIdLocal;
								fingers[idTerm].position += localIndex - positionInShared[idTerm];
								positionInShared[idTerm] = localIndex;
							}
							break;
						}
					}
				}
			}
		}
		else{
			int idTerm;
			for (int j = 0; j < sharedPivot.positionInOrderedList; ++j) {
				idTerm = iOrderedTermSharedList[sharedPivot.positionInOrderedList];

				if(docCurrent == fingers[threadIdx.x].docId)
					break;

				fingers[idTerm].docId = NO_MORE_DOC;

				if(THREAD_MASTER && positionInShared[idTerm] == -1)
					fingers[idTerm].position++;

				int docIdLocal, localIndex=0;
				for (localIndex = threadIdx.x+positionInShared[idTerm]; localIndex < DOC_QUANTITY_IN_MEMORY; localIndex+=blockDim.x) {
					docIdLocal = postings[idTerm].docId[localIndex];
					if(docIdLocal >= docCurrent && docIdLocal != NO_MORE_DOC){
						if(localIndex == 0 || (postings[idTerm].docId[localIndex-1] < docCurrent)){
							fingers[idTerm].docId = docIdLocal;
							fingers[idTerm].position += localIndex - positionInShared[idTerm];
							positionInShared[idTerm] =  localIndex;
						}
						break;
					}
				}
			}
		}

		__syncthreads();

		for (int termId = 0; termId < iTermNumber; ++termId) {
			if(fingers[termId].docId == NO_MORE_DOC && fingers[termId].position != NO_VALID_POSITION){
				int pos, localIndex;
				for (localIndex = threadIdx.x; localIndex < DOC_QUANTITY_IN_MEMORY; localIndex+=blockDim.x) {
					pos = fingers[termId].position+localIndex+1;
					if(pos < iSharedPositionInitialInList[termId] + iDocNumberByTermList[termId]){
						postings[termId].docId[localIndex] = iDocIdList[pos];
					}else{
						postings[termId].docId[localIndex] = NO_MORE_DOC;
					}
				}

				if(THREAD_MASTER && postings[termId].docId[0] == NO_MORE_DOC)
					fingers[termId].position = NO_VALID_POSITION;
				else{
					fingers[termId].docId = postings[termId].docId[0];
					positionInShared[termId] = -1;
					fingers[termId].position = pos;
				}
			}
		}
		__syncthreads();

		//Sort the terms in non decreasing order of DID
		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

		//Select term pivot
		if(THREAD_MASTER){
			selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
			score = 0.0;
		}
		__syncthreads();
	}

	if(blockIdx.x==499 && THREAD_MASTER)
		printf("-----%d %d----", blockIdx.x, count);



	sortLocalTopkDocAndStoreInGlobal((float*)&(dTopkScoreListGlobal[blockIdx.x*iTopK]),(int*)&(iTopkDocListGlobal[blockIdx.x*iTopK]),iTopK,&documentTopk);

//		if(THREAD_MASTER)
////			atomicAdd(&globalCount,count);
////
//		if


}

__global__ void matchWandParallel_VARIABLE_Batch_Block_Test(const int* iDocIdList, const unsigned short int* iFreqList,
														const float *dUBlistGlobal, const float *dIdfListGlobal, const int *iDocLenghtList,
														const short int* iTermNumberByQuery, int *iTopkDocListGlobal,
														float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
														const short int iTopK, const float iInitialThreshold,
														const int* iQueryTerms, const long long* ptrInitPostingList,
														int* ptrQueryPositions,int *iDocNumberByTermListGlobal,
														const int* iOrderQueryList){

	__shared__ int queryPosition;
	__shared__ pivot sharedPivot;
	__shared__ finger fingers[TERM_NUMBER];

	__shared__ documentTopkList documentTopk;

	__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];
	__shared__ long long iSharedPositionInitialInList[TERM_NUMBER];
	__shared__ float dUBlist[TERM_NUMBER];
	__shared__ float dIdfList[TERM_NUMBER];
	__shared__ float dAverageDocumentLength;

	__shared__ int iDocNumberByTermList[TERM_NUMBER];

	__shared__ float score;
	__shared__ bool isValidCandidate;
	__shared__ int docCurrent;
	__shared__ limitDocId limitDoc;
	__shared__ short int iTermNumber;

	int padding;

	float thresholdLocal;// = iInitialThreshold;
	thresholdLocal = iInitialThreshold;

	int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
	int localIndex;

//	int count = 0;

	if(THREAD_MASTER){
		documentTopk.padding = iTopK;
		dAverageDocumentLength = dAverageDocumentLengthGlobal;
//	}else if(THREAD_MASTER_2){
//		limitDoc.minDocId = -1;
		limitDoc.secondMaxDocId = -1;

		iTermNumber = iTermNumberByQuery[blockIdx.x];
	}

	__syncthreads();

	if(threadIdx.x < iTermNumber){
		queryPosition = ptrQueryPositions[iOrderQueryList[blockIdx.x]];
		int idTerm = iQueryTerms[queryPosition + threadIdx.x];

		iDocNumberByTermList[threadIdx.x] = iDocNumberByTermListGlobal[idTerm];

		dUBlist[threadIdx.x] = dUBlistGlobal[idTerm];//[threadIdx.x];
		dIdfList[threadIdx.x] = dIdfListGlobal[idTerm];//[threadIdx.x];

//		printf(" %.2f ",dUBlist[threadIdx.x]);
		iSharedPositionInitialInList[threadIdx.x] = ptrInitPostingList[idTerm];
	}

	//Inicializa a lista de Score e Documentos dos Topk
	//Considero que o Top_K seja um número múltiplo do tamanho do bloco
	#pragma unroll 4
	for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
		documentTopk.id[localIndex] = -1;
		documentTopk.score[localIndex] = 0.0;
	}

	//Define o max e o min
	if(threadIdx.x < iTermNumber){
		int docAmount = iDocNumberByTermList[threadIdx.x];
//		fingers[threadIdx.x].final = 0;
//		limitDoc.extraPosition[threadIdx.x] = 0;

		globalIndex = 0;

		int maxDoc;

		if(THREAD_MASTER) limitDoc.minDocId = 0;

		globalIndex = docAmount-1;

		maxDoc =  iDocIdList[iSharedPositionInitialInList[threadIdx.x] +  globalIndex];

		atomicMax(&(limitDoc.secondMaxDocId), maxDoc);

		fingers[threadIdx.x].docId = iDocIdList[iSharedPositionInitialInList[threadIdx.x]];
		fingers[threadIdx.x].position = iSharedPositionInitialInList[threadIdx.x];
	}

	__syncthreads();


	sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

	if(THREAD_MASTER){
		selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
		docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
//	}else if(THREAD_MASTER_2){
		score = 0.0;
	}

	__syncthreads();

	while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){

		isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);
//		count++;
//
//		if(fingers[sharedPivot.idTerm].docId == 38182)
//			printf("Oi");

		if(isValidCandidate){
			if(threadIdx.x < iTermNumber){
				int termId = iOrderedTermSharedList[threadIdx.x];
				float scoreL = 0.0;
				if(fingers[termId].docId == fingers[sharedPivot.idTerm].docId){
					scoreL = scoreTf_Idf(iFreqList[fingers[termId].position],
										iDocLenghtList[fingers[termId].position],
										dIdfList[termId],dAverageDocumentLength,1.0);
				}

				float aux = 0;
				for (int i = 0; i < TERM_NUMBER; ++i) {
					aux += __shfl_sync(0xFFFFFFFF,scoreL,i);
				}

				if(THREAD_MASTER) score = aux;
//		 		atomicAdd(&score,scoreL);
			}

			padding = documentTopk.padding;

			__syncthreads();

/*				If the heap is not full
			the candidate is inserted into the heap. If the heap is full
			and the new score is larger than the minimum score in the
			heap, the new document is inserted into the heap, replacing
			the one with the minimum score.

*/
			if(padding != 0 || thresholdLocal < score ){
				thresholdLocal = managerMinValue_v5(&documentTopk, fingers[sharedPivot.idTerm].docId, score,padding);
			}

			if(threadIdx.x < iTermNumber ){
				int docPivot = fingers[sharedPivot.idTerm].docId;
				if(fingers[threadIdx.x].docId ==  docPivot){
					fingers[threadIdx.x].position++;
					if(fingers[threadIdx.x].position >= (iDocNumberByTermList[threadIdx.x]+iSharedPositionInitialInList[threadIdx.x])){//Não Válido
						fingers[threadIdx.x].docId = NO_MORE_DOC;
						fingers[threadIdx.x].position = NO_VALID_POSITION;
					}else{
						fingers[threadIdx.x].docId = iDocIdList[fingers[threadIdx.x].position];
//			 			if(fingers[threadIdx.x].docId > limitDoc.secondMaxDocId){
//			 				fingers[threadIdx.x].docId = NO_MORE_DOC;
//			 				fingers[threadIdx.x].position = NO_VALID_POSITION;
//			 			}
					}
				}
			}
		}
		else{
			int pivotDoc = docCurrent;
			long long position;
			int docLocal;
			int idTerm;
			for (int j = 0; j < sharedPivot.positionInOrderedList; ++j) {
				idTerm = iOrderedTermSharedList[j];

				if(fingers[idTerm].docId == fingers[sharedPivot.idTerm].docId)//Até alcançar um finger q aponte a um documento pivo
					break;

				fingers[idTerm].docId = NO_MORE_DOC;
				position = fingers[idTerm].position + 1 + threadIdx.x;
				docLocal = -1;
				while(position < (iSharedPositionInitialInList[idTerm]+iDocNumberByTermList[idTerm])
						&& docLocal < pivotDoc){
					docLocal = iDocIdList[position];
					position += blockDim.x;

				}
				position -= blockDim.x;

				if(docLocal < pivotDoc || position >= (iSharedPositionInitialInList[idTerm]+iDocNumberByTermList[idTerm])){
					docLocal = NO_MORE_DOC;
					position = NO_VALID_POSITION;
				}

//				atomicMin(&(fingers[idTerm].docId) , docLocal);

				int docNeighbor, docAux = docLocal;
				for (int i = 16; i >= 1; i /= 2) {
					docNeighbor  = __shfl_down_sync(0xFFFFFFFF,docAux, i);

					if(docNeighbor < docAux)
						docAux = docNeighbor;
				}

				if( ((threadIdx.x & 0x1f) == 0)){
					atomicMin(&(fingers[idTerm].docId) , docAux);
				}


				__syncthreads();

				if(fingers[idTerm].docId == docLocal){
					fingers[idTerm].position = position;
				}
			}
		}

		__syncthreads();

		//Sort the terms in non decreasing order of DID
		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

		//Select term pivot
		if(THREAD_MASTER){
			selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
			score = 0.0;
		}
		__syncthreads();
	}

//	sortLocalTopkDocAndStoreInGlobal((float*)&(dTopkScoreListGlobal[blockIdx.x*iTopK]),(int*)&(iTopkDocListGlobal[blockIdx.x*iTopK]),iTopK,&documentTopk);

	int gIndex = blockIdx.x * iTopK + threadIdx.x;
	for (int localIndex = threadIdx.x; localIndex < iTopK; localIndex+=blockDim.x) {
		dTopkScoreListGlobal[gIndex] = documentTopk.score[localIndex];
		iTopkDocListGlobal[gIndex] = documentTopk.id[localIndex];

		gIndex+=blockDim.x;
	}
}

__global__ void matchWandParallel_VARIABLE_Batch_Block_2(const int* iDocIdList, const unsigned short int* iFreqList,
														const float *dUBlistGlobal, const float *dIdfListGlobal, const int *iDocLenghtList,
														const short int* iTermNumberByQuery, int *iTopkDocListGlobal,
														float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
														const short int iTopK, const float iInitialThreshold,
														const int* iQueryTerms, const long long* ptrInitPostingList,
														int* ptrQueryPositions,int *iDocNumberByTermListGlobal){
//
//	if(blockIdx.x!=1)
//		return;


	__shared__ int queryPosition;
	__shared__ pivot sharedPivot;
	__shared__ finger fingers[TERM_NUMBER];

	__shared__ documentTopkList documentTopk;

	__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];
	__shared__ long long iSharedPositionInitialInList[TERM_NUMBER];
	__shared__ float dUBlist[TERM_NUMBER];
	__shared__ float dIdfList[TERM_NUMBER];
	__shared__ float dAverageDocumentLength;

	__shared__ int iDocNumberByTermList[TERM_NUMBER];

	__shared__ float score;
	__shared__ bool isValidCandidate;
	__shared__ int docCurrent;
	__shared__ limitDocId limitDoc;
	__shared__ short int iTermNumber;

	int padding;

 	float thresholdLocal;// = iInitialThreshold;
 	thresholdLocal = iInitialThreshold;

	int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
	int localIndex;

//	int count = 0;

	if(THREAD_MASTER){
		documentTopk.padding = iTopK;
		dAverageDocumentLength = dAverageDocumentLengthGlobal;
//	}else if(THREAD_MASTER_2){
//		limitDoc.minDocId = -1;
		limitDoc.secondMaxDocId = -1;

		iTermNumber = iTermNumberByQuery[blockIdx.x];
	}

	__syncthreads();

	if(threadIdx.x < iTermNumber){
		queryPosition = ptrQueryPositions[blockIdx.x];
		int idTerm = iQueryTerms[queryPosition + threadIdx.x];

		iDocNumberByTermList[threadIdx.x] = iDocNumberByTermListGlobal[idTerm];

		dUBlist[threadIdx.x] = dUBlistGlobal[idTerm];//[threadIdx.x];
		dIdfList[threadIdx.x] = dIdfListGlobal[idTerm];//[threadIdx.x];

//		printf(" %.2f ",dUBlist[threadIdx.x]);
		iSharedPositionInitialInList[threadIdx.x] = ptrInitPostingList[idTerm];
	}

	//Inicializa a lista de Score e Documentos dos Topk
	//Considero que o Top_K seja um número múltiplo do tamanho do bloco
	#pragma unroll 4
	for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
		documentTopk.id[localIndex] = -1;
		documentTopk.score[localIndex] = 0.0;
	}

	//Define o max e o min
	if(threadIdx.x < iTermNumber){
		int docAmount = iDocNumberByTermList[threadIdx.x];
//		fingers[threadIdx.x].final = 0;
//		limitDoc.extraPosition[threadIdx.x] = 0;

		globalIndex = 0;

		int maxDoc;

		if(THREAD_MASTER) limitDoc.minDocId = 0;

		globalIndex = docAmount-1;

		maxDoc =  iDocIdList[iSharedPositionInitialInList[threadIdx.x] +  globalIndex];

		atomicMax(&(limitDoc.secondMaxDocId), maxDoc);

		fingers[threadIdx.x].docId = iDocIdList[iSharedPositionInitialInList[threadIdx.x]];
		fingers[threadIdx.x].position = iSharedPositionInitialInList[threadIdx.x];
	}

	__syncthreads();


	sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

	if(THREAD_MASTER){
		selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
		docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
//	}else if(THREAD_MASTER_2){
		score = 0.0;
	}

	__syncthreads();

	while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){

		isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);
//		count++;
//
//		if(fingers[sharedPivot.idTerm].docId == 38182)
//			printf("Oi");

		if(isValidCandidate){
			if(threadIdx.x < iTermNumber){
		 		int termId = iOrderedTermSharedList[threadIdx.x];
		 		float scoreL = 0.0;
		 		if(fingers[termId].docId == fingers[sharedPivot.idTerm].docId){
		 			scoreL = scoreTf_Idf(iFreqList[fingers[termId].position],
										iDocLenghtList[fingers[termId].position],
										dIdfList[termId],dAverageDocumentLength,1.0);
		 		}

		 		float aux = 0;
		 		for (int i = 0; i < TERM_NUMBER; ++i) {
		 			aux += __shfl_sync(0xFFFFFFFF,scoreL,i);
		 		}

		 		if(THREAD_MASTER) score = aux;
//		 		atomicAdd(&score,scoreL);
			}

			padding = documentTopk.padding;

			__syncthreads();

/*				If the heap is not full
			the candidate is inserted into the heap. If the heap is full
			and the new score is larger than the minimum score in the
			heap, the new document is inserted into the heap, replacing
			the one with the minimum score.

*/
			if(padding != 0 || thresholdLocal < score ){
				thresholdLocal = managerMinValue_v5(&documentTopk, fingers[sharedPivot.idTerm].docId, score,padding);
			}

			if(threadIdx.x < iTermNumber ){
			 	int docPivot = fingers[sharedPivot.idTerm].docId;
			 	if(fingers[threadIdx.x].docId ==  docPivot){
			 		fingers[threadIdx.x].position++;
			 		if(fingers[threadIdx.x].position >= (iDocNumberByTermList[threadIdx.x]+iSharedPositionInitialInList[threadIdx.x])){//Não Válido
			 			fingers[threadIdx.x].docId = NO_MORE_DOC;
			 			fingers[threadIdx.x].position = NO_VALID_POSITION;
			 		}else{
			 			fingers[threadIdx.x].docId = iDocIdList[fingers[threadIdx.x].position];
//			 			if(fingers[threadIdx.x].docId > limitDoc.secondMaxDocId){
//			 				fingers[threadIdx.x].docId = NO_MORE_DOC;
//			 				fingers[threadIdx.x].position = NO_VALID_POSITION;
//			 			}
			 		}
			 	}
			}
		}
		else{
			int pivotDoc = docCurrent;
			long long position;
			int docLocal;
			int idTerm;
			for (int j = 0; j < sharedPivot.positionInOrderedList; ++j) {
				idTerm = iOrderedTermSharedList[j];

				if(fingers[idTerm].docId == fingers[sharedPivot.idTerm].docId)//Até alcançar um finger q aponte a um documento pivo
					break;

				fingers[idTerm].docId = NO_MORE_DOC;
				position = fingers[idTerm].position + 1 + threadIdx.x;
				docLocal = -1;
				while(position < (iSharedPositionInitialInList[idTerm]+iDocNumberByTermList[idTerm])
						&& docLocal < pivotDoc){
					docLocal = iDocIdList[position];
					position += blockDim.x;

				}
				position -= blockDim.x;

				if(docLocal < pivotDoc || position >= (iSharedPositionInitialInList[idTerm]+iDocNumberByTermList[idTerm])){
					docLocal = NO_MORE_DOC;
					position = NO_VALID_POSITION;
				}

//				atomicMin(&(fingers[idTerm].docId) , docLocal);

				int docNeighbor, docAux = docLocal;
				for (int i = 16; i >= 1; i /= 2) {
					docNeighbor  = __shfl_down_sync(0xFFFFFFFF,docAux, i);

					if(docNeighbor < docAux)
						docAux = docNeighbor;
				}

				if( ((threadIdx.x & 0x1f) == 0)){
					atomicMin(&(fingers[idTerm].docId) , docAux);
				}


				__syncthreads();

				if(fingers[idTerm].docId == docLocal){
					fingers[idTerm].position = position;
				}
			}
		}

		__syncthreads();

		//Sort the terms in non decreasing order of DID
		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

		//Select term pivot
		if(THREAD_MASTER){
			selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
			score = 0.0;
		}
		__syncthreads();
	}

//	sortLocalTopkDocAndStoreInGlobal((float*)&(dTopkScoreListGlobal[blockIdx.x*iTopK]),(int*)&(iTopkDocListGlobal[blockIdx.x*iTopK]),iTopK,&documentTopk);

	int gIndex = blockIdx.x * iTopK + threadIdx.x;
	for (int localIndex = threadIdx.x; localIndex < iTopK; localIndex+=blockDim.x) {
		dTopkScoreListGlobal[gIndex] = documentTopk.score[localIndex];
		iTopkDocListGlobal[gIndex] = documentTopk.id[localIndex];

		gIndex+=blockDim.x;
	}

//		if(THREAD_MASTER)
////			atomicAdd(&globalCount,count);
//
//		if(THREAD_MASTER)
//			printf("-----%d %d----", blockIdx.x, count);

}


__global__ void matchWandParallel_VARIABLE_Batch_Block(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlist, const float *dIdfList, const int *iDocLenghtList,
										  const short int *iTermNumberByQuery, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
										  const int iBlockRoundNumber, const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* iQueryTerms, const long long* ptrPostingPositions,
										  int* ptrQueryPositions, int *iDocNumberByTermList){

	if(blockIdx.x != 4999)
		return;

	__shared__ short int iTermNumber;

	__shared__ pivot sharedPivot;
	__shared__ finger fingers[TERM_NUMBER];

	__shared__ postingList postingLists[TERM_NUMBER];
	__shared__ documentTopkList documentTopk;

	__shared__ long long ptrPostingPositionShared[TERM_NUMBER];
	__shared__ int iDocNumberByTermListShared[TERM_NUMBER];
	__shared__ int queryPosition;

	__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];

//	__shared__ int iGlobalInitialPositionInList;
	__shared__ unsigned short int iElementQuantityByBlock;

	__shared__ float score;
	__shared__ bool isValidCandidate;

//	__shared__ short int needSearchDocRange[TERM_NUMBER];
	__shared__ limitDocId limitDoc;

 	float thresholdLocal;// = iInitialThreshold;

 	thresholdLocal = iInitialThreshold;

	int globalIndex = 0;// = iInitialPositionGlobal + threadIdx.x;
	int localIndex;
//	long long positionInitialInTermPostingList;//int positionInitialInTermPostingList;

	if(thresholdLocal < globalThreshold)
		thresholdLocal = globalThreshold;

	if(THREAD_MASTER){
		documentTopk.padding = iTopK;
		iTermNumber = iTermNumberByQuery[blockDim.x];
	}else if(THREAD_MASTER_2){
		iElementQuantityByBlock = DOC_QUANTITY_IN_MEMORY;//iBlockRoundNumber * DOC_QUANTITY_IN_MEMORY;
//		iGlobalInitialPositionInList = 0;//iElementQuantityByBlock  * blockIdx.x * iGlobalRoundNumber;
	}

	//Inicializa a lista de Score e Documentos dos Topk
	//Considero que o Top_K seja um número múltiplo do tamanho do bloco
	for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
		documentTopk.id[localIndex] = -1;
		documentTopk.score[localIndex] = 0.0;
	}

//	if(THREAD_MASTER) documentTopk.padding = iTopK;

	__syncthreads();

	//Define o max e o min
	if(threadIdx.x < iTermNumber){
		limitDoc.extraPosition[threadIdx.x] = 0;
		queryPosition = ptrQueryPositions[blockDim.x];

		iDocNumberByTermListShared[threadIdx.x] = iDocNumberByTermList[iQueryTerms[queryPosition + threadIdx.x]];
		ptrPostingPositionShared[threadIdx.x] = ptrPostingPositions[iQueryTerms[queryPosition+threadIdx.x]];

		int docAmount = iDocNumberByTermListShared[threadIdx.x];//iDocNumberByTermList[threadIdx.x];
//		globalIndex = iGlobalInitialPositionInList;

		int aux, maxDoc;
		int maxNeighbor;

		if(THREAD_MASTER) limitDoc.minDocId = 0;

//		int isTail = globalIndex < docAmount;
//		globalIndex += iElementQuantityByBlock * iGlobalRoundNumber;
//		isTail &= globalIndex >= docAmount;

		globalIndex = docAmount - 1;
//		int isTail = iElementQuantityByBlock >= docAmount;
//
//		if(isTail){
//			globalIndex = iGlobalInitialPositionInList + (docAmount - iGlobalInitialPositionInList - 1);
//		}

//		maxDoc =  (isTail || globalIndex < docAmount) ? iDocIdList[ptrPostingPositionShared[threadIdx.x] +  globalIndex] - 1 :
//														-1;
		maxDoc =  iDocIdList[ptrPostingPositionShared[threadIdx.x] +  globalIndex];

		aux = maxDoc;
		for (int i = 1; i < iTermNumber; ++i) {
			maxNeighbor = __shfl_sync(0xFFFFFFFF,aux,i);
			if(maxNeighbor > maxDoc)
				maxDoc = maxNeighbor;
		}
		if(THREAD_MASTER) limitDoc.secondMaxDocId = maxDoc;
	}

//	__syncthreads();
//
//	//Busca faixa de documentos;
//	for (int internTermId = 0; internTermId < iTermNumber; ++internTermId) {
//		if(needSearchDocRange[internTermId])
//			searchRangeOfDocs_batch(iDocIdList,postingLists, internTermId, iGlobalInitialPositionInList,
//									&limitDoc,iElementQuantityByBlock,iGlobalRoundNumber,
//									iDocNumberByTermListShared[internTermId], ptrPostingPositionShared[internTermId]);
//	}

	__syncthreads();

	//Preenche a memória compartilhada
//	positionInitialInTermPostingList = 0;
	int docLocal, docAmount;
	for (int termId = 0; termId < iTermNumber; ++termId) {
//		globalIndex = iGlobalInitialPositionInList + limitDoc.extraPosition[termId] + threadIdx.x;
		globalIndex = threadIdx.x;

		docAmount = iDocNumberByTermListShared[termId];
		docLocal = -1;

		for (localIndex = threadIdx.x; localIndex < iElementQuantityByBlock; localIndex+=blockDim.x) {

			docLocal = (globalIndex < docAmount) ? iDocIdList[ptrPostingPositionShared[termId] + globalIndex]
			                                                  : NO_MORE_DOC;

			if(globalIndex > docAmount){
				postingLists[termId].docId[localIndex] = NO_MORE_DOC;
				fingers[termId].final = 1;
				break;
			}

			postingLists[termId].docId[localIndex] = docLocal;
			postingLists[termId].docLenght[localIndex] = iDocLenghtList[ptrPostingPositionShared[termId] + globalIndex];
			postingLists[termId].freq[localIndex] = iFreqList[ptrPostingPositionShared[termId] + globalIndex];

			globalIndex += blockDim.x;
		}

//		positionInitialInTermPostingList += iDocNumberByTermList[termId];
	}

//	__syncthreads();

	if(threadIdx.x < iTermNumber){
		fingers[threadIdx.x].docId = postingLists[threadIdx.x].docId[0];
		fingers[threadIdx.x].position = (fingers[threadIdx.x].docId == NO_MORE_DOC) ? NO_VALID_POSITION : 0;
		fingers[threadIdx.x].final = (fingers[threadIdx.x].final == 1) ? 1 : 0;
	}

	__syncthreads();

	__shared__ int docCurrent;

	sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

//	__syncthreads();

	if(THREAD_MASTER){
		selectTermPivot_2(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
		docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
	}else if(THREAD_MASTER_2){
		score = 0.0;
	}

	int padding;
	int threadIdInWarp = (threadIdx.x & 0x1f);
	int idWarp = ((blockDim.x >> 5) == 1 ) ? 1 :  threadIdx.x >> 5;

	__syncthreads();

	while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){

		if(THREAD_MASTER)
			isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);

		__syncthreads();

		if(isValidCandidate){

			if(threadIdx.x < iTermNumber){
				fullScore_3_1(&score, fingers[sharedPivot.idTerm].docId, iOrderedTermSharedList,
								fingers,postingLists, dIdfList, dAverageDocumentLength);
			}

			padding = documentTopk.padding;

			__syncthreads();

			if(thresholdLocal < score){
				thresholdLocal = managerMinValue_v5(&documentTopk, docCurrent, score,padding);
			}

			if(idWarp == 1 && threadIdInWarp < iTermNumber ){
				advancePivoTermFinger_4(sharedPivot,fingers, postingLists,iElementQuantityByBlock,threadIdInWarp);
			}
		}
		else{
			 advanceDocIdOfPredecessorTerm_4(postingLists,
										   	 iOrderedTermSharedList,
										   	 fingers,sharedPivot,fingers[sharedPivot.idTerm].docId,
										   	 iElementQuantityByBlock);
		}

		__syncthreads();

		for (int termId = 0; termId < iTermNumber; ++termId) {
			if(fingers[termId].docId == NO_MORE_DOC && 	fingers[termId].final == 0){

//				searchMoreDocs_batch(iDocIdList,iFreqList,iDocLenghtList,postingLists,
//							  	  	 termId,iGlobalInitialPositionInList,&limitDoc,
//							  	  	 iElementQuantityByBlock,&(fingers[termId]),docCurrent,
//							  	  	 iDocNumberByTermListShared[termId],ptrPostingPositionShared[termId]);

				searchMoreDocs_batch(iDocIdList,iFreqList,iDocLenghtList,postingLists,
									 termId, 0, &limitDoc,
									 iElementQuantityByBlock,&(fingers[termId]),docCurrent,
									 iDocNumberByTermListShared[termId],ptrPostingPositionShared[termId]);
//
//
//////
//					if(THREAD_MASTER && (documentTopk.padding < (iTopK >> 1)) && thresholdLocal > thresholdGlobal){
////					if(THREAD_MASTER && documentTopk.padding == 0 && thresholdLocal > thresholdGlobal){
//						atomicMaxD(&thresholdGlobal,thresholdLocal);
//					}
//
//					if((documentTopk.padding < (iTopK >> 1)))
//					if(thresholdLocal < thresholdGlobal){
//						thresholdLocal = thresholdGlobal;
//					}

			}
		}


//		__syncthreads();//Talvez não precise
		//Sort the terms in non decreasing order of DID
		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

//		__syncthreads();//Talvez não precise

		//Select term pivot
		if(THREAD_MASTER){
			selectTermPivot_2(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
			score = 0.0;
		}
		__syncthreads();
	}

//	if(THREAD_MASTER){
//	int max = iTopK - documentTopk.padding;
//	int i = 0;
//	while(i > max){
//		if(2*i+2 < max)
//			if(documentTopk.score[i] > documentTopk.score[2*i+2])
//				printf("ERRADO!!!\n");
//
//		if(2*i+1 < max)
//			if(documentTopk.score[i] > documentTopk.score[2*i+1])
//				printf("ERRADO!!!\n");
//
//		i++;
//	}}
//
	__syncthreads();

	sortLocalTopkDocAndStoreInGlobal_BLOCK(dTopkScoreListGlobal,iTopkDocListGlobal,iTopK,&documentTopk);
//	globalIndex =  iTopK * blockIdx.x + threadIdx.x + documentTopk.padding;
//	for (localIndex = threadIdx.x; localIndex < (iTopK - documentTopk.padding) ; localIndex += blockDim.x) {
//		iTopkDocListGlobal[globalIndex]   = documentTopk.id[localIndex];
//		dTopkScoreListGlobal[globalIndex] = documentTopk.score[localIndex];
//		globalIndex += blockDim.x;
//	}
//	__syncthreads();



}


__global__ void matchWandParallel_BATCH(const int* iDocIdList, const unsigned short int* iFreqList,
									    const float *dUBlist, const float *dIdfList, const int *iDocLenghtList,
										const short int iTermNumber, int *iTopkDocListGlobal,
										float *dTopkScoreListGlobal, const float dAverageDocumentLength,
										const int iBlockRoundNumber, const int iGlobalRoundNumber,
										const short int iTopK, const float iInitialThreshold,
										const int* iQueryTerms, const long long* ptrInitPostingList,
										int* ptrQueryPositions, int idQuery,int *iDocNumberByTermList){


	__shared__ pivot sharedPivot;
	__shared__ finger fingers[TERM_NUMBER];

	__shared__ postingList postingLists[TERM_NUMBER];
	__shared__ documentTopkList documentTopk;

	__shared__ long long ptrInitPostingListShared[TERM_NUMBER];
	__shared__ int iDocNumberByTermListShared[TERM_NUMBER];
	__shared__ int queryPosition;

	__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];

	__shared__ int iGlobalInitialPositionInList;
	__shared__ unsigned short int iElementQuantityByBlock;

	__shared__ float score;
	__shared__ bool isValidCandidate;

	__shared__ short int needSearchDocRange[TERM_NUMBER];
	__shared__ limitDocId limitDoc;

 	float thresholdLocal;// = iInitialThreshold;

 	thresholdLocal = iInitialThreshold;

	int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
	int localIndex;
//	long long positionInitialInTermPostingList;//int positionInitialInTermPostingList;

	if(thresholdLocal < globalThreshold)
		thresholdLocal = globalThreshold;

	if(THREAD_MASTER){
		documentTopk.padding = iTopK;
	}else if(THREAD_MASTER_2){
		iElementQuantityByBlock = DOC_QUANTITY_IN_MEMORY;//iBlockRoundNumber * DOC_QUANTITY_IN_MEMORY;
		iGlobalInitialPositionInList = iElementQuantityByBlock  * blockIdx.x * iGlobalRoundNumber;
	}

	#ifdef DEBUG
		if(THREAD_MASTER_2)
			if(iGlobalInitialPositionInList < 0)
				printf("Opa!!!!");
	#endif

	//Inicializa a lista de Score e Documentos dos Topk
	//Considero que o Top_K seja um número múltiplo do tamanho do bloco
	for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
		documentTopk.id[localIndex] = -1;
		documentTopk.score[localIndex] = 0.0;
	}

//	if(THREAD_MASTER) documentTopk.padding = iTopK;

//	__syncthreads();

	//Define o max e o min
	if(threadIdx.x < iTermNumber){
		limitDoc.extraPosition[threadIdx.x] = 0;
		queryPosition = ptrQueryPositions[idQuery];
		iDocNumberByTermListShared[threadIdx.x] = iDocNumberByTermList[iQueryTerms[queryPosition + threadIdx.x]];
		ptrInitPostingListShared[threadIdx.x] = ptrInitPostingList[iQueryTerms[queryPosition + threadIdx.x]];

		int docAmount = iDocNumberByTermListShared[threadIdx.x];//iDocNumberByTermList[threadIdx.x];
		globalIndex = iGlobalInitialPositionInList;
//		positionInitialInTermPostingList = 0;
//
//		for (int i = 0; i < threadIdx.x; ++i) {
//			positionInitialInTermPostingList += iDocNumberByTermList[iQueryTerms[i]];//iDocNumberByTermList[i];
//		}

//		positionInitialInTermPostingList = ptrPostingPositionShared[threadIdx.x];

		int aux, maxDoc;
		int maxNeighbor;
		if(blockIdx.x != 0){
			int maxDoc = (globalIndex < docAmount) ? iDocIdList[ptrInitPostingListShared[threadIdx.x] + globalIndex - 1] : -1;
			maxDoc++;
			aux = maxDoc;

			atomicMax(&limitDoc.minDocId, maxDoc);

//			for (int i = 1; i < iTermNumber; ++i) {
//				maxNeighbor = __shfl(aux,i);
//				if(maxNeighbor > maxDoc)
//					maxDoc = maxNeighbor;
//			}
//
//			if(THREAD_MASTER) limitDoc.minDocId = maxDoc; //atomicExch(&(limitDoc.minDocId), maxDoc);

			if(aux < limitDoc.minDocId && aux != 0){
				needSearchDocRange[threadIdx.x] = 1;
				limitDoc.extraPosition[threadIdx.x] = NO_MORE_DOC;
			}
		}else
			if(THREAD_MASTER) limitDoc.minDocId = 0;

		int isTail = globalIndex < docAmount;
		globalIndex += iElementQuantityByBlock * iGlobalRoundNumber;
		isTail &= globalIndex >= docAmount;

		if(isTail){
			globalIndex = iGlobalInitialPositionInList + (docAmount - iGlobalInitialPositionInList - 1);
		}

		maxDoc =  (isTail || globalIndex < docAmount) ? iDocIdList[ptrInitPostingListShared[threadIdx.x] +  globalIndex] :
														-1;
		aux = maxDoc;
		for (int i = 1; i < iTermNumber; ++i) {
			maxNeighbor = __shfl_down_sync(0xFFFFFFFF,aux,i);
			if(maxNeighbor > maxDoc)
				maxDoc = maxNeighbor;
		}
		if(THREAD_MASTER) limitDoc.secondMaxDocId = maxDoc;
	}

	__syncthreads();

	//Busca faixa de documentos;
	for (int internTermId = 0; internTermId < iTermNumber; ++internTermId) {
		if(needSearchDocRange[internTermId])
			searchRangeOfDocs_batch(iDocIdList,postingLists, internTermId, iGlobalInitialPositionInList,
									&limitDoc,iElementQuantityByBlock,iGlobalRoundNumber,
									iDocNumberByTermListShared[internTermId],
									ptrInitPostingListShared[internTermId]);
	}

	__syncthreads();

	//Preenche a memória compartilhada
//	positionInitialInTermPostingList = 0;
	int docLocal, docAmount;
	for (int termId = 0; termId < iTermNumber; ++termId) {
		globalIndex = iGlobalInitialPositionInList + limitDoc.extraPosition[termId] + threadIdx.x;
		docAmount = iDocNumberByTermListShared[termId];
		docLocal = -1;

		for (localIndex = threadIdx.x; localIndex < iElementQuantityByBlock; localIndex+=blockDim.x) {

			docLocal = (globalIndex < docAmount) ? iDocIdList[ptrInitPostingListShared[termId] + globalIndex]
			                                     : NO_MORE_DOC;

			if(docLocal > limitDoc.secondMaxDocId || globalIndex > docAmount){
				postingLists[termId].docId[localIndex] = NO_MORE_DOC;
				fingers[termId].final = 1;
				break;
			}

			postingLists[termId].docId[localIndex] = docLocal;
			postingLists[termId].docLenght[localIndex] = iDocLenghtList[ptrInitPostingListShared[termId] + globalIndex];
			postingLists[termId].freq[localIndex] = iFreqList[ptrInitPostingListShared[termId] + globalIndex];

			globalIndex += blockDim.x;
		}

//		positionInitialInTermPostingList += iDocNumberByTermList[termId];
	}

//	__syncthreads();

	if(threadIdx.x < iTermNumber){
		fingers[threadIdx.x].docId = postingLists[threadIdx.x].docId[0];
		fingers[threadIdx.x].position = (fingers[threadIdx.x].docId == NO_MORE_DOC) ? NO_VALID_POSITION : 0;
		fingers[threadIdx.x].final = 0;
	}

	__syncthreads();

	__shared__ int docCurrent;

	sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

//	__syncthreads();

	if(THREAD_MASTER){
		selectTermPivot_2(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
		docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
	}else if(THREAD_MASTER_2){
		score = 0.0;
	}

	int padding;
	int threadIdInWarp = (threadIdx.x & 0x1f);
	int idWarp = ((blockDim.x >> 5) == 1 ) ? 1 :  threadIdx.x >> 5;

	__syncthreads();

	while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){

		if(THREAD_MASTER)
			isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);

		__syncthreads();

		if(isValidCandidate){

			if(threadIdx.x < iTermNumber){
				fullScore_3_1(&score, fingers[sharedPivot.idTerm].docId, iOrderedTermSharedList,
								fingers,postingLists, dIdfList, dAverageDocumentLength);
			}

			padding = documentTopk.padding;

			__syncthreads();

			if(thresholdLocal < score){
				thresholdLocal = managerMinValue_v5(&documentTopk, docCurrent, score,padding);
			}

			if(idWarp == 1 && threadIdInWarp < iTermNumber ){
				advancePivoTermFinger_4(sharedPivot,fingers, postingLists,iElementQuantityByBlock,threadIdInWarp);
			}
		}
		else{
			 advanceDocIdOfPredecessorTerm_4(postingLists,
										   iOrderedTermSharedList,
										   fingers,sharedPivot,fingers[sharedPivot.idTerm].docId,
										   iElementQuantityByBlock);
		}

		__syncthreads();

		for (int termId = 0; termId < iTermNumber; ++termId) {
			if(fingers[termId].docId == NO_MORE_DOC && 	fingers[termId].final == 0){

				searchMoreDocs_batch(iDocIdList,iFreqList,iDocLenghtList,postingLists,
							  	  	 termId,iGlobalInitialPositionInList,&limitDoc,
							  	  	 iElementQuantityByBlock,&(fingers[termId]),docCurrent,
							  	  	 iDocNumberByTermListShared[termId],ptrInitPostingListShared[termId]);

//
//
//////
//					if(THREAD_MASTER && (documentTopk.padding < (iTopK >> 1)) && thresholdLocal > thresholdGlobal){
////					if(THREAD_MASTER && documentTopk.padding == 0 && thresholdLocal > thresholdGlobal){
//						atomicMaxD(&thresholdGlobal,thresholdLocal);
//					}
//
//					if((documentTopk.padding < (iTopK >> 1)))
//					if(thresholdLocal < thresholdGlobal){
//						thresholdLocal = thresholdGlobal;
//					}

			}
		}


//		__syncthreads();//Talvez não precise
		//Sort the terms in non decreasing order of DID
		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

//		__syncthreads();//Talvez não precise

		//Select term pivot
		if(THREAD_MASTER){
			selectTermPivot_2(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
			score = 0.0;
		}
		__syncthreads();
	}

//	if(THREAD_MASTER){
//	int max = iTopK - documentTopk.padding;
//	int i = 0;
//	while(i > max){
//		if(2*i+2 < max)
//			if(documentTopk.score[i] > documentTopk.score[2*i+2])
//				printf("ERRADO!!!\n");
//
//		if(2*i+1 < max)
//			if(documentTopk.score[i] > documentTopk.score[2*i+1])
//				printf("ERRADO!!!\n");
//
//		i++;
//	}}
//
//	__syncthreads();

	sortLocalTopkDocAndStoreInGlobal(dTopkScoreListGlobal,iTopkDocListGlobal,iTopK,&documentTopk);
//	globalIndex =  iTopK * blockIdx.x + threadIdx.x + documentTopk.padding;
//	for (localIndex = threadIdx.x; localIndex < (iTopK - documentTopk.padding) ; localIndex += blockDim.x) {
//		iTopkDocListGlobal[globalIndex]   = documentTopk.id[localIndex];
//		dTopkScoreListGlobal[globalIndex] = documentTopk.score[localIndex];
//		globalIndex += blockDim.x;
//	}
//	__syncthreads();

}

__global__ void preProcessingWand(const int* iDocIdList,
								  const short int iTermNumber,
								  const int* iDocNumberByTermList,
								  const int* iInitialPositionPostingList,
								  const int docIdNumberByBlock,
								  int* extraPositions, int* docMaxList){

	__shared__ int iGlobalInitialPositionInList;
	int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
//	int positionInitialInTermPostingList;

	__shared__ int sharedMinDoc;
	__shared__ int sharedMaxDoc;
	__shared__ int sharedExtraPositions[TERM_NUMBER];
	__shared__ int sharedInitialDocId[TERM_NUMBER];
	__shared__ int sharedDocNumberByList[TERM_NUMBER];

	if(THREAD_MASTER){
		iGlobalInitialPositionInList = docIdNumberByBlock  * blockIdx.x;
	}

//	__syncthreads();

	if(threadIdx.x < iTermNumber){
		sharedDocNumberByList[threadIdx.x] = iDocNumberByTermList[threadIdx.x];
		int docAmount = sharedDocNumberByList[threadIdx.x];
		globalIndex = iGlobalInitialPositionInList;
//		positionInitialInTermPostingList = iInitialPositionPostingList[threadIdx.x];

		int maxDoc;
//		int aux, maxDoc;
//		int maxNeighbor;
		if(blockIdx.x != 0){
			int maxDoc = (globalIndex < docAmount) ? iDocIdList[iInitialPositionPostingList[threadIdx.x] + globalIndex - 1] : -1;
			maxDoc++;
			sharedInitialDocId[threadIdx.x] = maxDoc;
//			aux = maxDoc;
//			for (int i = iTermNumber-1; i > 0; --i) {
//				maxNeighbor = __shfl(aux,i);
//				if(maxNeighbor > maxDoc) maxDoc = maxNeighbor;
//			}
			atomicMax(&sharedMinDoc, maxDoc);
//			if(THREAD_MASTER) sharedMinDoc = maxDoc;

//			if(aux < limitDoc.minDocId && aux != 0)
//				needSearchDocRange[threadIdx.x] = 1;
		}else
			sharedMinDoc = 0;

		int isTail = globalIndex < docAmount;
		globalIndex += docIdNumberByBlock;
		isTail &= globalIndex >= docAmount;

		if(isTail){
			globalIndex = iGlobalInitialPositionInList + (docAmount - iGlobalInitialPositionInList - 1);
		}

		maxDoc =  (isTail || globalIndex < docAmount) ? (iDocIdList[iInitialPositionPostingList[threadIdx.x] +  globalIndex]-1) : -1;
//		aux = maxDoc;
//		for (int i = 1; i < iTermNumber; ++i) {
//			maxNeighbor = __shfl(aux,i);
//			if(maxNeighbor > maxDoc)
//				maxDoc = maxNeighbor;
//		}
//		if(THREAD_MASTER) sharedMaxDoc = maxDoc;

		atomicMax(&sharedMaxDoc, maxDoc);
	}

	__syncthreads();

	for (int iTerm = 0; iTerm < iTermNumber; ++iTerm) {
		if(sharedInitialDocId[iTerm] < sharedMinDoc){
			globalIndex = iInitialPositionPostingList[iTerm] + iGlobalInitialPositionInList + threadIdx.x;

			int docLocal = -1;
			while (docLocal < sharedMinDoc && globalIndex < sharedDocNumberByList[iTerm]){
				docLocal = iDocIdList[globalIndex];
				globalIndex += blockDim.x;
			}
			globalIndex-= blockDim.x;

			long long int initialPosition;
			if(docLocal < sharedMinDoc)//Caso não encontre
				initialPosition = NO_VALID_POSITION;
			else
				initialPosition = globalIndex - iGlobalInitialPositionInList - iInitialPositionPostingList[iTerm];

			int positionNeighbor;
			for (int i = 16; i >= 1; i /= 2) {
				positionNeighbor  = __shfl_down_sync(0xFFFFFFFF,initialPosition, i);

				if(positionNeighbor < initialPosition)
					initialPosition = positionNeighbor;
			}

			if( ((threadIdx.x & 0x1f) == 0) && initialPosition != NO_MORE_DOC){
				atomicMin(&sharedExtraPositions[iTerm] , initialPosition);
			}
			//__syncthreads();


//			if(THREAD_MASTER){
//				globalIndex = iGlobalInitialPositionInList + iElementQuantityByBlock * roundGlobalNumber + threadIdx.x;
//				globalIndex += limitDoc->extraPosition[termId];
//			//
//				if(globalIndex < iDocNumberByTermList[termId]){
//					if(limitDoc->secondMaxDocId < iDocIdList[positionInListGlobal + globalIndex] -1)
//						limitDoc->secondMaxDocId = iDocIdList[positionInListGlobal + globalIndex]-1;
//				}
//			}
		}

		if(threadIdx.x < iTermNumber){
			extraPositions[iTermNumber*blockIdx.x + threadIdx.x] = sharedExtraPositions[threadIdx.x];

			if(THREAD_MASTER) docMaxList[blockIdx.x] = sharedMaxDoc;
		}
	}
}

__global__ void matchWandParallel_VARIABLE_3_Teste(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlist, const float *dIdfList,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
										  const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* d_iDocNumberByTermList,
										  const int* extraPositions, const int* docMaxList){

		__shared__ pivot sharedPivot;
		__shared__ finger fingers[TERM_NUMBER];

		__shared__ postingList postingLists[TERM_NUMBER];
		__shared__ documentTopkList documentTopk;

		__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];

		__shared__ int iGlobalInitialPositionInList;
		__shared__ unsigned short int iElementQuantityByBlock;

		__shared__ float score;
		__shared__ bool isValidCandidate;

//		__shared__ short int needSearchDocRange[TERM_NUMBER];
		__shared__ limitDocId limitDoc;

//		int count = 0;
		float thresholdLocal;// = iInitialThreshold;

		thresholdLocal = iInitialThreshold;

		int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
		int localIndex;
		int positionInitialInTermPostingList;

		if(thresholdLocal < globalThreshold)
			thresholdLocal = globalThreshold;

		if(threadIdx.x < iTermNumber){
			limitDoc.extraPosition[threadIdx.x] = extraPositions[blockIdx.x*iTermNumber + threadIdx.x];
			fingers[threadIdx.x].final = 0;
		}

		if(THREAD_MASTER){
			documentTopk.padding = iTopK;
			limitDoc.secondMaxDocId = docMaxList[blockIdx.x];
		}else if(THREAD_MASTER_2){
			iElementQuantityByBlock = DOC_QUANTITY_IN_MEMORY;//iBlockRoundNumber * DOC_QUANTITY_IN_MEMORY;
			iGlobalInitialPositionInList = iElementQuantityByBlock  * blockIdx.x * iGlobalRoundNumber;
		}

		//Inicializa a lista de Score e Documentos dos Topk
		//Considero que o Top_K seja um número múltiplo do tamanho do bloco
		for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
			documentTopk.id[localIndex] = -1;
			documentTopk.score[localIndex] = 0.0;
		}

		__syncthreads();

		//Preenche a memória compartilhada
		positionInitialInTermPostingList = 0;
		int docLocal, docAmount;
		for (int termId = 0; termId < iTermNumber; ++termId) {
			globalIndex = iGlobalInitialPositionInList + limitDoc.extraPosition[termId] + threadIdx.x;
			docAmount = d_iDocNumberByTermList[termId];
			docLocal = -1;

			for (localIndex = threadIdx.x; localIndex < iElementQuantityByBlock; localIndex+=blockDim.x) {

				docLocal = (globalIndex < docAmount) ? iDocIdList[positionInitialInTermPostingList + globalIndex]
																  : NO_MORE_DOC;

				if(docLocal > limitDoc.secondMaxDocId || globalIndex > docAmount){
					postingLists[termId].docId[localIndex] = NO_MORE_DOC;
					fingers[termId].final = 1;
					break;
				}

				postingLists[termId].docId[localIndex] = docLocal;
				postingLists[termId].docLenght[localIndex] = iDocLenghtList[positionInitialInTermPostingList + globalIndex];
				postingLists[termId].freq[localIndex] = iFreqList[positionInitialInTermPostingList + globalIndex];

				globalIndex += blockDim.x;
			}

			positionInitialInTermPostingList += d_iDocNumberByTermList[termId];
		}

	//	__syncthreads();

		if(threadIdx.x < iTermNumber){
			fingers[threadIdx.x].docId = postingLists[threadIdx.x].docId[0];
			fingers[threadIdx.x].position = (fingers[threadIdx.x].docId == NO_MORE_DOC) ? NO_VALID_POSITION : 0;
//			fingers[threadIdx.x].final = 0;
		}

		__syncthreads();

			__shared__ int docCurrent;

		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

	//	__syncthreads();

		if(THREAD_MASTER){
			selectTermPivot_2(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
		}else if(THREAD_MASTER_2){
			score = 0.0;
		}

		int padding;
		int threadIdInWarp = (threadIdx.x & 0x1f);
		int idWarp = ((blockDim.x >> 5) == 1 ) ? 1 :  threadIdx.x >> 5;

		__syncthreads();

		while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){


			if(THREAD_MASTER){
				isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);
			}

			__syncthreads();

			if(isValidCandidate){

				if(threadIdx.x < iTermNumber){
					fullScore_3_1(&score, fingers[sharedPivot.idTerm].docId, iOrderedTermSharedList,
								  fingers,postingLists, dIdfList, dAverageDocumentLength);
				}

				padding = documentTopk.padding;

				__syncthreads();

	/*				If the heap is not full
				the candidate is inserted into the heap. If the heap is full
				and the new score is larger than the minimum score in the
				heap, the new document is inserted into the heap, replacing
				the one with the minimum score.

	*/
				if(padding != 0 || thresholdLocal < score ){
					thresholdLocal = managerMinValue_v5(&documentTopk, fingers[sharedPivot.idTerm].docId, score,padding);
				}

				if(idWarp == 1 && threadIdInWarp < iTermNumber){
					advancePivoTermFinger_4(sharedPivot,fingers, postingLists,iElementQuantityByBlock,threadIdInWarp);
				}
			}
			else{
				 advanceDocIdOfPredecessorTerm_4(postingLists,
											   iOrderedTermSharedList,
											   fingers,sharedPivot,fingers[sharedPivot.idTerm].docId,
											   iElementQuantityByBlock);
			}

			__syncthreads();

			for (int termId = 0; termId < iTermNumber; ++termId) {
				if(fingers[termId].docId == NO_MORE_DOC && 	fingers[termId].final == 0){

					searchMoreDocs(iDocIdList,iFreqList,iDocLenghtList,postingLists,
								  termId,iGlobalInitialPositionInList,
								  &limitDoc,iElementQuantityByBlock,
								  &(fingers[termId]),docCurrent,d_iDocNumberByTermList);

					if (SHAREDTHESHOLD == 1){//SHARED_READ
						if(THREAD_MASTER && documentTopk.padding == 0 && thresholdLocal > globalThreshold){
//							atomicMaxD(&globalThreshold,thresholdLocal);
							globalThreshold = thresholdLocal;
//							atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
						}

						if(thresholdLocal < globalThreshold){
							thresholdLocal = globalThreshold;
						}
					}else if (SHAREDTHESHOLD == 2){ //TSHARED_WRITEREAD
						if(THREAD_MASTER && (documentTopk.padding < (iTopK >> 1)) && thresholdLocal > globalThreshold){
//							globalThreshold,thresholdLocal);
							globalThreshold = thresholdLocal;
//							atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
						}

						if((documentTopk.padding < (iTopK >> 1)))
							if(thresholdLocal < globalThreshold){
								thresholdLocal = globalThreshold;
							}
					}
				}
			}

			//Sort the terms in non decreasing order of DID
			sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

			//Select term pivot
			if(THREAD_MASTER){
				selectTermPivot_2(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
				docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
				score = 0.0;
			}
			__syncthreads();
		}

		sortLocalTopkDocAndStoreInGlobal(dTopkScoreListGlobal,iTopkDocListGlobal,iTopK,&documentTopk);

		if(THREAD_MASTER && thresholdLocal > globalThreshold){
			atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
		}
//		if(threadIdx.x == 0)
//			printf("---------%d----------",count);
}

__global__ void matchWandParallel_BATCH_2(const int* iDocIdList, const unsigned short int* iFreqList,
										const float *dUBlistGlobal, const float *dIdfListGlobal, const int *iDocLenghtList,
										const int iTermNumber, int *iTopkDocListGlobal,
										float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
										const int iBlockRoundNumber, const int iGlobalRoundNumber,
										const short int iTopK, const float iInitialThreshold,
										const int* iQueryTerms, const long long* ptrInitPostingList,
										int* ptrQueryPositions, int idQuery,int *iDocNumberByTermListGlobal){

//
//		if(idQuery != 18 || blockIdx.x != 0)
//			return;

		__shared__ int queryPosition;
		__shared__ pivot sharedPivot;
		__shared__ finger fingers[TERM_NUMBER];

		__shared__ documentTopkList documentTopk;

		__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];
		__shared__ long long iSharedPositionInitialInList[TERM_NUMBER];
		__shared__ float dUBlist[TERM_NUMBER];
		__shared__ float dIdfList[TERM_NUMBER];
		__shared__ float dAverageDocumentLength;

		__shared__ int iDocNumberByTermList[TERM_NUMBER];
		__shared__ int iGlobalInitialPositionInList;

		__shared__ float score;
		__shared__ bool isValidCandidate;
		__shared__ int docCurrent;
		__shared__ limitDocId limitDoc;

		int padding;

	 	float thresholdLocal;// = iInitialThreshold;
	 	thresholdLocal = iInitialThreshold;

		int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
		int localIndex;
//		long long positionInitialInTermPostingList;

		if(thresholdLocal < globalThresholdBatch[idQuery])
			thresholdLocal = globalThresholdBatch[idQuery];

//		int count = 0;

		if(THREAD_MASTER){
			documentTopk.padding = iTopK;
			dAverageDocumentLength = dAverageDocumentLengthGlobal;
	//	}else if(THREAD_MASTER_2){
			limitDoc.minDocId = 0;
			limitDoc.secondMaxDocId = 0;
			iGlobalInitialPositionInList = DOC_QUANTITY_IN_MEMORY  * blockIdx.x * iGlobalRoundNumber;
		}

		if(threadIdx.x < iTermNumber){
			queryPosition = ptrQueryPositions[idQuery];
			int idTerm = iQueryTerms[queryPosition + threadIdx.x];

			fingers[threadIdx.x].docId = NO_MORE_DOC;
			fingers[threadIdx.x].position = NO_VALID_POSITION;

			iDocNumberByTermList[threadIdx.x] = iDocNumberByTermListGlobal[idTerm];

			dUBlist[threadIdx.x] = dUBlistGlobal[idTerm];//[threadIdx.x];
			dIdfList[threadIdx.x] = dIdfListGlobal[idTerm];//[threadIdx.x];

			iSharedPositionInitialInList[threadIdx.x] = ptrInitPostingList[idTerm];
		}

		//Inicializa a lista de Score e Documentos dos Topk
		//Considero que o Top_K seja um número múltiplo do tamanho do bloco
		for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
			documentTopk.id[localIndex] = -1;
			documentTopk.score[localIndex] = 0.0;
		}

		//Define o max e o min
		if(threadIdx.x < iTermNumber){
			int docAmount = iDocNumberByTermList[threadIdx.x];
			fingers[threadIdx.x].final = 0;
			limitDoc.extraPosition[threadIdx.x] = 0;

			globalIndex = iGlobalInitialPositionInList;

			int maxDoc;
			if(blockIdx.x != 0){
				maxDoc = (globalIndex < docAmount) ? iDocIdList[iSharedPositionInitialInList[threadIdx.x] + globalIndex - 1] : -1;
				maxDoc++;

				atomicMax(&(limitDoc.minDocId), maxDoc);
			}else{
				if(THREAD_MASTER) limitDoc.minDocId = 0;
			}

			int isTail = globalIndex < docAmount;
			globalIndex = globalIndex + DOC_QUANTITY_IN_MEMORY * iGlobalRoundNumber;
			isTail = isTail && globalIndex >= docAmount;

			if(isTail){
				globalIndex = iGlobalInitialPositionInList + (docAmount - iGlobalInitialPositionInList - 1);
			}

			maxDoc =  ( (isTail || (globalIndex < docAmount))
							? (iDocIdList[iSharedPositionInitialInList[threadIdx.x] +  globalIndex-1]) : -1);
			atomicMax(&(limitDoc.secondMaxDocId), maxDoc);
		}

		__syncthreads();

		long long pos;
		int docLocal;
		for (int idTerm = 0; idTerm < iTermNumber; ++idTerm) {
			pos = iSharedPositionInitialInList[idTerm] + iGlobalInitialPositionInList + threadIdx.x;
			docLocal = -1;
			while(pos < (iSharedPositionInitialInList[idTerm]+iDocNumberByTermList[idTerm])
					&& docLocal < limitDoc.minDocId && docLocal <= limitDoc.secondMaxDocId ){
				docLocal = iDocIdList[pos];
				pos += blockDim.x;
			}
			docLocal = ((docLocal != -1)
					&& (docLocal >= limitDoc.minDocId && docLocal <= limitDoc.secondMaxDocId)) ? docLocal : NO_MORE_DOC;
			pos = (docLocal != NO_MORE_DOC) ? pos-blockDim.x : NO_VALID_POSITION;

			atomicMin(&(fingers[idTerm].docId) , docLocal);

			__syncthreads();

			if(fingers[idTerm].docId == docLocal){
				fingers[idTerm].position = pos;
			}
		}

		__syncthreads();

		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

		if(THREAD_MASTER){
			selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
	//	}else if(THREAD_MASTER_2){
			score = 0.0;
		}

		__syncthreads();

		while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){

			isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);
//			count++;

			if(isValidCandidate){
				  if(threadIdx.x < iTermNumber){
			 		int termId = iOrderedTermSharedList[threadIdx.x];
			 		float scoreL = 0.0;
			 		if(fingers[termId].docId == fingers[sharedPivot.idTerm].docId){
			 			scoreL = scoreTf_Idf(iFreqList[fingers[termId].position],
											iDocLenghtList[fingers[termId].position],
											dIdfList[termId],dAverageDocumentLength,1.0);
			 		}

			 		float aux = 0;
			 		for (int i = 0; i < TERM_NUMBER; ++i) {
			 			aux += __shfl_sync(0xFFFFFFFF,scoreL,i);
			 		}

			 		if(THREAD_MASTER) score = aux;
	//		 		atomicAdd(&score,scoreL);
				}

				padding = documentTopk.padding;

				__syncthreads();

	/*				If the heap is not full
				the candidate is inserted into the heap. If the heap is full
				and the new score is larger than the minimum score in the
				heap, the new document is inserted into the heap, replacing
				the one with the minimum score.

	*/
				if(padding != 0 || thresholdLocal < score ){
					thresholdLocal = managerMinValue_v5(&documentTopk, fingers[sharedPivot.idTerm].docId, score,padding);
				}

				if(threadIdx.x < iTermNumber ){

				 	int docPivot = fingers[sharedPivot.idTerm].docId;
				 	if(fingers[threadIdx.x].docId ==  docPivot){
				 		fingers[threadIdx.x].position++;
				 		if(fingers[threadIdx.x].position >= (iDocNumberByTermList[threadIdx.x]+iSharedPositionInitialInList[threadIdx.x])){//Não Válido
				 			fingers[threadIdx.x].docId = NO_MORE_DOC;
				 			fingers[threadIdx.x].position = NO_VALID_POSITION;
				 		}else{
				 			fingers[threadIdx.x].docId = iDocIdList[fingers[threadIdx.x].position];
				 			if(fingers[threadIdx.x].docId > limitDoc.secondMaxDocId){
				 				fingers[threadIdx.x].docId = NO_MORE_DOC;
				 				fingers[threadIdx.x].position = NO_VALID_POSITION;
				 			}
				 		}
				 	}
				}
			}
			else{
				int pivotDoc = docCurrent;
				int position;
				int docLocal;
				int idTerm;
				for (int j = 0; j < sharedPivot.positionInOrderedList; ++j) {
					idTerm = iOrderedTermSharedList[j];

					if(fingers[idTerm].docId == fingers[sharedPivot.idTerm].docId)//Até alcançar um finger q aponte a um documento pivo
						break;

					fingers[idTerm].docId = NO_MORE_DOC;
					position = fingers[idTerm].position + 1 + threadIdx.x;
					docLocal = -1;
					while(position < (iSharedPositionInitialInList[idTerm]+iDocNumberByTermList[idTerm])
							&& docLocal < pivotDoc && docLocal <= limitDoc.secondMaxDocId){
						docLocal = iDocIdList[position];
						position += blockDim.x;
					}
					docLocal = (docLocal >= pivotDoc && docLocal <= limitDoc.secondMaxDocId) ? docLocal : NO_MORE_DOC;
					position = (docLocal != NO_MORE_DOC) ? position-blockDim.x : NO_VALID_POSITION;

					__syncthreads();

					atomicMin(&(fingers[idTerm].docId) , docLocal);

					__syncthreads();

					if(fingers[idTerm].docId == docLocal){
						fingers[idTerm].position = position;
					}
				}
			}


			__syncthreads();

			//Sort the terms in non decreasing order of DID
			sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

			//Select term pivot
			if(THREAD_MASTER){
				selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
				docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
				score = 0.0;
			}

			if (SHAREDTHESHOLD == 1){//SHARED_READ
				if(THREAD_MASTER && documentTopk.padding == 0 && thresholdLocal > globalThresholdBatch[idQuery]){
	//							atomicMaxD(&globalThreshold,thresholdLocal);
//					atomicMax((unsigned long long int*)&(globalThresholdBatch[idQuery]),(unsigned long long int)thresholdLocal);
//					atomicMaxD((volatile double*)&(globalThresholdBatch[idQuery]),thresholdLocal);
					globalThresholdBatch[idQuery] = thresholdLocal;
				}

				if(thresholdLocal < globalThresholdBatch[idQuery]){
					thresholdLocal = globalThresholdBatch[idQuery];
				}
			}else if (SHAREDTHESHOLD == 2){ //TSHARED_WRITEREAD
				if(THREAD_MASTER && (documentTopk.padding < (iTopK >> 1)) && thresholdLocal > globalThresholdBatch[idQuery]){
//					atomicMax((unsigned long long int*)&(globalThresholdBatch[idQuery]),(unsigned long long int)thresholdLocal);
//					atomicMaxD(((volatile double*)&(globalThresholdBatch[idQuery])),thresholdLocal);
					 globalThresholdBatch[idQuery] = thresholdLocal;
				}

				if((documentTopk.padding < (iTopK >> 1)))
					if(thresholdLocal < globalThresholdBatch[idQuery]){
						thresholdLocal = globalThresholdBatch[idQuery];
					}
			}
			__syncthreads();
		}

		sortLocalTopkDocAndStoreInGlobal(dTopkScoreListGlobal,iTopkDocListGlobal,iTopK,&documentTopk);

//		if(thresholdLocal > globalThreshold)
//			globalThreshold = thresholdLocal;

//		if(THREAD_MASTER)
////			atomicAdd(&globalCount,count);
//
//		if(THREAD_MASTER && idQuery == 0)
//			printf("-----%d %d----", blockIdx.x, count);

}


__global__ void matchWandParallel_VARIABLE_4_2(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlistGlobal, const float *dIdfListGlobal,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
										  const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* iDocNumberByTermListGlobal){

//	if(blockIdx.x != 720)
//		return;
	__shared__ pivot sharedPivot;
	__shared__ finger fingers[TERM_NUMBER];

	__shared__ documentTopkList documentTopk;

	__shared__ postingList2 postings[TERM_NUMBER];

	__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];
	__shared__ long long iSharedPositionInitialInList[TERM_NUMBER];
	__shared__ float dUBlist[TERM_NUMBER];
	__shared__ float dIdfList[TERM_NUMBER];
	__shared__ float dAverageDocumentLength;

	__shared__ int iDocNumberByTermList[TERM_NUMBER];
	__shared__ int iGlobalInitialPositionInList;

	__shared__ float score;
	__shared__ bool isValidCandidate;
	__shared__ int docCurrent;
	__shared__ limitDocId limitDoc;

//	int count = iTopK;

//	__shared__ int paddingInShared;
	int padding;

 	float thresholdLocal = iInitialThreshold;
 	thresholdLocal = iInitialThreshold;

	int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
	int localIndex;
	int positionInitialInTermPostingList;


	if(thresholdLocal < globalThreshold)
		thresholdLocal = globalThreshold;

	if(THREAD_MASTER){
		documentTopk.padding = iTopK;
		dAverageDocumentLength = dAverageDocumentLengthGlobal;
//	}else if(THREAD_MASTER_2){
		iGlobalInitialPositionInList = DOC_QUANTITY_IN_MEMORY  * blockIdx.x * iGlobalRoundNumber;
		limitDoc.minDocId = 0;
		limitDoc.secondMaxDocId = 0;
	}

	if(threadIdx.x < iTermNumber){
//		paddingInShared=0;
		fingers[threadIdx.x].docId = NO_MORE_DOC;
		fingers[threadIdx.x].position = NO_VALID_POSITION;
		iDocNumberByTermList[threadIdx.x] = iDocNumberByTermListGlobal[threadIdx.x];
		dUBlist[threadIdx.x] = dUBlistGlobal[threadIdx.x];
		dIdfList[threadIdx.x] = dIdfListGlobal[threadIdx.x];
	}

	//Inicializa a lista de Score e Documentos dos Topk
	//Considero que o Top_K seja um número múltiplo do tamanho do bloco
	for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
		documentTopk.id[localIndex] = -1;
		documentTopk.score[localIndex] = 0.0;
	}

	//Define o max e o min
	if(threadIdx.x < iTermNumber){
		int docAmount = iDocNumberByTermList[threadIdx.x];
		fingers[threadIdx.x].final = 0;
		limitDoc.extraPosition[threadIdx.x] = 0;

		globalIndex = iGlobalInitialPositionInList;
		positionInitialInTermPostingList = 0;

		for (int i = 0; i < threadIdx.x; ++i) {
			positionInitialInTermPostingList += iDocNumberByTermList[i];
		}
		iSharedPositionInitialInList[threadIdx.x] = positionInitialInTermPostingList;

		int maxDoc;
		if(blockIdx.x != 0){
			maxDoc = (globalIndex < docAmount) ? iDocIdList[positionInitialInTermPostingList + globalIndex - 1] : -1;
			maxDoc++;
			atomicMax(&(limitDoc.minDocId), maxDoc);
		}else{
			if(THREAD_MASTER) limitDoc.minDocId = 0;
		}

		int isTail = globalIndex < docAmount;
		globalIndex += DOC_QUANTITY_IN_MEMORY * iGlobalRoundNumber - 1;
		isTail = (isTail && globalIndex >= docAmount);

		if(isTail){
			globalIndex = iGlobalInitialPositionInList + (docAmount - iGlobalInitialPositionInList - 1);
		}

		maxDoc =  (isTail || globalIndex < docAmount) ? iDocIdList[positionInitialInTermPostingList +  globalIndex] : -1;
		atomicMax(&(limitDoc.secondMaxDocId), maxDoc);
	}

	__syncthreads();

	long long pos;
//	int docLocal;
	for (int idTerm = 0; idTerm < iTermNumber; ++idTerm) {
		pos = iSharedPositionInitialInList[idTerm] + iGlobalInitialPositionInList + threadIdx.x;
		int docLocal = -1;
		while(pos < (iSharedPositionInitialInList[idTerm]+iDocNumberByTermList[idTerm])
				&& docLocal < limitDoc.minDocId && docLocal <= limitDoc.secondMaxDocId ){
			docLocal = iDocIdList[pos];
			pos += blockDim.x;
		}
		docLocal = ( (docLocal != -1)
				&& (docLocal >= limitDoc.minDocId && docLocal <= limitDoc.secondMaxDocId)) ? docLocal : NO_MORE_DOC;

		pos = (docLocal != NO_MORE_DOC) ? pos-blockDim.x : NO_VALID_POSITION;

//		atomicMin(&(fingers[idTerm].docId) , docLocal);

		int docNeighbor, docAux = docLocal;
		for (int i = 16; i >= 1; i /= 2) {
			docNeighbor  = __shfl_down_sync(0xFFFFFFFF,docAux, i);

			if(docNeighbor < docAux)
				docAux = docNeighbor;
		}

		if( ((threadIdx.x & 0x1f) == 0)){
			atomicMin(&(fingers[idTerm].docId) , docAux);
		}


		__syncthreads();

		if(fingers[idTerm].docId == docLocal){
			fingers[idTerm].position = pos;
		}
	}

	__syncthreads();

	for (int termId = 0; termId < iTermNumber; ++termId) {
		if(fingers[termId].position != NO_VALID_POSITION){
			long long gIndex = fingers[termId].position + threadIdx.x;
			for (int localIndex = threadIdx.x; localIndex < DOCS_TEST; localIndex+=blockDim.x) {
				if(gIndex < (iSharedPositionInitialInList[termId]+iDocNumberByTermList[termId])
					&& (iDocIdList[gIndex] <= limitDoc.secondMaxDocId) ){
					postings[termId].docId[localIndex] =  iDocIdList[gIndex];
					postings[termId].freq[localIndex] =   iFreqList[gIndex];
					postings[termId].docLenght[localIndex] =  iDocLenghtList[gIndex];
					if(localIndex == 0) postings[termId].positionInShared = 0;
				}
				else{
					postings[termId].docId[localIndex] =  NO_MORE_DOC;
					if(localIndex == 0) postings[termId].positionInShared = NO_VALID_POSITION;
				}
				gIndex += blockDim.x;
			}
		}
		else{
			postings[termId].positionInShared = NO_VALID_POSITION;
		}
	}


	sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

	if(THREAD_MASTER){
		selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
		docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
		score = 0.0;
	}

	__syncthreads();

	while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){

		isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);
//		count++;

//		if(fingers[sharedPivot.idTerm].docId==33769946 && THREAD_MASTER)
//			printf("blockId.x %d!!!\n",blockIdx.x);

		if(isValidCandidate){
			if(threadIdx.x < iTermNumber){
		 		int termId = iOrderedTermSharedList[threadIdx.x];
		 		float scoreL = 0.0;
		 		if(fingers[termId].docId == fingers[sharedPivot.idTerm].docId){
		 			scoreL = scoreTf_Idf(postings[termId].freq[postings[termId].positionInShared],
		 					postings[termId].docLenght[postings[termId].positionInShared],
										dIdfList[termId],dAverageDocumentLength,1.0);
		 		}
		 		float aux = 0;
		 		for (int i = 0; i < TERM_NUMBER; ++i) {
		 			aux += __shfl_sync(0xFFFFFFFF,scoreL,i);
		 		}

		 		if(THREAD_MASTER) score = aux;
//		 		atomicAdd(&score,scoreL);
			}

			padding = documentTopk.padding;

			__syncthreads();

/*				If the heap is not full
			the candidate is inserted into the heap. If the heap is full
			and the new score is larger than the minimum score in the
			heap, the new document is inserted into the heap, replacing
			the one with the minimum score.

*/
			if(padding != 0 || thresholdLocal < score ){
//				if(THREAD_MASTER && fingers[sharedPivot.idTerm].docId==6364669)//&& score == 3.53512168))//40920063
//					printf("blockIdx.x %d\n",blockIdx.x);

				thresholdLocal = managerMinValue_v5(&documentTopk, fingers[sharedPivot.idTerm].docId, score,padding);
//				if(count != 0) count--;
			}

//			float test = checkMinHeapProperty(documentTopk,score,fingers[sharedPivot.idTerm].docId,iTopK);

//			if(count != documentTopk.padding){
//				printf("Padding error! count %d | padding %d |  blockIdx %d | docId %d\n",count, documentTopk.padding, blockIdx.x, fingers[sharedPivot.idTerm].docId);
//			}
//
//			int result = __syncthreads_or(test != 0.0);
//			if(THREAD_MASTER && result != 0){
//				printf("Oi\n");
//				return;
//			}

			if(threadIdx.x < iTermNumber ){
			 	int docPivot = fingers[sharedPivot.idTerm].docId;
			 	int posInShared;
			 	if(fingers[threadIdx.x].docId ==  docPivot){
			 		fingers[threadIdx.x].position++;
			 		postings[threadIdx.x].positionInShared++;
			 		posInShared = postings[threadIdx.x].positionInShared;

			 		if(posInShared >= DOCS_TEST || postings[threadIdx.x].docId[posInShared]  == NO_MORE_DOC){
			 			fingers[threadIdx.x].docId = NO_MORE_DOC;
			 			if(docPivot == docCurrent)
			 				atomicInc((unsigned int*)(&docCurrent),docCurrent);
			 		}else{
			 			fingers[threadIdx.x].docId = postings[threadIdx.x].docId[posInShared];
			 		}
			 	}
			}
		}
		else{
			int pivotDoc = docCurrent;
			int position;
			int docLocal;
			int idTerm;
//			__syncthreads();
			for (int j = 0; j < sharedPivot.positionInOrderedList; ++j) {
				idTerm = iOrderedTermSharedList[j];

				if(fingers[idTerm].docId == pivotDoc)//Até alcançar um finger q aponte a um documento pivo
					break;

				position = postings[idTerm].positionInShared + 1 + threadIdx.x;
				fingers[idTerm].docId = NO_MORE_DOC;

				if(position < DOCS_TEST)
					docLocal = postings[idTerm].docId[position];
				else
					docLocal = NO_MORE_DOC;

				while( (position < DOCS_TEST) && (docLocal < pivotDoc) ){
					docLocal = postings[idTerm].docId[position];
					position += blockDim.x;
				}

				docLocal = (docLocal > pivotDoc) ? docLocal : NO_MORE_DOC;
				position = (docLocal != NO_MORE_DOC) ? position-blockDim.x : DOCS_TEST;

//				__syncthreads();

				int docNeighbor, docAux = docLocal;
				for (int i = 16; i >= 1; i /= 2) {
					docNeighbor  = __shfl_down_sync(0xFFFFFFFF,docAux, i);

					if(docNeighbor < docAux)
						docAux = docNeighbor;
				}

				if( ((threadIdx.x & 0x1f) == 0)){
					atomicMin(&(fingers[idTerm].docId) , docAux);
				}

				__syncthreads();

				if(fingers[idTerm].docId == docLocal){
					if(position != DOCS_TEST){
						fingers[idTerm].position += (position-postings[idTerm].positionInShared);
						postings[idTerm].positionInShared += threadIdx.x + 1;
					}
					else {
						postings[idTerm].positionInShared = DOCS_TEST;
//						fingers[idTerm].position = NO_VALID_POSITION;
					}
				}
			}
		}

		for (int termId = 0; termId < iTermNumber; ++termId) {
			long long gIndex;
			int count=0,isValid=0, docLocal, isOutRange=0;
			if(postings[termId].positionInShared >= DOCS_TEST && postings[termId].positionInShared != NO_VALID_POSITION){
				gIndex = fingers[termId].position + threadIdx.x;
				for (int localIndex = threadIdx.x; localIndex < DOCS_TEST; localIndex+=blockDim.x) {

					count=0;isValid=0;isOutRange=0;
					do{
						isOutRange = gIndex >= (iSharedPositionInitialInList[termId]+iDocNumberByTermList[termId]);
						docLocal = (!isOutRange) ? iDocIdList[gIndex] : NO_MORE_DOC;
						isOutRange = isOutRange || (docLocal > limitDoc.secondMaxDocId);
						isValid =  isOutRange || (docLocal >= docCurrent);

//						count = __syncthreads_count(!isValid);
						count = __ballot_sync(0xFFFFFFFF,!isValid);
						count = __popc(count);

//						if((threadIdx.x & 0x1f) == 0){
//							atomicAdd(&paddingInShared,count);
//						}
//						__syncthreads();
//						count = paddingInShared;
						gIndex += count;
						if(localIndex == 0) fingers[termId].position += count;
					}while(count != 0);

					if(!isOutRange){
						postings[termId].docId[localIndex] = docLocal;
						postings[termId].freq[localIndex] =   iFreqList[gIndex];
						postings[termId].docLenght[localIndex] =  iDocLenghtList[gIndex];
						if(localIndex == 0) postings[termId].positionInShared = 0;
					}
					else{
						postings[termId].docId[localIndex] =  NO_MORE_DOC;
						if(localIndex == 0) postings[termId].positionInShared = NO_VALID_POSITION;
					}
					gIndex += blockDim.x;
				}

				if(threadIdx.x == 0){
					fingers[termId].docId = postings[termId].docId[0];
				}
//				paddingInShared=0;
//				__syncthreads();
			}
		}

		__syncthreads();

		//Sort the terms in non decreasing order of DID
		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

		//Select term pivot
		if(THREAD_MASTER){
			selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
			score = 0.0;
		}

		__syncthreads();


		if (SHAREDTHESHOLD == 1){//SHARED_READ
			if(THREAD_MASTER && documentTopk.padding == 0 && thresholdLocal > globalThreshold){
//				atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
				globalThreshold=thresholdLocal;
			}

			if(thresholdLocal < globalThreshold){
				thresholdLocal = globalThreshold;
			}

		}else if (SHAREDTHESHOLD == 2){ //TSHARED_WRITEREAD
			if(THREAD_MASTER && (documentTopk.padding < (iTopK >> 1)) && thresholdLocal > globalThreshold){
//				atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
				globalThreshold=thresholdLocal;
			}

			if((documentTopk.padding < (iTopK >> 1)))
				if(thresholdLocal < globalThreshold){
					thresholdLocal = globalThreshold;
				}
		}

	}//Fim do WAND - pivot = NO_MORE_DOC


//	for (int i = blockIdx.x*iTopK+threadIdx.x; i < blockIdx.x*iTopK; i+= blockDim.x) {
//		printf("---%d %d---",blockIdx.x,iTopkDocListGlobal[i]);
//	}

	sortLocalTopkDocAndStoreInGlobal(dTopkScoreListGlobal,iTopkDocListGlobal,iTopK,&documentTopk);


//	__syncthreads();



//	float test = checkSorting(documentTopk, dTopkScoreListGlobal, iTopkDocListGlobal, iTopK);
//
//	int result = __syncthreads_or(test != 0.0);
//	if(THREAD_MASTER && result != 0){
//		printf("Oi no Sorting!\n");
//		return;
//	}

//	if(thresholdLocal > globalThreshold)
//		thresholdGlobal = thresholdLocal;

//	if(THREAD_MASTER)
//		atomicAdd(&globalCount,count);
//////
//	if(THREAD_MASTER)
//		printf("-----%d----", globalCount);
}

__global__ void matchWandParallel_VARIABLE_4(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlistGlobal, const float *dIdfListGlobal,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
										  const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* iDocNumberByTermListGlobal){

//	if(blockIdx.x != 1104)
//		return;

	__shared__ pivot sharedPivot;
	__shared__ finger fingers[TERM_NUMBER];

	__shared__ documentTopkList documentTopk;

	__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];
	__shared__ long long iSharedPositionInitialInList[TERM_NUMBER];
	__shared__ float dUBlist[TERM_NUMBER];
	__shared__ float dIdfList[TERM_NUMBER];
	__shared__ float dAverageDocumentLength;

	__shared__ int iDocNumberByTermList[TERM_NUMBER];
	__shared__ int iGlobalInitialPositionInList;

	__shared__ float score;
	__shared__ bool isValidCandidate;
	__shared__ int docCurrent;
	__shared__ limitDocId limitDoc;

//	int count = iTopK;

	int padding;

 	float thresholdLocal = iInitialThreshold;
 	thresholdLocal = iInitialThreshold;

	int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
	int localIndex;
	int positionInitialInTermPostingList;

	if(thresholdLocal < globalThreshold)
		thresholdLocal = globalThreshold;

	if(THREAD_MASTER){
		documentTopk.padding = iTopK;
		dAverageDocumentLength = dAverageDocumentLengthGlobal;
//	}else if(THREAD_MASTER_2){
		iGlobalInitialPositionInList = DOC_QUANTITY_IN_MEMORY  * blockIdx.x * iGlobalRoundNumber;
		limitDoc.minDocId = 0;
		limitDoc.secondMaxDocId = 0;
	}

	if(threadIdx.x < iTermNumber){
		fingers[threadIdx.x].docId = NO_MORE_DOC;
		fingers[threadIdx.x].position = NO_VALID_POSITION;
		iDocNumberByTermList[threadIdx.x] = iDocNumberByTermListGlobal[threadIdx.x];
		dUBlist[threadIdx.x] = dUBlistGlobal[threadIdx.x];
		dIdfList[threadIdx.x] = dIdfListGlobal[threadIdx.x];
	}

	//Inicializa a lista de Score e Documentos dos Topk
	//Considero que o Top_K seja um número múltiplo do tamanho do bloco
	for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
		documentTopk.id[localIndex] = -1;
		documentTopk.score[localIndex] = 0.0;
	}

//	__syncthreads();

//	if(blockIdx.x == 83 && THREAD_MASTER)
//		printf("Oi! \n");

	//Define o max e o min
	if(threadIdx.x < iTermNumber){
		int docAmount = iDocNumberByTermList[threadIdx.x];
		fingers[threadIdx.x].final = 0;
		limitDoc.extraPosition[threadIdx.x] = 0;

		globalIndex = iGlobalInitialPositionInList;
		positionInitialInTermPostingList = 0;

		for (int i = 0; i < threadIdx.x; ++i) {
			positionInitialInTermPostingList += iDocNumberByTermList[i];
		}
		iSharedPositionInitialInList[threadIdx.x] = positionInitialInTermPostingList;

		int maxDoc;
		if(blockIdx.x != 0){
			maxDoc = (globalIndex < docAmount) ? iDocIdList[positionInitialInTermPostingList + globalIndex - 1] : -1;
			maxDoc++;
			atomicMax(&(limitDoc.minDocId), maxDoc);
		}else{
			if(THREAD_MASTER) limitDoc.minDocId = 0;
		}

		int isTail = globalIndex < docAmount;
		globalIndex += DOC_QUANTITY_IN_MEMORY * iGlobalRoundNumber - 1;
		isTail = (isTail && globalIndex >= docAmount);

		if(isTail){
			globalIndex = iGlobalInitialPositionInList + (docAmount - iGlobalInitialPositionInList - 1);
		}

		maxDoc =  (isTail || globalIndex < docAmount) ? iDocIdList[positionInitialInTermPostingList +  globalIndex] : -1;
		atomicMax(&(limitDoc.secondMaxDocId), maxDoc);
	}

	__syncthreads();

	long long pos;
	int docLocal;
	for (int idTerm = 0; idTerm < iTermNumber; ++idTerm) {
		pos = iSharedPositionInitialInList[idTerm] + iGlobalInitialPositionInList + threadIdx.x;
		docLocal = -1;
		while(pos < (iSharedPositionInitialInList[idTerm]+iDocNumberByTermList[idTerm])
				&& docLocal < limitDoc.minDocId && docLocal <= limitDoc.secondMaxDocId ){
			docLocal = iDocIdList[pos];
			pos += blockDim.x;
		}
		docLocal = ( (docLocal != -1)
				&& (docLocal >= limitDoc.minDocId && docLocal <= limitDoc.secondMaxDocId)) ? docLocal : NO_MORE_DOC;
		pos = (docLocal != NO_MORE_DOC) ? pos-blockDim.x : NO_VALID_POSITION;

		atomicMin(&(fingers[idTerm].docId) , docLocal);

		__syncthreads();

		if(fingers[idTerm].docId == docLocal){
			fingers[idTerm].position = pos;
		}
	}

	sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

	if(THREAD_MASTER){
		selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
		docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
//	}else if(THREAD_MASTER_2){
		score = 0.0;
	}

	__syncthreads();

	while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){

		isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);
//		count++;

		if(isValidCandidate){
			if(threadIdx.x < iTermNumber){
		 		int termId = iOrderedTermSharedList[threadIdx.x];
		 		float scoreL = 0.0;
		 		if(fingers[termId].docId == fingers[sharedPivot.idTerm].docId){
		 			scoreL = scoreTf_Idf(iFreqList[fingers[termId].position],
										iDocLenghtList[fingers[termId].position],
										dIdfList[termId],dAverageDocumentLength,1.0);
		 		}

		 		float aux = 0;
		 		for (int i = 0; i < TERM_NUMBER; ++i) {
		 			aux += __shfl_sync(0xFFFFFFFF,scoreL,i);
		 		}

		 		if(THREAD_MASTER) score = aux;
//		 		atomicAdd(&score,scoreL);
			}

			padding = documentTopk.padding;

			__syncthreads();

/*				If the heap is not full
			the candidate is inserted into the heap. If the heap is full
			and the new score is larger than the minimum score in the
			heap, the new document is inserted into the heap, replacing
			the one with the minimum score.

*/
			if(padding != 0 || thresholdLocal < score ){
				if(THREAD_MASTER && fingers[sharedPivot.idTerm].docId==46517642)//&& score == 3.53512168))//40920063
					printf("blockIdx.x %d\n",blockIdx.x);

				thresholdLocal = managerMinValue_v5(&documentTopk, fingers[sharedPivot.idTerm].docId, score,padding);
//				if(count != 0) count--;
			}

//			float test = checkMinHeapProperty(documentTopk,score,fingers[sharedPivot.idTerm].docId,iTopK);

//			if(count != documentTopk.padding){
//				printf("Padding error! count %d | padding %d |  blockIdx %d | docId %d\n",count, documentTopk.padding, blockIdx.x, fingers[sharedPivot.idTerm].docId);
//			}
//
//			int result = __syncthreads_or(test != 0.0);
//			if(THREAD_MASTER && result != 0){
//				printf("Oi\n");
//				return;
//			}

			if(threadIdx.x < iTermNumber ){

			 	int docPivot = fingers[sharedPivot.idTerm].docId;
			 	if(fingers[threadIdx.x].docId ==  docPivot){
			 		fingers[threadIdx.x].position++;
			 		if(fingers[threadIdx.x].position >= (iDocNumberByTermList[threadIdx.x]+iSharedPositionInitialInList[threadIdx.x])){//Não Válido
			 			fingers[threadIdx.x].docId = NO_MORE_DOC;
			 			fingers[threadIdx.x].position = NO_VALID_POSITION;
			 		}else{
			 			fingers[threadIdx.x].docId = iDocIdList[fingers[threadIdx.x].position];
			 			if(fingers[threadIdx.x].docId > limitDoc.secondMaxDocId){
			 				fingers[threadIdx.x].docId = NO_MORE_DOC;
			 				fingers[threadIdx.x].position = NO_VALID_POSITION;
			 			}
			 		}
			 	}
			}
		}
		else{
			int pivotDoc = docCurrent;
			long long position;
			int docLocal;
			int idTerm;
			for (int j = 0; j < sharedPivot.positionInOrderedList; ++j) {
				idTerm = iOrderedTermSharedList[j];

				if(fingers[idTerm].docId == fingers[sharedPivot.idTerm].docId)//Até alcançar um finger q aponte a um documento pivo
					break;

				fingers[idTerm].docId = NO_MORE_DOC;
				position = fingers[idTerm].position + 1 + threadIdx.x;
				docLocal = -1;
				while(position < (iSharedPositionInitialInList[idTerm]+iDocNumberByTermList[idTerm])
						&& docLocal < pivotDoc && docLocal <= limitDoc.secondMaxDocId){
					docLocal = iDocIdList[position];
					position += blockDim.x;
				}
				docLocal = (docLocal >= pivotDoc && docLocal <= limitDoc.secondMaxDocId) ? docLocal : NO_MORE_DOC;
				position = (docLocal != NO_MORE_DOC) ? position-blockDim.x : NO_VALID_POSITION;

				__syncthreads();

				atomicMin(&(fingers[idTerm].docId) , docLocal);

				__syncthreads();

				if(fingers[idTerm].docId == docLocal){
					fingers[idTerm].position = position;
				}
			}
		}

		__syncthreads();

		//Sort the terms in non decreasing order of DID
		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

		//Select term pivot
		if(THREAD_MASTER){
			selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
			score = 0.0;
		}

		__syncthreads();

		if (SHAREDTHESHOLD == 1){//SHARED_READ
			if(THREAD_MASTER && documentTopk.padding == 0 && thresholdLocal > globalThreshold){
	//							atomicMaxD(&globalThreshold,thresholdLocal);
//				atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
				globalThreshold = thresholdLocal;
			}

			if(thresholdLocal < globalThreshold){
				thresholdLocal = globalThreshold;
			}
		}else if (SHAREDTHESHOLD == 2){ //TSHARED_WRITEREAD
			if(THREAD_MASTER && (documentTopk.padding < (iTopK >> 1)) && thresholdLocal > globalThreshold){
//				atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
//								atomicMaxD(&globalThreshold,thresholdLocal);
				globalThreshold = thresholdLocal;
			}

			if((documentTopk.padding < (iTopK >> 1)))
				if(thresholdLocal < globalThreshold){
					thresholdLocal = globalThreshold;
				}
		}

	}

//	for (int i = blockIdx.x*iTopK+threadIdx.x; i < blockIdx.x*iTopK; i+= blockDim.x) {
//		printf("---%d %d---",blockIdx.x,iTopkDocListGlobal[i]);
//	}

	sortLocalTopkDocAndStoreInGlobal(dTopkScoreListGlobal,iTopkDocListGlobal,iTopK,&documentTopk);

//	float test = checkSorting(documentTopk, dTopkScoreListGlobal, iTopkDocListGlobal, iTopK);
//
//	int result = __syncthreads_or(test != 0.0);
//	if(THREAD_MASTER && result != 0){
//		printf("Oi no Sorting!\n");
//		return;
//	}

//	if(thresholdLocal > globalThreshold)
//		thresholdGlobal = thresholdLocal;

//	if(THREAD_MASTER)
//		atomicAdd(&globalCount,count);
//////
//	if(THREAD_MASTER)
//		printf("-----%d----", globalCount);
}



__global__ void matchWandParallel_FIXED_3(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlistGlobal, const float *dIdfListGlobal,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
										  const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* iDocNumberByTermListGlobal){
	//	if(blockIdx.x != 0)
	//		return;

		__shared__ pivot sharedPivot;
		__shared__ finger fingers[TERM_NUMBER];

		__shared__ documentTopkList documentTopk;

		__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];
//		__shared__ long long int iSharedPositionInitialInList[TERM_NUMBER];
		__shared__ float dUBlist[TERM_NUMBER];
		__shared__ float dIdfList[TERM_NUMBER];
		__shared__ float dAverageDocumentLength;

		__shared__ int iDocNumberByTermList[TERM_NUMBER];
		__shared__ int iGlobalInitialPositionInList;

		__shared__ float score;
		__shared__ bool isValidCandidate;
		__shared__ int docCurrent;


		__shared__ long long finalPositions[TERM_NUMBER];
	//	int count =0;

		int padding;

	 	float thresholdLocal = iInitialThreshold;
	 	thresholdLocal = iInitialThreshold;

		int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
		int localIndex;
		long long int positionInitialInTermPostingList;

		if(thresholdLocal < globalThreshold)
			thresholdLocal = globalThreshold;

		if(THREAD_MASTER){
			documentTopk.padding = iTopK;
			dAverageDocumentLength = dAverageDocumentLengthGlobal;
			iGlobalInitialPositionInList = DOC_QUANTITY_IN_MEMORY  * blockIdx.x * iGlobalRoundNumber;
		}

		if(threadIdx.x < iTermNumber){
			iDocNumberByTermList[threadIdx.x] = iDocNumberByTermListGlobal[threadIdx.x];

			dUBlist[threadIdx.x] = dUBlistGlobal[threadIdx.x];
			dIdfList[threadIdx.x] = dIdfListGlobal[threadIdx.x];

			globalIndex = iGlobalInitialPositionInList;
			positionInitialInTermPostingList = 0;

			for (int i = 0; i < threadIdx.x; ++i) {
				positionInitialInTermPostingList += iDocNumberByTermList[i];
			}
//			iSharedPositionInitialInList[threadIdx.x] = positionInitialInTermPostingList;

			fingers[threadIdx.x].position = positionInitialInTermPostingList + globalIndex;

			if(fingers[threadIdx.x].position < (positionInitialInTermPostingList+iDocNumberByTermList[threadIdx.x])){
				fingers[threadIdx.x].docId = iDocIdList[fingers[threadIdx.x].position];
			}else{
				fingers[threadIdx.x].position = NO_VALID_POSITION;
				fingers[threadIdx.x].docId = NO_PIVOT_TERM;
			}

			finalPositions[threadIdx.x] = positionInitialInTermPostingList + globalIndex + DOC_QUANTITY_IN_MEMORY * iGlobalRoundNumber;

			if(finalPositions[threadIdx.x] >= (positionInitialInTermPostingList+iDocNumberByTermList[threadIdx.x]))
				finalPositions[threadIdx.x] = positionInitialInTermPostingList+iDocNumberByTermList[threadIdx.x];
		}

		//Inicializa a lista de Score e Documentos dos Topk
		//Considero que o Top_K seja um número múltiplo do tamanho do bloco
		for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
			documentTopk.id[localIndex] = -1;
			documentTopk.score[localIndex] = 0.0;
		}

		__syncthreads();

		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

		if(THREAD_MASTER){
			selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
	//	}else if(THREAD_MASTER_2){
			score = 0.0;
		}

		__syncthreads();

		while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){

			isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);
	//		count++;

			if(isValidCandidate){
				if(threadIdx.x < iTermNumber){
			 		int termId = iOrderedTermSharedList[threadIdx.x];
			 		float scoreL = 0.0;
			 		if(fingers[termId].docId == fingers[sharedPivot.idTerm].docId){
			 			scoreL = scoreTf_Idf(iFreqList[fingers[termId].position],
											iDocLenghtList[fingers[termId].position],
											dIdfList[termId],dAverageDocumentLength,1.1);
			 		}

			 		float aux = 0;
			 		for (int i = 0; i < TERM_NUMBER; ++i) {
			 			aux += __shfl_sync(0xFFFFFFFF,scoreL,i);
			 		}

			 		if(THREAD_MASTER) score = aux;
	//		 		atomicAdd(&score,scoreL);
				}

				padding = documentTopk.padding;

				__syncthreads();

	/*				If the heap is not full
				the candidate is inserted into the heap. If the heap is full
				and the new score is larger than the minimum score in the
				heap, the new document is inserted into the heap, replacing
				the one with the minimum score.

	*/
				if(padding != 0 || thresholdLocal < score ){
					thresholdLocal = managerMinValue_v5(&documentTopk, fingers[sharedPivot.idTerm].docId, score,padding);
				}

				if(threadIdx.x < iTermNumber ){

				 	int docPivot = fingers[sharedPivot.idTerm].docId;
				 	if(fingers[threadIdx.x].docId ==  docPivot){
				 		fingers[threadIdx.x].position++;
				 		if(fingers[threadIdx.x].position >= finalPositions[threadIdx.x] ){//Não Válido
				 			fingers[threadIdx.x].docId = NO_MORE_DOC;
				 			fingers[threadIdx.x].position = NO_VALID_POSITION;
				 		}else{
				 			fingers[threadIdx.x].docId = iDocIdList[fingers[threadIdx.x].position];
				 		}
				 	}
				}
			}
			else{
				int pivotDoc = docCurrent;
				long long int position;
				int docLocal;
				int idTerm;
				for (int j = 0; j < sharedPivot.positionInOrderedList; ++j) {
					idTerm = iOrderedTermSharedList[j];

					if(fingers[idTerm].docId == fingers[sharedPivot.idTerm].docId)//Até alcançar um finger q aponte a um documento pivo
						break;

					fingers[idTerm].docId = NO_MORE_DOC;
					position = fingers[idTerm].position + 1 + threadIdx.x;
					docLocal = -1;
					while(position < finalPositions[idTerm] && docLocal < pivotDoc){
						docLocal = iDocIdList[position];
						position += blockDim.x;
					}
					position -= blockDim.x;
					if((docLocal < pivotDoc ||  position >= finalPositions[idTerm])){
						docLocal = NO_MORE_DOC;
						position = NO_VALID_POSITION;
					}

					__syncthreads();

					atomicMin(&(fingers[idTerm].docId) , docLocal);

					__syncthreads();

					if(fingers[idTerm].docId == docLocal){
						fingers[idTerm].position = position;
					}
				}
			}

			__syncthreads();

			//Sort the terms in non decreasing order of DID
			sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

			//Select term pivot
			if(THREAD_MASTER){
				selectTermPivot_No_SharedMemory(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
				docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
				score = 0.0;
			}

			__syncthreads();

			if (SHAREDTHESHOLD == 1){//SHARED_READ
				if(THREAD_MASTER && documentTopk.padding == 0 && thresholdLocal > globalThreshold){
		//							atomicMaxD(&globalThreshold,thresholdLocal);
//					atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
					globalThreshold = thresholdLocal;
				}

				if(thresholdLocal < globalThreshold){
					thresholdLocal = globalThreshold;
				}
			}else if (SHAREDTHESHOLD == 2){ //TSHARED_WRITEREAD
				if(THREAD_MASTER && (documentTopk.padding < (iTopK >> 1)) && thresholdLocal > globalThreshold){
//					atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
		//							atomicMaxD(&globalThreshold,thresholdLocal);
					globalThreshold = thresholdLocal;
				}

				if((documentTopk.padding < (iTopK >> 1)))
					if(thresholdLocal < globalThreshold){
						thresholdLocal = globalThreshold;
					}
			}
		}

		sortLocalTopkDocAndStoreInGlobal(dTopkScoreListGlobal,iTopkDocListGlobal,iTopK,&documentTopk);



//		if(thresholdLocal > globalThreshold)
//			globalThreshold = thresholdLocal;

	//	if(THREAD_MASTER)
	//		atomicAdd(&globalCount,count);
	////
	//	if(THREAD_MASTER)
	//		printf("-----%d----", globalCount);
}

__global__ void matchWandParallel_VARIABLE_3(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlist, const float *dIdfList,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
										  const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* iDocNumberByTermList){
//		if(blockIdx.x != 1687)
//			return;

//		int count = 0;
		__shared__ pivot sharedPivot;
		__shared__ finger fingers[TERM_NUMBER];

		__shared__ postingList postingLists[TERM_NUMBER];
		__shared__ documentTopkList documentTopk;

		__shared__ unsigned int iOrderedTermSharedList[TERM_NUMBER];
//		__shared__ float dUBlist[TERM_NUMBER];

//		__shared__ int iDocNumberByTermList[TERM_NUMBER];


		__shared__ int iGlobalInitialPositionInList;
		__shared__ unsigned short int iElementQuantityByBlock;

		__shared__ float score;
		__shared__ bool isValidCandidate;
		__shared__ int docCurrent;

		__shared__ short int needSearchDocRange[TERM_NUMBER];
		__shared__ limitDocId limitDoc;

	 	float thresholdLocal;// = iInitialThreshold;

//	 	int count = 0;

	 	thresholdLocal = iInitialThreshold;

		int globalIndex;// = iInitialPositionGlobal + threadIdx.x;
		int localIndex;
		int positionInitialInTermPostingList;

		if(thresholdLocal < globalThreshold)
			thresholdLocal = globalThreshold;

		if(THREAD_MASTER){
			documentTopk.padding = iTopK;
		}else if(THREAD_MASTER_2){
			iElementQuantityByBlock = DOC_QUANTITY_IN_MEMORY;//iBlockRoundNumber * DOC_QUANTITY_IN_MEMORY;
			iGlobalInitialPositionInList = iElementQuantityByBlock  * blockIdx.x * iGlobalRoundNumber;
		}

		//Inicializa a lista de Score e Documentos dos Topk
		//Considero que o Top_K seja um número múltiplo do tamanho do bloco
		for (localIndex = threadIdx.x; localIndex < iTopK; localIndex += blockDim.x) {
			documentTopk.id[localIndex] = -1;
			documentTopk.score[localIndex] = 0.0;
		}

	//	if(THREAD_MASTER) documentTopk.padding = iTopK;

		__syncthreads();

		//Define o max e o min
		if(threadIdx.x < iTermNumber){
//			iDocNumberByTermList[threadIdx.x] = globalDocNumberByTermList[threadIdx.x];
			fingers[threadIdx.x].final = 0;
			limitDoc.extraPosition[threadIdx.x] = 0;
//			dUBlist[threadIdx.x] = dUBlistGlobal[blockIdx.x * iTermNumber + threadIdx.x];
			int docAmount = iDocNumberByTermList[threadIdx.x];
			globalIndex = iGlobalInitialPositionInList;
			positionInitialInTermPostingList = 0;

			for (int i = 0; i < threadIdx.x; ++i) {
				positionInitialInTermPostingList += iDocNumberByTermList[i];
			}
//			if(threadIdx.x == 0 && blockIdx.x == 1687){
//				printf("Oi");
//			}
			int aux, maxDoc;
			int maxNeighbor;
			if(blockIdx.x != 0){
				int maxDoc = (globalIndex < docAmount) ? iDocIdList[positionInitialInTermPostingList + globalIndex - 1] : -1;
				maxDoc++;
				aux = maxDoc;

//				atomicMax(&limitDoc.minDocId, maxDoc);
//				__syncwarp(0xFFFFFFFF);
				for (int i = 1; i < iTermNumber; ++i) {
					maxNeighbor = __shfl_sync(0xFFFFFFFF,aux,i);
					if(maxNeighbor > maxDoc)
						maxDoc = maxNeighbor;
				}
	//
				if(THREAD_MASTER) {
					limitDoc.minDocId = maxDoc; //atomicExch(&(limitDoc.minDocId), maxDoc);
				}
				__syncwarp(0xFFFFFFFF);

				if(aux < limitDoc.minDocId && aux != 0)
					needSearchDocRange[threadIdx.x] = 1;
			}else
				if(THREAD_MASTER) limitDoc.minDocId = 0;

			int isTail = globalIndex < docAmount;
			globalIndex += iElementQuantityByBlock * iGlobalRoundNumber;
			isTail &= globalIndex >= docAmount;

			if(isTail){
				globalIndex = iGlobalInitialPositionInList + (docAmount - iGlobalInitialPositionInList - 1);
			}

			maxDoc =  (isTail || globalIndex < docAmount) ? iDocIdList[positionInitialInTermPostingList +  globalIndex] - 1 :
															-1;
			aux = maxDoc;
			for (int i = 1; i < iTermNumber; ++i) {
				maxNeighbor = __shfl_sync(0xFFFFFFFF,aux,i);
				if(maxNeighbor > maxDoc)
					maxDoc = maxNeighbor;
			}
			if(THREAD_MASTER) limitDoc.secondMaxDocId = maxDoc;
		}

		__syncthreads();

		//Busca faixa de documentos;
		for (int termId = 0; termId < iTermNumber; ++termId) {
			if(needSearchDocRange[termId])
				searchRangeOfDocs(iDocIdList,postingLists, termId,
								  iGlobalInitialPositionInList, &limitDoc,
								  iElementQuantityByBlock,iGlobalRoundNumber,iDocNumberByTermList);
		}

		__syncthreads();

		//Preenche a memória compartilhada
		positionInitialInTermPostingList = 0;
		int docLocal, docAmount;
		for (int termId = 0; termId < iTermNumber; ++termId) {
			globalIndex = iGlobalInitialPositionInList + limitDoc.extraPosition[termId] + threadIdx.x;
			docAmount = iDocNumberByTermList[termId];
			docLocal = -1;

			for (localIndex = threadIdx.x; localIndex < iElementQuantityByBlock; localIndex+=blockDim.x) {

				docLocal = (globalIndex < docAmount) ? iDocIdList[positionInitialInTermPostingList + globalIndex]
				                                                  : NO_MORE_DOC;

				if(docLocal > limitDoc.secondMaxDocId || globalIndex > docAmount){
					postingLists[termId].docId[localIndex] = NO_MORE_DOC;
					fingers[termId].final = 1;
					break;
				}

				postingLists[termId].docId[localIndex] = docLocal;
				postingLists[termId].docLenght[localIndex] = iDocLenghtList[positionInitialInTermPostingList + globalIndex];
				postingLists[termId].freq[localIndex] = iFreqList[positionInitialInTermPostingList + globalIndex];

				globalIndex += blockDim.x;
			}

			positionInitialInTermPostingList += iDocNumberByTermList[termId];
		}

		if(threadIdx.x < iTermNumber){
			fingers[threadIdx.x].docId = postingLists[threadIdx.x].docId[0];
			fingers[threadIdx.x].position = (fingers[threadIdx.x].docId == NO_MORE_DOC) ? NO_VALID_POSITION : 0;
//			fingers[threadIdx.x].final = 0 | fingers[threadIdx.x].final;
		}

//		if(threadIdx.x == 0 && blockIdx.x == 3430){
//			printf("Oi");
//		}
		__syncthreads();

//		__shared__ int docCurrent;

		sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

	//	__syncthreads();

		if(THREAD_MASTER){
			selectTermPivot_2(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
			docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
		}else if(THREAD_MASTER_2){
			score = 0.0;
		}

		int padding;
		int threadIdInWarp = (threadIdx.x & 0x1f);
		int idWarp = ((blockDim.x >> 5) == 1 ) ? 1 :  threadIdx.x >> 5;

		__syncthreads();

		while((sharedPivot.positionInOrderedList < iTermNumber) && (sharedPivot.idTerm < iTermNumber)){


			if(THREAD_MASTER){
				isValidCandidate = (fingers[sharedPivot.idTerm].docId == fingers[iOrderedTermSharedList[0]].docId);
			}
			__syncthreads();

			if(isValidCandidate){

				if(threadIdx.x < iTermNumber){
					fullScore_3_1(&score, fingers[sharedPivot.idTerm].docId, iOrderedTermSharedList,
								  fingers,postingLists, dIdfList, dAverageDocumentLength);
				}

				padding = documentTopk.padding;

				__syncthreads();

/*				If the heap is not full
				the candidate is inserted into the heap. If the heap is full
				and the new score is larger than the minimum score in the
				heap, the new document is inserted into the heap, replacing
				the one with the minimum score.

 */
				if(padding != 0 || thresholdLocal < score ){
					thresholdLocal = managerMinValue_v5(&documentTopk, fingers[sharedPivot.idTerm].docId, score,padding);
				}

				if(idWarp == 1 && threadIdInWarp < iTermNumber ){
					advancePivoTermFinger_4(sharedPivot,fingers, postingLists,iElementQuantityByBlock,threadIdInWarp);
				}
			}
			else{
				 advanceDocIdOfPredecessorTerm_4(postingLists,
											   iOrderedTermSharedList,
											   fingers,sharedPivot,fingers[sharedPivot.idTerm].docId,
											   iElementQuantityByBlock);
			}

			__syncthreads();

			for (int termId = 0; termId < iTermNumber; ++termId) {
				if(fingers[termId].docId == NO_MORE_DOC && 	fingers[termId].final == 0){
//					if(termId == 0) count++;
//
//					if(blockIdx.x == 27 && count == 48 && THREAD_MASTER)
//						printf("Oi!");

					searchMoreDocs(iDocIdList,iFreqList,iDocLenghtList,postingLists,
								  termId,iGlobalInitialPositionInList,
								  &limitDoc,iElementQuantityByBlock,
								  &(fingers[termId]),docCurrent,iDocNumberByTermList);


					//#endif

					if (SHAREDTHESHOLD == 1){//SHARED_READ
						if(THREAD_MASTER && documentTopk.padding == 0 && thresholdLocal > globalThreshold){
//							atomicMaxD(&globalThreshold,thresholdLocal);
							atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
						}

						if(thresholdLocal < globalThreshold){
							thresholdLocal = globalThreshold;
						}
					}else if (SHAREDTHESHOLD == 2){ //TSHARED_WRITEREAD
						if(THREAD_MASTER && (documentTopk.padding < (iTopK >> 1)) && thresholdLocal > globalThreshold){
							atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
//							atomicMaxD(&globalThreshold,thresholdLocal);
						}

						if((documentTopk.padding < (iTopK >> 1)))
							if(thresholdLocal < globalThreshold){
								thresholdLocal = globalThreshold;
							}
					}
					//#endif

				}
			}


	//		__syncthreads();//Talvez não precise
			//Sort the terms in non decreasing order of DID
			sortingTerms_2(fingers, iOrderedTermSharedList, iTermNumber);

	//		__syncthreads();//Talvez não precise

			//Select term pivot
			if(THREAD_MASTER){
				selectTermPivot_2(&sharedPivot,iOrderedTermSharedList,fingers,dUBlist,iTermNumber,thresholdLocal);
				docCurrent = (sharedPivot.idTerm != NO_PIVOT_TERM) ? fingers[sharedPivot.idTerm].docId : NO_MORE_DOC;
				score = 0.0;
			}
			__syncthreads();
		}

//		if(threadIdx.x == 0){// && blockIdx.x == 1687){
//			printf("----%d %d----",blockIdx.x,count);
//		}

		sortLocalTopkDocAndStoreInGlobal(dTopkScoreListGlobal,iTopkDocListGlobal,iTopK,&documentTopk);
//		globalIndex =  iTopK * blockIdx.x + threadIdx.x + documentTopk.padding;
//		for (localIndex = threadIdx.x; localIndex < (iTopK - documentTopk.padding) ; localIndex += blockDim.x) {
//			iTopkDocListGlobal[globalIndex]   = documentTopk.id[localIndex];
//			dTopkScoreListGlobal[globalIndex] = documentTopk.score[localIndex];
//			globalIndex += blockDim.x;
//		}
//		__syncthreads();

		if(THREAD_MASTER && thresholdLocal > globalThreshold){
			atomicMax((unsigned long long int*)&globalThreshold,(unsigned long long int)thresholdLocal);
		}
}
