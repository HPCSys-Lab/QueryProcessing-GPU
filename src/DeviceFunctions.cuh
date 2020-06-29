/*
 * DeviceFunctions.cuh
 *
 *  Created on: 08/12/2017
 *      Author: roussian
 */

#ifndef DEVICEFUNCTIONS_CUH_
#define DEVICEFUNCTIONS_CUH_

#include "Structs.cuh"

__device__  void selectTermPivot_No_SharedMemory(pivot* pivot,
								   unsigned int *iOrderedTermListShared,
								   finger* fingers,
								   const float *dUBlist,
								   int iTermNumber,
								   float threshold);

__device__ void selectTermPivot_2(pivot* pivot,unsigned int *iOrderedTermListShared, finger* fingers,
								  const float *dUBlist, int iTermNumber, float threshold);

__device__ void sortingTerms_2(finger* fingers, unsigned int *iOrderedTermListShared, const short int iTermNumber);

__device__ void advancePivoTermFinger_2(pivot pivot, finger *fingers,postingList *postingLists);

__device__ void advancePivoTermFinger_4(pivot pivot, finger *fingers, postingList *postingLists,
										unsigned short int iElementQuantityByBlock,	int threadIdInWarp);

__device__ void advanceDocIdOfPredecessorTerm_3(postingList *postingList, unsigned int *iOrderedTermListShared,
											    finger *fingers, pivot pivot,  const int iBlockRoundNumber);

__device__ void advanceDocIdOfPredecessorTerm_4(postingList *postingList, unsigned int *iOrderedTermListShared,finger *fingers,
 											    pivot pivot, int docPivot, const unsigned short int iElementQuantityByBlock);

__device__ void fullScore_3(float *score, pivot pivot, unsigned int *iOrderedTermSharedList, finger *fingers,
							postingList *postingList, const float *dIdfList, const float dAverageDocLength);

__device__ void fullScore_3_1(float *score, int docPivot, unsigned int *iOrderedTermSharedList, finger *fingers,
							  postingList *postingList, const float *dIdfList, const float dAverageDocLength);

__device__ float managerMinValue_v4(documentTopkList *documentTopk, int newDocId, float newScore, int padding);
__device__ float managerMinValue_v5(documentTopkList *documentTopk, int newDocId, float newScore, int padding);

__device__ void searchRangeOfDocs(const int* iDocIdList, postingList *postingLists, int termId, int iGlobalInitialPositionInList,
								 limitDocId *limitDoc, unsigned short int iElementQuantityByBlock,int roundGlobalNumber,const int *iDocNumberByTermList);

__device__ void searchRangeOfDocs_batch(const int* iDocIdList,
		  	  	  	  	  	  	  	  	postingList *postingLists, int termId,
		  	  	  	  	  	  	  	  	int iGlobalInitialPositionInList, limitDocId *limitDoc,
		  	  	  	  	  	  	  	  	unsigned short int iElementQuantityByBlock,
		  	  	  	  	  	  	  	  	int roundGlobalNumber,int iPostingListSize, long long positionInListGlobal);

__device__ void  searchMoreDocs_batch(const int* iDocIdList, const unsigned short int* iFreqList,
									 const int *iDocLengthList, postingList *postingLists, int termId,
									 int iGlobalInitialPositionInList, limitDocId *limitDoc,
									 unsigned short int iElementQuantityByBlock, finger *fingerT,
									 int docCurrent, int iPostingListSize, long long positionInListGlobal);


__device__ void searchMoreDocs(const int* iDocIdList, const unsigned short int* iFreqList,  const int *iDocLengthList,
		  	  	  	  	  	   postingList *postingLists, int termId, int iGlobalInitialPositionInList, limitDocId *limitDoc,
		  	  	  	  	  	   unsigned short int iElementQuantityByBlock, finger *fingerT, int docCurrent,const int *iDocNumberByTermList);

__device__ void sortLocalTopkDocAndStoreInGlobal( float *dTopkScoreListGlobal,int *iTopkDocListGlobal, int iTopk,
												documentTopkList *localTopkDoc);

__device__ void sortLocalTopkDocAndStoreInGlobal_BLOCK( float *dTopkScoreListGlobal,int *iTopkDocListGlobal, int iTopk,
												documentTopkList *localTopkDoc);


__device__  double atomicMaxD(double volatile *address, double volatile val);


__device__   float scoreTf_Idf(int tf, int dDocLength, float idf, float averageDocumentLength, float keyFrequency);

#endif /* DEVICEFUNCTIONS_CUH_ */
