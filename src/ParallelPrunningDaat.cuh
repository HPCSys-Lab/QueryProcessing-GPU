/*
 * ParallelPrunningDaat.cuh
 *
 *  Created on: 08/12/2017
 *      Author: roussian
 */

#ifndef PARALLELPRUNNINGDAAT_CUH_
#define PARALLELPRUNNINGDAAT_CUH_
#include "Structs.cuh"


//__constant__ int iDocNumberByTermListConstant[TERM_NUMBER];

__global__ void preProcessingWand(const int* iDocIdList,
								  const short int iTermNumber,
								  const int* d_iDocNumberByTermList,
								  const int* iInitialPositionPostingList,
								  const int docIdNumberByBlock,
								  int* extraPositions, int* docMaxList);

__global__ void matchWandParallel_VARIABLE_4(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlist, const float *dIdfList,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
										  const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* iDocNumberByTermList);

__global__ void matchWandParallel_VARIABLE_4_2(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlist, const float *dIdfList,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
										  const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* iDocNumberByTermList);

__global__ void matchWandParallel_VARIABLE_3_Teste(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlist, const float *dIdfList,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
										  const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* d_iDocNumberByTermList,
										  const int* extraPositions, const int* docMaxList);

__global__ void matchWandParallel_FIXED_3(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlistGlobal, const float *dIdfListGlobal,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
										  const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* iDocNumberByTermListGlobal);

__global__ void matchWandParallel_FIXED_2(const int* iDocIdList, const unsigned short int* iFreqList,
								  const float *dUBlist, const float *dIdfList,
								  const int *iDocLenghtList,
								  const short int iTermNumber, int *iTopkDocListGlobal,
								  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
								  const int iGlobalRoundNumber, //const int iBlockRoundNumber,
								  const short int iTopK,
								  const float iInitialThreshold, const int* d_iDocNumberByTermList);

__global__ void matchWandParallel_VARIABLE_3(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlist, const float *dIdfList,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
										  const int iGlobalRoundNumber, const short int iTopK,
										  const float iInitialThreshold,const int* d_iDocNumberByTermList);

__global__ void matchWandParallel_BATCH(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlist, const float *dIdfList,
										  const int *iDocLenghtList, const short int iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
										  const int iBlockRoundNumber, const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* iQueryTerms, const long long* ptrPostingPositions,
										  int* d_ptrQueryPositions, int idQuery,int *iDocNumberByTermList);

__global__ void matchWandParallel_BATCH_2(const int* iDocIdList, const unsigned short int* iFreqList,
										const float *dUBlistGlobal, const float *dIdfListGlobal, const int *iDocLenghtList,
										const int iTermNumber, int *iTopkDocListGlobal,
										float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
										const int iBlockRoundNumber, const int iGlobalRoundNumber,
										const short int iTopK, const float iInitialThreshold,
										const int* iQueryTerms, const long long* ptrInitPostingList,
										int* ptrQueryPositions, int idQuery,int *iDocNumberByTermListGlobal);

__global__ void matchWandParallel_VARIABLE_Batch_Block(const int* iDocIdList, const unsigned short int* iFreqList,
										  const float *dUBlist, const float *dIdfList, const int *iDocLenghtList,
										  const short int *iTermNumber, int *iTopkDocListGlobal,
										  float *dTopkScoreListGlobal, const float dAverageDocumentLength,
										  const int iBlockRoundNumber, const int iGlobalRoundNumber,
										  const short int iTopK, const float iInitialThreshold,
										  const int* iQueryTerms, const long long* ptrPostingPositions,
										  int* ptrQueryPositions, int *iDocNumberByTermList);

__global__ void matchWandParallel_VARIABLE_Batch_Block_2(const int* iDocIdList, const unsigned short int* iFreqList,
														const float *dUBlistGlobal, const float *dIdfListGlobal, const int *iDocLenghtList,
														const short int* iTermNumber, int *iTopkDocListGlobal,
														float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
														const short int iTopK, const float iInitialThreshold,
														const int* iQueryTerms, const long long* ptrInitPostingList,
														int* ptrQueryPositions, int *iDocNumberByTermListGlobal);

__global__ void matchWandParallel_VARIABLE_Batch_Block_3(const int* iDocIdList, const unsigned short int* iFreqList,
														const float *dUBlistGlobal, const float *dIdfListGlobal, const int *iDocLenghtList,
														const short int* iTermNumber, int *iTopkDocListGlobal,
														float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
														const short int iTopK, const float iInitialThreshold,
														const int* iQueryTerms, const long long* ptrInitPostingList,
														int* ptrQueryPositions, int *iDocNumberByTermListGlobal);

__global__ void matchWandParallel_VARIABLE_Batch_Block_Test(const int* iDocIdList, const unsigned short int* iFreqList,
														const float *dUBlistGlobal, const float *dIdfListGlobal, const int *iDocLenghtList,
														const short int* iTermNumberByQuery, int *iTopkDocListGlobal,
														float *dTopkScoreListGlobal, const float dAverageDocumentLengthGlobal,
														const short int iTopK, const float iInitialThreshold,
														const int* iQueryTerms, const long long* ptrInitPostingList,
														int* ptrQueryPositions,int *iDocNumberByTermListGlobal,
														const int* iOrderQueryList);


__global__ void mergeTopkLists_v3(float *dTopkScoreList,
								  int *iTopkDocList,
								  int iTopk,
								  int iMergeNumber,
								  int iSkipTopkBetweenMerges,
								  int iSkipTopkBetweenBlocks,
								  int iTotalElementos);

#endif /* PARALLELPRUNNINGDAAT_CUH_ */
