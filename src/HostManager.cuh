/*
 * HostManager.cuh
 *
 *  Created on: 11/12/2017
 *      Author: roussian
 */

#ifndef HOSTMANAGER_CUH_
#define HOSTMANAGER_CUH_

__host__ void querySingleProcessingHost(int iTopk, int iBlockSize, int iBlockNumberRound,
										int iGlobalNumberRound, int iMergeNumberByBlock,
										int iQueryType, int iExperimentNumber);

__host__ void queryBatchProcessingHost_Mix(int iTopk, int iBlockSize, int iBlockNumberRound,
										int iGlobalNumberRound, int iMergeNumberByBlock,
										int iQueryType);


__host__ void queryBatchProcessingHost(int iTopk, int iBlockSize, int iBlockNumberRound,
										int iGlobalNumberRound, int iMergeNumberByBlock,
										int iQueryType);

__host__ void queryBatchProcessingHost_2(int iTopk, int iBlockSize, int iBlockNumberRound,
										int iGlobalNumberRound, int iMergeNumberByBlock,
										int iQueryType);

__host__ void queryBatchProcessingHost_ByBlock(int iTopk, int iBlockSize, int iBlockNumberRound,
											   int iGlobalNumberRound, int iMergeNumberByBlock,
											   int iQueryType, int iBatchSize);

__host__ void querySingleProcessingHost_Teste(int iTopk, int iBlockSize, int iBlockNumberRound,
											  int iGlobalNumberRound, int iMergeNumberByBlock,
											  int iQueryType, int iExperimentNumber);

#endif /* HOSTMANAGER_CUH_ */
