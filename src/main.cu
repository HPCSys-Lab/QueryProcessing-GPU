/*
 * main.c
 *
 *  Created on: 06/12/2017
 *      Author: roussian
 */
#include "HostManager.cuh"
#include <stdio.h>

int main(int argc, char *argv[])
{
//	cudaSetDevice(0);
	//Argumentos
	if( argc < 5 ) {
		printf( "\n Parametros incorretos.\n Uso: <top_K>, <blockSize>, <BlockRoundNumber>, <iGlobalNumberRound>,"
				" <MergeNumberByBlock> <QueryType> onde: \n" );
		printf( "\t <top_K> - quantidade de documentos retornados (precisa ser multiplo do blockSize).\n" );
		printf( "\t <blockSize> - tamanho do bloco.\n" );
		printf( "\t <BlockRoundNumber> - numero de partes continuas que cada bloco ira processar.\n" );
		printf( "\t <GlobalNumberRound> - numero de partes nao continuas das listas invertidas que cada bloco ira processar.\n" );
		printf( "\t <MergeNumberByBlock> - numero de merge que cada bloco irá executar.\n" );
		printf( "\t <QueryType> (Optional) - [0] OR Query  --- [1] AND Query.\n" );
        return 0;
	}

	//Quantidade de postings em cada lista em função do tamanho do bloco que cada bloco irá processar
	int iTopk = atoi( argv[1] );
	int iBlockSize = atoi( argv[2] );
	int iBlockNumberRound = atoi( argv[3] );
	int iGlobalNumberRound = atoi( argv[4] );
	int iMergeNumberByBlock = atoi( argv[5] );
	int iQueryType = 0;

	if(argc == 7)
	   iQueryType = atoi( argv[6] );


//	#ifdef BATCH
//		queryBatchProcessingHost_Mix(iTopk, iBlockSize, iBlockNumberRound, iGlobalNumberRound, iMergeNumberByBlock, iQueryType);
//	#else
	querySingleProcessingHost(iTopk, iBlockSize, iBlockNumberRound, iGlobalNumberRound, iMergeNumberByBlock,  iQueryType, 1);
//	#endif

//	queryBatchProcessingHost_ByBlock(iTopk, iBlockSize, iBlockNumberRound,
//							 iGlobalNumberRound, iMergeNumberByBlock, iQueryType,500);


	exit(EXIT_SUCCESS);
}

