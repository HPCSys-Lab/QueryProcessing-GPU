/*
 * Structs.cuh
 *
 *  Created on: 08/12/2017
 *      Author: roussian
 */

#ifndef STRUCTS_CUH_
#define STRUCTS_CUH_

#define THREAD_MASTER threadIdx.x == 0
#define THREAD_MASTER_2 threadIdx.x == 32
#define THREAD_FINAL threadIdx.x == blockDim.x - 1
#define INITIAL_VALUE -1

#define NO_PIVOT_TERM 2147483641
#define NO_MORE_DOC 2147483641
#define NO_VALID_POSITION 1223372036854775809//65535
#define DOCS_TEST 64
#ifndef SHAREDTHESHOLD
	#define SHAREDTHESHOLD 0
#endif

#ifndef TOP_K
	#define TOP_K 128
#endif

#define HEIGHT_HEAP 7//log2f(TOP_K)

#ifndef DOC_QUANTITY_IN_MEMORY
	#define DOC_QUANTITY_IN_MEMORY 512
#endif

#ifndef TERM_NUMBER
	#define TERM_NUMBER 2
#endif



typedef struct limitDocId{
	int minDocId;
	int firstMaxDocId;
	int secondMaxDocId;
	int extraPosition[TERM_NUMBER];
} limitDocId;

typedef struct postingList{
	int docId[DOC_QUANTITY_IN_MEMORY];
	int docLenght[DOC_QUANTITY_IN_MEMORY];
	unsigned short int freq[DOC_QUANTITY_IN_MEMORY];
	short int maxIndex;
	short int processedDocNumber;
} postingList;

typedef struct postingList2{
	int docId[DOCS_TEST];
	int docLenght[DOCS_TEST];
	unsigned short int freq[DOCS_TEST];
	long long positionInShared;
} postingList2;

typedef struct documentTopkList{
	int id[TOP_K];
	float score[TOP_K];
	int padding;
} documentTopkList;

typedef struct pivot{
	unsigned int idTerm;
	unsigned int positionInOrderedList;
} pivot;

typedef struct finger{
	long long position;
	int docId;
	int final;
} finger;



#endif /* STRUCTS_CUH_ */
