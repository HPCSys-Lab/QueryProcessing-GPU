/*
 * FileManager.h
 *
 *  Created on: 07/12/2017
 *      Author: roussian
 */

#ifndef FILEMANAGER_H_
#define FILEMANAGER_H_

void readQuery(int* iTermNumber, float* dAverageDocLength,  int* iTopK,
			   int*** h_iDocIdList, int*** h_iLenghtList,  unsigned short int*** h_iFreqList,
			   float** h_dIdfList, float** h_dUBList, int** h_iDocNumberList, int experimentNumber);


void readInvertedList(int* iTermNumber, float* dAverageDocLength,
					  int*** h_iDocIdList, int*** h_iLenghtList,
					  unsigned short int*** h_iFreqList,float** h_dIdfList, float** h_dUBList,
					  int** h_iDocNumberList, unsigned long long* docTotalNumber);

void readQueryBatch(int ***queryBatch, int *queryNumber, int **h_iTermNumberList);

#endif /* FILEMANAGER_H_ */
