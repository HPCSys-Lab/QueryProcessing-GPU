/*
 * FileManager.c
 *
 *  Created on: 06/12/2017
 *      Author: roussian
 */

#include "FileManager.cuh"
#include "FileLocation.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void readQuery(int* iTermNumber, float* dAverageDocLength,  int* iTopK,
			   int*** h_iDocIdList, int*** h_iLenghtList,  unsigned short int*** h_iFreqList,
			   float** h_dIdfList, float** h_dUBList, int** h_iDocNumberList, int experimentNumber){

//	printf("--------------------->");
//	printf("PATH: %s\n",INVERTED_LIST_FILE);

//	char experiment_number[] ="";
//	sprintf(experiment_number, "%d", experimentNumber);

	const char *fileName = INVERTED_LIST_FILE;

//	sprintf(fileName,INVERTED_LIST_FILE);
	//strncat(fileName, experiment_number,1);

	char line[100];
//	char* token;
	int sizeLine = 100;

	int i = 0;
	int count = 0;

	FILE *fp;

	char* result;
	fp = fopen(fileName,"r");

//	const char s[2] = " ";

	if (!fp) {
	  printf( "Where is the file?\n");
	  exit( EXIT_FAILURE);
	}

	result = fgets(line,sizeLine,fp);
	sscanf(line, "%d", iTermNumber);
	if(result == NULL) printf("Erro in FileManager!!!\n");

	//---------------1ª Part of the list allocation------------------------
	*h_iDocIdList = (int**) malloc(*iTermNumber * sizeof(int*));
	*h_iLenghtList = (int**) malloc(*iTermNumber * sizeof(int*));
	*h_iFreqList = (unsigned short**) malloc(*iTermNumber * sizeof(unsigned short int*));

	*h_dIdfList = (float*) malloc(*iTermNumber * sizeof(float));
	*h_dUBList  = (float*) malloc(*iTermNumber * sizeof(float));

	*h_iDocNumberList = (int*) malloc(*iTermNumber *sizeof(int));
	//------------------------------------------------------------------

	result = fgets(line,sizeLine,fp);
	sscanf(line, "%f", dAverageDocLength);

//	fgets(line,sizeLine,fp);
//	sscanf(line, "%d", iTopK);

	do{
		 result = fgets(line,sizeLine,fp);
		 sscanf(line, "%f", &((*h_dIdfList)[i]));

		 result = fgets(line,sizeLine,fp);
		 sscanf(line, "%f", &((*h_dUBList)[i]));

		 result = fgets(line,sizeLine,fp);
		 sscanf(line, "%d", &((*h_iDocNumberList)[i]));
//		 printf("#Doc no termo %i: %i. \n",i, (*h_iDocNumberList)[i]);

		 //---------------2ª Part of the list allocation------------------------
		 (*h_iDocIdList)[i] = (int*) malloc((*h_iDocNumberList)[i] * sizeof(int));
		 (*h_iLenghtList)[i] = (int*) malloc((*h_iDocNumberList)[i] * sizeof(int));
		 (*h_iFreqList)[i] = (unsigned short*) malloc((*h_iDocNumberList)[i] * sizeof(unsigned short));
		 //--------------------------------------------------------------------

		 result = fgets(line,sizeLine,fp);

		 count = 0;
		 while(*line != '-'){
			 sscanf(line,"%d %hu %d", &((*h_iDocIdList)[i][count]), &((*h_iFreqList)[i][count]), &((*h_iLenghtList)[i][count]));
			 result = fgets(line,sizeLine,fp);
			 count++;
		 }

//		 printf("Count no termo %i: %i. \n",i, count);
		 i++;
	}while ((!feof(fp)) && (i < *iTermNumber));

	fflush(fp);
	fclose(fp);
}

/*
 * Regra na construção do índice invertido: A posição do termo == A id do Termo
 */

void readInvertedList(int* iTermNumberInVocabulary, float* dAverageDocLength,
					  int*** h_iDocIdList, int*** h_iLenghtList,
					  unsigned short int*** h_iFreqList, float** h_dIdfList,
					  float** h_dUBList, int** h_iDocNumberList, unsigned long long* docTotalNumber){

	int i=0, numberReadPosting=0, sizeLine = 100;
//	char fileName[24] = INDEX_FILENAME, line[sizeLine];
	char fileName[100] = INVERTED_LIST_FILE, line[sizeLine];


	FILE *fp = fopen(fileName,"r");

	char* result;
	if (!fp) {
	  printf( "Where is the file of inverted list?\n");
	  exit( EXIT_FAILURE);
	}

	result = fgets(line,sizeLine,fp);
	sscanf(line, "%d", iTermNumberInVocabulary);

	if(result == NULL) printf("Erro in FileManager!!!\n");

	result = fgets(line,sizeLine,fp);
	sscanf(line, "%f", dAverageDocLength);

	//---------------1ª Part of the list allocation------------------------
	*h_iDocIdList = (int**) malloc(*iTermNumberInVocabulary * sizeof(int*));
	*h_iLenghtList = (int**) malloc(*iTermNumberInVocabulary * sizeof(int*));
	*h_iFreqList = (unsigned short**) malloc(*iTermNumberInVocabulary * sizeof(unsigned short int*));

	*h_dIdfList = (float*) malloc(*iTermNumberInVocabulary * sizeof(float));

	*h_dUBList  = (float*) malloc(*iTermNumberInVocabulary * sizeof(float));
	*h_iDocNumberList = (int*) malloc(*iTermNumberInVocabulary *sizeof(int));
	//------------------------------------------------------------------

	do{
		int idTerm;
		result = fgets(line,sizeLine,fp);
		sscanf(line, "%d", &idTerm);
		if(result == NULL) printf("Error in FileManager!!!\n");

		result = fgets(line,sizeLine,fp);
		sscanf(line, "%f", &((*h_dIdfList)[idTerm]));
		if(result == NULL) printf("Error in FileManager!!!\n");

		result = fgets(line,sizeLine,fp);
		sscanf(line, "%f", &((*h_dUBList)[idTerm]));
		if(result == NULL) printf("Error in FileManager!!!\n");

		result = fgets(line,sizeLine,fp);
		sscanf(line, "%d", &((*h_iDocNumberList)[idTerm]));
		if(result == NULL) printf("Error in FileManager!!!\n");

		*docTotalNumber += (*h_iDocNumberList)[idTerm];
		 //---------------2ª Part of the list allocation------------------------
		(*h_iDocIdList)[idTerm] = (int*) malloc((*h_iDocNumberList)[idTerm] * sizeof(int));
		(*h_iLenghtList)[idTerm] = (int*) malloc((*h_iDocNumberList)[idTerm] * sizeof(int));
		(*h_iFreqList)[idTerm] = (unsigned short*) malloc((*h_iDocNumberList)[idTerm] * sizeof(unsigned short));
		 //--------------------------------------------------------------------

		result = fgets(line,sizeLine,fp);

		numberReadPosting = 0;
		while(!feof(fp) && *line != '-' && line != NULL){
			sscanf(line,"%d %hu %d", &((*h_iDocIdList)[idTerm][numberReadPosting]), &((*h_iFreqList)[idTerm][numberReadPosting]), &((*h_iLenghtList)[idTerm][numberReadPosting]));
			result = fgets(line,sizeLine,fp);
			numberReadPosting++;
		}
		i++;
	}while ((!feof(fp)) && (i < *iTermNumberInVocabulary));

	fflush(fp);
	fclose(fp);
}

/*
 * Faz a leitura do batch da query e armazena na queryBatch.
 * Esse método precisa melhorar no futuro, pois faz a leitura de forma primitiva.
 */
void readQueryBatch(int ***queryBatch, int *queryNumber, int **h_iTermNumberList){

	char fileName[32] = QUERY_BATCH_TRANSCRIPT;
	FILE *fp;
	char* result;
	int sizeLine = 100;
	char *linePtr = (char *) malloc(sizeLine * sizeof(char));//[sizeLine];
	char *line = linePtr, *saveptr, *token;

	int idQuery=0, queryIdTerm, indexTerm, termNumberByQuery;

	fp = fopen(fileName,"r");
	if (!fp) {
	  printf( "Where is the file of query batch?\n");
	  exit( EXIT_FAILURE);
	}
	result = fgets(line,sizeLine,fp);
	if(result == NULL) printf("Error in FileManager!!!\n");

	*queryBatch = (int**) malloc(*queryNumber * sizeof(int*));
	*h_iTermNumberList = (int*) malloc(*queryNumber * sizeof(int*));
	do {
		indexTerm = 0;
		termNumberByQuery = 1;
		//Faz a leitura da query considerando que há espaço a cada idTerm
		for (int i = 0; line[i] != '\0'; ++i) {
	       if(line[i] == ' ')
	           termNumberByQuery++;
		}
		(*h_iTermNumberList)[idQuery] = termNumberByQuery;
		(*queryBatch)[idQuery] = (int*) malloc(termNumberByQuery * sizeof(int));

		do {
			token = strtok_r(line, QUERY_DELIMITER, &saveptr);
			sscanf(token, "%d", &queryIdTerm);

			(*queryBatch)[idQuery][indexTerm]  = queryIdTerm;
			indexTerm++;
			line = NULL;
		} while (indexTerm < termNumberByQuery);

		line = linePtr;
		result = fgets(line,sizeLine,fp);
		idQuery++;

	}while ( (!feof(fp)) && (line != NULL) && (idQuery < *queryNumber) );

	free(linePtr);
//	*queryNumber = idQuery;
}
