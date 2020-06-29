CC=nvcc

SRCDIR=/home/roussian.gaioso/ParallelPrunningDaat/src
PROJECT_NAME=ParallelPrunningDAAT
															     

PROGFLAGS=-DTOP_K=$(K) -DTERM_NUMBER=$(TERMS) $(BATCH) -DDOC_QUANTITY_IN_MEMORY=$(DOCSBYMEM) $(DEBUG) -DSHAREDTHESHOLD=$(THRESHOLD) -DINVERTED_LIST_FILE=\"$(INVERTED_FILE)\"
CFLAGS=--device-c -arch=sm_61 --use_fast_math -O3 -Xcompiler -fopenmp#-maxrregcount 32 #-O3 --compiler-options -Wall #-gencode arch=compute_35,code=sm_35 -I
LFLAGS=--cudart static --relocatable-device-code=true -arch=sm_61 -link -Xcompiler -fopenmp#-gencode arch=compute_35,code=sm_35 -link

#WITHOUTSHARED 0
#TSHARED_READ 1
#TSHARED_WRITEREAD 2
#TOP_K=128
#TERM_NUMBER
#DOC_QUANTITY_IN_MEMORY

#DEPS = $(pastsubst %,$(SRCDIR)/%, DeviceFunctions.cuh FileManager.cuh FileLocation.cuh HostManager.cuh ParallelPrunningDaat.cuh Structs.cuh)
#DEPS=$(pastsubst %,$(SRCDIR)/%, $(wildcard *.cuh))
#SRC=$(pastsubst %,$(SRCDIR)/%, $(wildcard *.cu))
DEPS=$(wildcard $(SRCDIR)*.cuh)
SRC=$(wildcard $(SRCDIR)*.cu)
OBJ= main.o FileManager.o DeviceFunctions.o HostManager.o  ParallelPrunningDaat.o UnityTest.o

all: $(PROJECT_NAME)

$(PROJECT_NAME): $(OBJ) 
	$(CC) $(LFLAGS) -o $@ $^

#%.o: $(SRCDIR)/%.c $(DEPS)
#	$(CC) $(CFLAGS) $(PROFLAGS) -o $@ $< 

FileManager.o: $(SRCDIR)/FileManager.cu $(SRCDIR)/FileManager.cuh $(SRCDIR)/FileLocation.cuh $(SRCDIR)/Structs.cuh
	$(CC) $(CFLAGS) $(PROGFLAGS) -o $@ $<

DeviceFunctions.o: $(SRCDIR)/DeviceFunctions.cu $(SRCDIR)/DeviceFunctions.cuh $(SRCDIR)/Structs.cuh
	$(CC) $(CFLAGS) $(PROGFLAGS) -o $@ $<

UnityTest.o: $(SRCDIR)/UnityTest.cu $(SRCDIR)/UnityTest.cuh $(SRCDIR)/Structs.cuh
	$(CC) $(CFLAGS) $(PROGFLAGS) -o $@ $<

ParallelPrunningDaat.o: $(SRCDIR)/ParallelPrunningDaat.cu $(SRCDIR)/ParallelPrunningDaat.cuh $(SRCDIR)/Structs.cuh DeviceFunctions.o UnityTest.o
	$(CC) $(CFLAGS) $(PROGFLAGS) -o $@ $< 

HostManager.o: $(SRCDIR)/HostManager.cu $(SRCDIR)/HostManager.cuh $(SRCDIR)/Structs.cuh ParallelPrunningDaat.o
	$(CC) $(CFLAGS) $(PROGFLAGS) -o $@ $<

main.o: $(SRCDIR)/main.cu HostManager.o
	$(CC) $(CFLAGS) $(PROGFLAGS) -o $@ $<

.PHONY: all clean


clean:
	rm *.o $(PROJECT_NAME)


#nvcc --compile --relocatable-device-code=true -O3 -gencode arch=compute_35,code=sm_35 ../../ParallelPrunningDAAT/src/DeviceFunctions.cu
#nvcc --compile --relocatable-device-code=true -O3 -gencode arch=compute_35,code=sm_35 ../../ParallelPrunningDAAT/src/FileManager.cu
#nvcc --compile --relocatable-device-code=true -O3 -gencode arch=compute_35,code=sm_35 ../../ParallelPrunningDAAT/src/HostManager.cu
#nvcc --compile --relocatable-device-code=true -O3 -gencode arch=compute_35,code=sm_35 ../../ParallelPrunningDAAT/src/ParallelPrunningDaat.cu
#nvcc --compile --relocatable-device-code=true -O3 -gencode arch=compute_35,code=sm_35 ../../ParallelPrunningDAAT/src/main.cu 

#/usr/local/cuda-8.0/bin/nvcc -DDEBUG -O3 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "src/ParallelPrunningDaat.o" "../src/ParallelPrunningDaat.cu"

#nvcc --cudart static --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -link -o  "ParallelPrunningDAAT"  ./src/DeviceFunctions.o ./src/FileManager.o ./src/HostManager.o ./src/ParallelPrunningDaat.o ./src/main.o



#nvcc --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52 ../../ParallelPrunningDAAT/src/ParallelPrunningDaat.cu
#nvcc --compile --relocatable-device-code=true -arch=sm_52  -DTOP_K=128 -DTERM_NUMBER=2 -DDOC_QUANTITY_IN_MEMORY=128 -DTSHARED_WRITEREAD  -o ParallelPrunningDaat.o /home/roussian.gaioso/ParallelPrunningDAAT/src/ParallelPrunningDaat.cu


