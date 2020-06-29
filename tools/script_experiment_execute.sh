#!/bin/bash
trap "exit" INT

#for entry in `ls /home/roussian.gaioso/experiments/input/invertedList`; do #postingList_2t_20000_
#  for topk in 128 #1024 
#     do
#	echo -------------Top-k:$topk--------------------------
# 	for docsnumber in 32 64 128 #128 256  #256 512 
#	do
#  		echo -------------DOCSNUMBER:$docsnumber--------------------------
#		for threshold in 0 1 2
#		do
#		  echo -------------THRESHOLD:$threshold--------------------------
#		
#			for threads in 64 #256 512
#			do	
#				echo -------------threads:$threads--------------------------
#				for partition_number in 1  2 5 10 15 20 50  #20 #50 100
#				do
#				   make clean
#				   make K=$topk TERMS=5 DOCSBYMEM=$docsnumber  THRESHOLD=$threshold INVERTED_FILE=$entry DEBUG="-maxrregcount 128" BATCH="-DBATCH" #DEBUG="-DDEBUG"
#				   ./ParallelPrunningDAAT $topk $threads 1 $partition_number 31
#				done 
#			done
 #  		done	
#	done	
 #  done
#done



for entry in `ls /home/roussian.gaioso/experiments/input/postingList_2t_*`; do #postingList_2t_20000_
   echo $entry
   for topk in 128 #1024 
     do
	echo -------------Top-k:$topk--------------------------
  	for docsnumber in 32 64 128 #256 512 
	do
  		echo -------------DOCSNUMBER:$docsnumber--------------------------
		for threshold in 0 1 2
		do
		  echo -------------THRESHOLD:$threshold--------------------------
		
			for threads in 32 #256 512
			do	
				echo -------------threads:$threads--------------------------
				for partition_number in 1 10 100  #20 #50 100
				do
				   make clean
				   make K=$topk TERMS=2 DOCSBYMEM=$docsnumber  THRESHOLD=$threshold INVERTED_FILE=$entry DEBUG="-maxrregcount 128" #DEBUG="-DDEBUG -g -G"#
				   ./ParallelPrunningDAAT $topk $threads 1 $partition_number 1
				done 
			done
   		done	
	done	
   done
done



for entry in `ls /home/roussian.gaioso/experiments/input/postingList_2t_*`; do #postingList_2t_20000_
   echo $entry
   for topk in 128 #1024 
     do
	echo -------------Top-k:$topk--------------------------
  	for docsnumber in 32 64 128 #256 512 
	do
  		echo -------------DOCSNUMBER:$docsnumber--------------------------
		for threshold in 0 1 2
		do
		  echo -------------THRESHOLD:$threshold--------------------------
		
			for threads in 32 #256 512
			do	
				echo -------------threads:$threads--------------------------
				for partition_number in 1 10 100  #20 #50 100
				do
				   make clean
				   make K=$topk TERMS=2 DOCSBYMEM=$docsnumber  THRESHOLD=$threshold INVERTED_FILE=$entry DEBUG="-maxrregcount 128" #DEBUG="-DDEBUG -g -G"#
				   ./ParallelPrunningDAAT $topk $threads 1 $partition_number 1 1
				done 
			done
   		done	
	done	
   done
done

#for entry in `ls /home/roussian.gaioso/experiments/input/postingList_2t_*`; do #postingList_2t_20000_
#   echo $entry
#   for topk in 128 #1024 
#   do
#	echo -------------Top-k:$topk--------------------------
 #  	for docsnumber in 32 64 #256 512 
#	do
 # 		echo -------------DOCSNUMBER:$docsnumber--------------------------
#		for threshold in 0 1 2
#		do
#		  echo -------------THRESHOLD:$threshold--------------------------
		
#			for threads in 64 #256 512
#			do	
#				echo -------------threads:$threads--------------------------
#				for partition_number in  1 10 100 #20 #50 100
#				do
#				   make clean
#				   make K=$topk TERMS=2 DOCSBYMEM=$docsnumber  THRESHOLD=$threshold INVERTED_FILE=$entry DEBUG="-maxrregcount 128" #DEBUG="-DDEBUG -g -G"
#				   ./ParallelPrunningDAAT $topk $threads 1 $partition_number 2 1
#				done 
#			done
#   		done	
#	done	
#   done
#done

#for entry in `ls /home/roussian.gaioso/experiments/input/postingList_3*	`; do
#   echo $entry
#   for topk in 128 512 1024 
#   do
#	echo -------------Top-k:$topk--------------------------
#   	for threshold in 0 1 2
#	do
#   	        echo -------------THRESHOLD:$threshold--------------------------
#		for docsnumber in 64 128 256 512 
#		do
#		   echo -------------DOCSNUMBER:$docsnumber--------------------------
#		   make clean
#		   make K=$topk TERMS=3 DOCSBYMEM=$docsnumber DEBUG="-DDEBUG" THRESHOLD=$threshold INVERTED_FILE=$entry
#		   ./ParallelPrunningDAAT $topk 128 10 10 3
 #  		done	
#	done	
 #  done
#	break
#done

#for entry in `ls /home/roussian.gaioso/experiments/input/postingList_5*	`; do
#   echo $entry
#   for topk in 128 512 1024 
#   do
#	echo -------------Top-k:$topk--------------------------
#   	for threshold in 0 1 2
#	do
#   	        echo -------------THRESHOLD:$threshold--------------------------
#		for docsnumber in 64 128 256 512 
#		do
#		   echo -------------DOCSNUMBER:$docsnumber--------------------------
#		   make clean
#		   make K=$topk TERMS=5 DOCSBYMEM=$docsnumber DEBUG="-DDEBUG" THRESHOLD=$threshold INVERTED_FILE=$entry
#		   ./ParallelPrunningDAAT $topk 128 10 10 3
 #  		done	
#	done	
 #  done
#	break
#done
