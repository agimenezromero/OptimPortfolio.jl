#!/bin/bash

export LC_NUMERIC="en_US.UTF-8"

filename='lambdas.txt'

n=1 #To skip first line

while read line; do

	if (("$n" > 1))
	then
		#julia MVSK_optimization.jl "$line" "$n"
		run -t 5 -c 1 -m 8 -o run_$n.out -e run_$n.err julia MVSK_optimization.jl "$line" "$n"
		
	fi
	
	n=$((n+1))
	
	sleep 0.01

done < $filename
