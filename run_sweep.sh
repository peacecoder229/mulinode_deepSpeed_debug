#!/bin/bash

echo "Start running tests"

for i in  "1024";
do
	for m in  "llama8b3p1" ;
	do
		echo "./intel_ipx_processScalewithdeepspeed.sh --model $m --tag EMR-lat-Perf --newtoken [56] --inputtoken $i --ipex"
		./intel_ipx_processScalewithdeepspeed.sh --model $m --tag EMR-lat-Perf --newtoken [56] --inputtoken $i --ipex
	done
done

