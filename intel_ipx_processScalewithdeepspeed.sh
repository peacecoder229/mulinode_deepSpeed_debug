#!/bin/bash

# Define arrays of values you want to test
#cpupower frequency-set --governor performance
sleep 1
intrathread_values=(48)
device_type_values=("cpu")
#device_type_values=("cuda")
#ins_values=({0..2} {0..4} {0..6})
#declare -A ins_values=(["64"]=2 ["32"]=4 ["16"]=8)
#declare -A ins_values=(["64"]=2 ["32"]=4 ["16"]=8) 
#declare -A ins_values=(["64"]=2 ["64"]=1 ["128"]=2)

ins_values=("128,2")
#ins_values=("60,1" "120,2" "240,4")
#ins_values=("240,4", "120,2", "60,1" )
#ins_values=("43,1" "86,2" "128,3" "256,6")
#ins_values=("256,6")
#bs_values=(64)
bs_values=(1) 
#bs_values=(1)
rm -f tmp_util
#declare -a pids
i=0  #track odd instnaces 
k=0  #track eeven istancs
#echo "intrathread,device_type,ins,bs,{pids[-1]},mininfT,maxinfT,avginfT,gpuutilavg,gpuutilmax,memutilmax" >> result_sum.txt
# Loop over all combinations of values
#for intrathread in ${intrathread_values[@]}; do
#README: for each device type  it will spanwn all instances in parallel as definedd by maxinstnaces and collect data.

EMON=false
IPEX=false
PROF=false
TAG=""
TOKENS=()

DNNVERBOSE=false
NOAMX=false
while [[ $# -gt 0 ]]; do
    arg="$1"
    case $arg in
        --dnnverbose)
        DNNVERBOSE=true
        shift # Remove --dnnverbose from processing
        ;;
        --noamx)
        NOAMX=true
        shift # Remove --noamx from processing
        ;;
        --prof)
        PROF=true
        shift # Remove --emon from processing
	;;
        --ipex)
        IPEX=true
        shift # Remove --emon from processing
	;;
        --emon)
        EMON=true
        shift # Remove --emon from processing
	;;
	--tag)
	shift # Move to next argument which should be the value for --tag
	TAG=$1
	shift # 
        ;;
	--model)
	shift # Move to next argument which should be the value for --tag
	model=$1
	shift # 
        ;;
	--inputtoken)
	shift # Move to next argument which should be the value for --tag
	intk=$1
	shift # 
        ;;
	--newtoken)
	echo "Entered --newtoken case"  # Debug prin
        shift
	echo "Argument after shift: $1"  # Debug print
	IFS=',' read -ra TOKENS <<< "${1//[\[\]]/}"
    	echo "Collected tokens after --newtokens: ${TOKENS[@]}"  # Debug print
	shift
        ;;
	--toklat)
	toklatcmd="--token-latency"
	shift
	;;
        *)
        # other arguments can be processed here if necessary
        shift
        ;;
    esac
done

if [[ -z "${toklatcmd}" ]]; then
	    toklatcmd=""
fi

# Use toklatcmd as needed in your script
echo "toklatcmd value: ${toklatcmd}"  # Debug print to verify the behavior
port=29500
#for token in "8" "16" "24" "32" "64"; do
echo "Collected tokens: ${TOKENS[@]}"

sleep 2


case $model in
    gptj6b)
        #modelid="EleutherAI/gpt-j-6B"
	modelid="/home/rdas/llama3p1/intel-extension-for-pytorch/examples/cpu/inference/python/llm/saved_results/gpt-j_local_shard"
	#samplestart="7500"
	samplestart="10"
        ;;
    llama7b)
        modelid="meta-llama/Llama-2-7b-hf"
        ;;
    llama8b)
        #modelid="meta-llama/Meta-Llama-3-8B"
	modelid="./saved_results/llama3_8b_shard"
        ;;
    llama8b3p1)
        #modelid="meta-llama/Meta-Llama-3.1-8B"
	modelid="./saved_results/llama_8b_shard/"
        ;;
    llama13b)
        modelid="meta-llama/Llama-2-13b-hf"
        ;;
    phi3mini)
	modelid="/home/rdas/llama3p1/intel-extension-for-pytorch/examples/cpu/inference/python/llm/saved_results/phi3-shard/"
	;;
    llama70b)
        modelid="meta-llama/Llama-2-70b-hf"
        ;;
    llama70b3p1)
        modelid="./saved_results/llama_70b_shard/"
        ;;
    codellama34b)
        modelid="codellama/CodeLlama-34b-Python-hf"
        ;;
    mixtral8)
        #modelid="mistralai/Mixtral-8x7B-v0.1"
	#modelid="/home/rdas/llama3p1/intel-extension-for-pytorch/examples/cpu/inference/python/llm/saved_results/mixtral_local_shard"
	modelid="/home/rdas/latest/intel-extension-for-pytorch/examples/cpu/inference/python/llm/utils/mixtral"
	#modelid="/root/intel-extension-for-pytorch/examples/cpu/llm/inference/utils/mixtral"
	#modelid="/root/intel-extension-for-pytorch/examples/cpu/inference/python/llm/utils/mixtral"
        ;;
    *)
        echo "Invalid modelid identifier"
        exit 1
        ;;
esac


env=""

unset ONEDNN_MAX_CPU_ISA


if $DNNVERBOSE; then
    env="ONEDNN_VERBOSE=1"
fi

if $NOAMX; then
	export ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16
    #if [[ -n $env ]]; then
    #    env="$env ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16"
    #else
    #    env="ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16"
    #fi
fi

echo "Environment variables set: $env"

#set -x

for token in "${TOKENS[@]}" ; do
  for device_type in ${device_type_values[@]}; do
    echo "$device_type"
    for bs in ${bs_values[@]}; do
      echo 3 > /proc/sys/vm/drop_caches && swapoff -a && swapon -a && printf '\n%s\n' 'Ram-cache and Swap Cleared'
      #python3 check_pt_gpu.py
      #for maxins in ${ins_values[@]}; do
      for pair in "${ins_values[@]}"; do
	IFS=',' read -r intrathread maxins <<< "$pair"
	#maxins=${ins_values[$intrathread]}
	alltoken=$(( token+intk ))
	k=0
	i=0
	ins=0
	newdir="log_dev${device_type}_ins${maxins}_BS${bs}"
	mkdir -p "Log/${newdir}"
	finalcmd=""
	core_list=""
	tmpfile="Log/res_${ins}.txt"
	rm -f $tmpfile
	if (( maxins == 1 )); then
		samplestart="100"
		# If maxins is 1, core_list is "0-(intrathread-1)"
		core_list="0-$((intrathread - 1))"
	else
		samplestart="100"
		delta=$((intrathread / maxins))
		#if [[ $maxins -eq 3 && $i -lt 2 ]]; then
		start=0

		for ((i=0; i<maxins; i++)); do
			if [[ $i -eq 0 || $i -eq 1 || $i -eq 3 || $i -eq 4 ]]; then
				current_delta=$delta
			else
				current_delta=$delta
			fi
			#start=$((i * current_delta))
			#end=$(((i + 1) * current_delta - 1))
			end=$(( start + current_delta -1 ))
			[[ $i -gt 0 ]] && core_list+=","
			core_list+="${start}-${end}"
			start=$((end + 1))
		done
	fi



		#!/bin/bash

		# Assuming PROF and IPEX can be either "true" or "false"

	if [ "$PROF" == "true" ] && [ "$IPEX" == "true" ]; then
		echo "Both PROF and IPEX are true."
		cmd="deepspeed  --num_gpus $maxins --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank --bind_core_list $core_list run_generation_with_deepspeed.py --device cpu  -m ${modelid} --ipex  --batch-size $bs --benchmark --max-new-tokens $token --input-tokens $intk --num-iter 3 --dtype bfloat16 --prof"

	elif [ "$IPEX" == "true" ]; then
		echo "Only IPEX is true."
		#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  ${modelid} --dtype bfloat16 --num-iter 3 --batch-size $bs --input-tokens $intk  --max_new_tokens $token --ipex --jit"
		cmd="deepspeed --num_gpus $maxins --master_addr `hostname -I | sed -e 's/\s.*$//'`  --bind_cores_to_rank  --bind_core_list $core_list  run.py --benchmark -m ${modelid} --dtype bfloat16 --ipex  --greedy --input-tokens $intk --num-iter 5 --num-warmup 2 --batch-size $bs --max-new-tokens $token --token-latency  --autotp"
		#cmd="deepspeed  --num_gpus $maxins --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank --bind_core_list $core_list run_generation_with_deepspeed.py --device cpu  -m ${modelid} --ipex  ${toklatcmd}  --batch-size $bs --benchmark --max-new-tokens $token --input-tokens $intk --num-iter 3 --dtype bfloat16"

	else
		echo "Default case."
		#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  ${modelid} --dtype bfloat16 --num-iter 3 --batch-size $bs --input-tokens $intk  --max_new_tokens $token"
		#cmd="numactl -C ${s}-${e} python gpt_inf.py --device $device_type  --batchsize $bs --insid $ins --threads $intrathread --maxtokens $token --modelid ${modelid}"
		#cmd="numactl -C ${s}-${e} python3 run_textgen.py --insid $ins  --ckpt_dir llama-2-7b --tokenizer_path tokenizer.modelid --max_seq_len $token --max_batch_size $bs --device $device_type --promptfile prompt.txt --intrathread $intrathread"
		#cmd="numactl -C ${s}-${e} python3 run_textgen.py --insid $ins  --ckpt_dir llama-2-${modelid} --tokenizer_path tokenizer.modelid --max_gen_len $token --max_batch_size $bs --device $device_type --promptfile prompt.txt --intrathread $intrathread"
		#cmd="numactl -C ${s}-${e} python3 run_textgen.py --insid $ins --modelid_type ${modelid} --ckpt_dir llama-2-${modelid} --tokenizer_path tokenizer.modelid --max_gen_len $token --max_batch_size $bs --device $device_type --promptfile prompt.txt --intrathread $intrathread"
		#cmd="numactl -C ${s}-${e} python3 run_textgen.py --insid $ins  --ckpt_dir llama-2-${modelid} --tokenizer_path tokenizer.modelid --max_gen_len $token --max_batch_size $bs --device $device_type --promptfile prompt.txt --intrathread $intrathread"
		#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  ${modelid} --dtype bfloat16 --num-iter 3 --batch-size $bs --input-tokens $intk  --max_new_tokens $token"
		#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --device $device_type --insid $ins --benchmark -m  ${modelid} --dtype bfloat16 --num-iter 3 --batch-size $bs --input-tokens $intk  --max_new_tokens $token"
		cmd="deepspeed  --num_gpus $maxins --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank --bind_core_list $core_list run_generation_with_deepspeed.py --device cpu  -m ${modelid}   --batch-size $bs --benchmark --max-new-tokens $token --input-tokens $intk --num-iter 3 --dtype bfloat16"
	fi




	# Run your command and get the PID of the process
	#modelid is Llama-2-7b or Llama-2-13b
	#cmd="numactl -C ${s}-${e} python3 run_textgen.py --insid $ins  --ckpt_dir llama-2-7b --tokenizer_path tokenizer.modelid --max_seq_len 64 --max_batch_size $bs --device $device_type --promptfile prompt.txt --intrathread $intrathread"
	#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  7b --dtype bfloat16 --num-iter 1 --batch-size $bs --input-tokens 612  --max_new_tokens $token --ipex --jit --accuracy-only"
	#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  ${modelid} --dtype bfloat16 --num-iter 1 --batch-size $bs --input-tokens 612  --max_new_tokens $token --ipex --jit"
	#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  7b --dtype bfloat16 --num-iter 1 --batch-size $bs --input-tokens 612  --max_new_tokens $token"

	# Save the PID of the last background job (the one we just launched)

# Echo values to the results file
#echo "$intrathread,$device_type,$ins,$bs,${pids[-1]}" >> result_sum.txt
	if [[ -n $env ]]; then
	    cmd="${env} ${cmd}"
	fi

	echo $cmd
      finalcmd="$cmd > \"$tmpfile\""
      echo $finalcmd
      mkdir -p GPU_UTIL
      rm -f /tmp/cpumemsize_tmp.txt
      if [ "$device_type" = "gpu" ] || [ "$device_type" = "cuda" ]; then
	      python3 -u  print_gpu_util_stats.py 1 > tmp_util &
      else
	      echo "skipping GPU Util"
      fi
      if $EMON; then
	      #tmc -T all -x rdas -i "NA" -SD $samplestart -e /opt/intel/sep/config/edp/filtered_events.txt -D "${model}_Prscale${maxins}_B${bs}" -n -u -c "$finalcmd"
	      tmc -T all -x rdas -i "NA" -SD $samplestart -e /opt/intel/sep/config/edp/filtered_memcpu.txt -D "4S_${model}_Przoom${maxins}_B${bs}" -n -u -c "$finalcmd"
      else

	      #( eval "$finalcmd" ) &


	      (
		  while : ; do
		    cpumemsize=$(ps aux | grep python | awk '{print $2}' | xargs -I{} pmap -x {} | awk '/total/ {print $4}' | awk '{sum += $1} END {print sum}')
		    echo $cpumemsize >> /tmp/cpumemsize_tmp.txt
		    sleep 2
		  done
	       ) &

		bg_pid=$!
	
	       eval "$finalcmd" 
	       echo "$finalcmd" 


      fi

      kill $bg_pid
      
      if [[ -f /tmp/cpumemsize_tmp.txt ]]; then

      	maverage=$(awk '{sum+=$1} END {if(NR > 0) print sum/NR; else print 0}' /tmp/cpumemsize_tmp.txt)
      	mmax_value=$(awk 'BEGIN {max = 0} {if ($1>max) max=$1} END {print max}' /tmp/cpumemsize_tmp.txt)
      else
	maverage="NA"
	mmax_value="NA"
      fi

      #cpumemsize=$(cat /tmp/cpumemsize_tmp.txt | tail -n 1)
      echo "CPU used memory $maverage $mmax_value KB"
      #cpumemsize=$( ps aux | grep python | awk '{print $2}' | xargs -I{} pmap -x {} | awk '/total/ {print $4}' | awk '{sum += $1} END {print sum}')
      #eval "$finalcmd"
      echo "##################################Run Complete ##########################################" 
      if [ "$device_type" = "gpu" ]  || [ "$device_type" = "cuda" ]; then
	      pkill -f print_gpu_util_stats.py
	      cp tmp_util GPU_UTIL/${model}_BS${bs}_T${intrathread}_TK${token}${TAG}_util.txt
	      gpuutilavg=$(./generate_min_max_avg_sum.py --inputfile=tmp_util | grep mean | awk '{print $3}')
	      gpuutilmax=$(./generate_min_max_avg_sum.py --inputfile=tmp_util | grep Max | awk '{print $3}')
	      memutilavg=$(./generate_min_max_avg_sum.py --inputfile=tmp_util | grep mean | awk '{print $2}')
	      memutilmax=$(./generate_min_max_avg_sum.py --inputfile=tmp_util | grep Max | awk '{print $2}')
      else
	      echo "No GPU"
      fi

      sleep 15

      res=""
      tmpfile="Log/res_${ins}.txt"
       #res+='$(grep "^inftime=" "$tmpfile")'
      res+="$(grep "^inftime=" "$tmpfile") "
      res+="$(grep "^gflops=" "$tmpfile") "
      res+="$(grep "^firstlat=" "$tmpfile") "
      res+="AMX=${NOAMX},"


      res+=", $TAG"

      touch cpulog_${TAG}.txt
      touch gpulog_${TAG}.txt
      touch gpuutil_${TAG}.txt

      if [ "$device_type" = "cpu" ]; then
	      inscnt=$(( ins+1 ))
	      echo "$maverage,$mmax_value,$model,$intrathread,$device_type,$maxins,$bs,$alltoken,$res" >> cpulog_${TAG}.txt
      else
	      inscnt=$(( ins+1 ))
	      echo "$maverage,$mmax_value,$model,$intrathread,$device_type,$inscnt,$bs,$res" >> gpulog_${TAG}.txt
	      echo "$model,$intrathread,$device_type,$inscnt,$bs,$gpuutilavg,$gpuutilmax,$memutilavg,$memutilmax,$res" >> gpuutil_${TAG}.txt
      fi
      #Using -u immediatey writes stdout to file

      # Wait for all jobs to finish before moving to the next bs value

      #pkill -f print_gpu_util_stats.py
     
     sleep 5
     #echo "$intrathread,$device_type,$ins,$bs,${pids[-1]},$mininfT,$maxinfT,$avginfT,$gpuutilavg,$gpuutilmax,$memutilavg,$memutilmax," >> result_sum.txt
      # Clear the PID array for the next loop iteration
    done  # for maxins

    done
  done
 done
#done

