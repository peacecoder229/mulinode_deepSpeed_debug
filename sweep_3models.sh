echo "Start running tests"

#for m in "gptj6b" "llama8b" "mixtral8";
for m in "gptj6b" ;
#for m in "mixtral8";
do
    if [ "$m" = "gptj6b" ]; then
        input_tokens=("1024")
    elif [ "$m" = "llama8b" ]; then
        input_tokens=("2048")
    elif [ "$m" = "mixtral8" ]; then
        input_tokens=("4096")
    fi

    for i in "${input_tokens[@]}";
    do
        echo "./intel_ipx_processScalewithdeepspeed.sh --model $m --tag EMRvsH100 --newtoken [56] --inputtoken $i --ipex"
        ./intel_ipx_processScalewithdeepspeed.sh --model $m --tag EMRvsH100 --newtoken [56] --inputtoken $i --ipex
    done
done

