/root/nkbhat_debug/openmpi-5.0.5/build/install/bin/mpiexec -N 1 --allow-run-as-root -x PYTHONPATH /home/rdas/LLM/intel-extension-for-pytorch/examples/cpu/llm/inference -x OMP_NUM_THREADS 85 -x MASTER_ADDR 10.242.51.116 -x MASTER_PORT 29500 -x WORLD_SIZE 2 -x LOCAL_SIZE 2 -host JF5300-B11A346T:2 -n 1 -x RANK 0 -x LOCAL_RANK 0 numactl -C 0-84 /root/miniforge3/envs/llm/bin/python -u run.py --benchmark -m ./saved_results/llama_8b_shard --dtype bfloat16 --ipex --greedy --input-tokens 1024 --num-iter 2 --num-warmup 1 --batch-size 1 --max-new-tokens 32 --token-latency --autotp : -n 1 -host JF5300-B11A346T:2 -x RANK 1 -x LOCAL_RANK 1 numactl -C 85,171-254 /root/miniforge3/envs/llm/bin/python -u run.py --benchmark -m ./saved_results/llama_8b_shard --dtype bfloat16 --ipex --greedy --input-tokens 1024 --num-iter 2 --num-warmup 1 --batch-size 1 --max-new-tokens 32 --token-latency --autotp