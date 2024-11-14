/root/miniforge3/envs/llm/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:270: UserWarning: Device capability of ccl unspecified, assuming `cpu` and `cuda`. Please specify it via the `devices` argument of `register_backend`.
  warnings.warn(
Warning: Cannot load xpu CCL. CCL doesn't work for XPU device due to /root/miniforge3/envs/llm/lib/python3.10/site-packages/oneccl_bindings_for_pytorch/lib/liboneccl_bindings_for_pytorch_xpu.so: cannot open shared object file: No such file or directory
[2024-11-13 07:00:26,588] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cpu (auto detect)
[2024-11-13 07:00:28,472] [INFO] [runner.py:463:main] Using IP address of 10.242.51.116 for node JF5300-B11A346T
Traceback (most recent call last):
  File "/root/miniforge3/envs/llm/bin/deepspeed", line 6, in <module>
    main()
  File "/root/miniforge3/envs/llm/lib/python3.10/site-packages/deepspeed/launcher/runner.py", line 524, in main
    runner = IMPIRunner(args, world_info_base64, resource_pool)
  File "/root/miniforge3/envs/llm/lib/python3.10/site-packages/deepspeed/launcher/multinode_runner.py", line 248, in __init__
    super().__init__(args, world_info_base64)
  File "/root/miniforge3/envs/llm/lib/python3.10/site-packages/deepspeed/launcher/multinode_runner.py", line 22, in __init__
    self.validate_args()
  File "/root/miniforge3/envs/llm/lib/python3.10/site-packages/deepspeed/launcher/multinode_runner.py", line 266, in validate_args
    raise ValueError(f"{self.name} backend does not support limiting num nodes/gpus")
ValueError: impi backend does not support limiting num nodes/gpus
