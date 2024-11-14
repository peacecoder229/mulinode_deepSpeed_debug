#/bin/bash


function ccl_config()
{
	worker=$1

	if [ "$worker" = "4" ];then
		#export CCL_WORKER_AFFINITY='60,61,62,63'
		export CCL_WORKER_AFFINITY=auto
		export CCL_WORKER_COUNT=4
	fi
	export CCL_ALLREDUCE=rabenseifner # Other algorithms inlcude nreduce, ring and recursive_doubling. Rabenseifner algorithm is more friendly for latency sensitive workload

	export CCL_LOG_LEVEL=info
	export CCL_ATL_TRANSPORT=mpi #Other option is ofi
	export I_MPI_DEBUG=10

	#cp /usr/lib64/libfabric.so.1.18.1  ${CONDA_PREFIX}/lib/python3.9/site-packages/oneccl_bind_pt-2.1.0+cpu-py3.9-linux-x86_64.egg/oneccl_bindings_for_pytorch/lib/libfabric.so.1 
	#Do we need to copy above explicitly 
}


function unset_ccl_config() {
	    unset CCL_WORKER_AFFINITY
	        unset CCL_WORKER_COUNT
		    unset CCL_ALLREDUCE
		        unset CCL_LOG_LEVEL
			    unset CCL_ATL_TRANSPORT
}



function build_launch_arg(){
	margs="--genv CCL_WORKER_COUNT=${CCL_WORKER_COUNT}"
	margs="$margs --genv CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY}"
	margs="$margs --genv CCL_ATL_TRANSPORT=$CCL_ATL_TRANSPORT"   # Select the transport for inter-process communications
}




