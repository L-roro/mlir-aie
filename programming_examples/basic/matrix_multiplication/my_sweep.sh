#!/usr/bin/bash

# run this script from one of the subdirectories to perform a sweep,
# e.g. from within whole_array, run ../my_sweep.sh.

runargs="--iters 20 --warmup 10"
iterations=1

# M_lo=256
# M_step=1024
# M_hi=4352
# # M_hi=256
# Ms=$(seq $M_lo $M_step $M_hi)
# K_lo=256
# K_step=1024
# K_hi=4352
# # K_hi=256
# Ks=$(seq $K_lo $K_step $K_hi)
# N_lo=256
# N_step=1024
# N_hi=4352
# # N_hi=256
# Ns=$(seq $N_lo $N_step $N_hi)

Ms=(512 1024 2560 4096)
Ks=(512 1024 2560 4096)
Ns=(512 1024 2560 4096)

perform_sweep() {
    echo "Performing function sweep"
    # Print configuration used to run for reproducibility
    env >>$log_out
    cat Makefile >>$log_out

    printf "M,K,N" >>$csv_out

    for i in $(seq 1 $iterations); do
        printf ",It"$i >>$csv_out
    done

    printf ",Status" >>$csv_out

    printf "\n" >>$csv_out

    for M in "${Ms[@]}"; do
        for K in "${Ks[@]}"; do
            for N in "${Ns[@]}"; do
                export M=$M
                export K=$K
                export N=$N
                echo ${M}x${K}x${N} 1>&2
                make clean 1>>$log_out 2>&1
                printf "${M},${K},${N}" >>$csv_out
                for i in $(seq 1 $iterations); do
                    make run >.tmp_run.log
                    cat .tmp_run.log $run_output >>$log_out
                    t=$(cat .tmp_run.log | sed -rn 's/^Avg NPU matmul time: ([0-9.]+)us.$/\1/p')
                    printf ",${t}" >>$csv_out
                    if cat .tmp_run.log | grep -q -F "PASS!"; then
                        printf ",PASS" >>$csv_out
                    else
                        printf ",FAIL" >>$csv_out
                    fi
                done
                printf "\n" >>$csv_out
            done
        done
    done
}

run_selected_hyperparameters() {
    EXPECTED_ARGS=11
    if [[ "$#" -ne "$EXPECTED_ARGS" ]]; then
        echo "Usage: $0 <m> <k> <n> <r> <s> <t> <dtype_in> <dtype_out> <n_aie_cols> <emulate_bfloat16_mmul_with_bfp16> <use_chess>"
        echo "Error: Incorrect number of arguments. Expected $EXPECTED_ARGS, got $#."
        exit 1
    fi

    if [ "${11}" == 1 ]; then
        compiler="chess"
    else
        compiler="peano"
    fi

    csv_out="results/${1}x${2}x${3}_${4}x${5}x${6}_${8}_out_${9}_col_${compiler}.csv"
    log_out="logs/${1}x${2}x${3}_${4}x${5}x${6}_${8}_out_${9}_col_${compiler}.log"

    export m=$1
    export k=$2
    export n=$3
    export dtype_in=$7
    export dtype_out=$8
    export n_aie_cols=$9
    export emulate_bfloat16_mmul_with_bfp16=${10}
    export use_chess=${11}

    # Set correct values for python and kernel files, note that the lines are harcoded!
    ./../sweep_helper.sh /scratch/luisr/mlir-aie/aie_kernels/aie2p/mm.cc 241 $4 $5 $6
    ./../sweep_helper.sh /scratch/luisr/mlir-aie/aie_kernels/aie2p/mm.cc 260 $4 $5 $6
    ./../sweep_helper.sh /scratch/luisr/mlir-aie/aie_kernels/aie2p/mm.cc 277 $4 $5 $6
    ./../sweep_helper.sh /scratch/luisr/mlir-aie/aie_kernels/aie2p/mm.cc 294 $4 $5 $6
    ./../sweep_helper.sh /scratch/luisr/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py 38 $4 $5 $6
    ./../sweep_helper.sh /scratch/luisr/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py 35 $4 $5 $6

    perform_sweep
}

mkdir results
mkdir logs

# This is constant for all runs
export runargs="${runargs}"

# run_selected_hyperparameters: m k n r s t dtype_in dtype_out n_aie_cols emulate_bfloat16_mmul_with_bfp16 use_chess

run_without_emulation() {
    run_selected_hyperparameters 32 32 32 4 8 8 bf16 $2 8 0 $1
    run_selected_hyperparameters 32 32 32 8 8 4 bf16 $2 8 0 $1
    run_selected_hyperparameters 32 32 32 4 8 4 bf16 $2 8 0 $1

    run_selected_hyperparameters 32 32 32 4 8 8 bf16 $2 4 0 $1
    run_selected_hyperparameters 32 32 32 8 8 4 bf16 $2 4 0 $1
    run_selected_hyperparameters 32 32 32 4 8 4 bf16 $2 4 0 $1

    run_selected_hyperparameters 64 64 64 4 8 8 bf16 $2 8 0 $1
    run_selected_hyperparameters 64 64 64 8 8 4 bf16 $2 8 0 $1
    run_selected_hyperparameters 64 64 64 4 8 4 bf16 $2 8 0 $1

    run_selected_hyperparameters 64 64 64 4 8 8 bf16 $2 4 0 $1
    run_selected_hyperparameters 64 64 64 8 8 4 bf16 $2 4 0 $1
    run_selected_hyperparameters 64 64 64 4 8 4 bf16 $2 4 0 $1
}

run_with_emulation() {
    run_selected_hyperparameters 64 64 64 8 8 8 bf16 bf16 4 1 $1
    run_selected_hyperparameters 64 64 64 8 8 8 bf16 bf16 8 1 $1

    run_selected_hyperparameters 32 32 32 8 8 8 bf16 bf16 4 1 $1
    run_selected_hyperparameters 32 32 32 8 8 8 bf16 bf16 8 1 $1

    run_selected_hyperparameters 32 32 32 8 8 8 bf16 f32 4 1 $1
    run_selected_hyperparameters 32 32 32 8 8 8 bf16 f32 8 1 $1

    # These tests will fail because of memory requirements
    # run_selected_hyperparameters 64 64 64 8 8 8 bf16 f32 4 1 $1
    # run_selected_hyperparameters 64 64 64 8 8 8 bf16 f32 8 1 $1
}

run_with_emulation 0
run_without_emulation 0 bf16

run_with_emulation 1
run_without_emulation 1 bf16

# This takes too much time already, so not running these
# run_without_emulation 0 f32
# run_without_emulation 1 f32