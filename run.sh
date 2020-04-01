#!/bin/bash

function HELP {
  echo "Usage: ./run.sh -n NUM_GPUS -m MODE"
  exit 1
}

#parse options
np=1
while getopts :n:m:h FLAG; do
  case $FLAG in
    n)
        np=$OPTARG
        ;;
    m)
        mode=$OPTARG
        [[ ! $mode =~ CUDA|HIP|OpenCL|OpenMP|Serial ]] && {
            echo "Incorrect run mode provided"
            exit 1
        }
        ;;
    h)  #show help
        HELP
        ;;
    \?) #unrecognized option - show help
        HELP
        ;;
  esac
done

if [ -z $mode ]
then
    echo "No mode supplied, defaulting to HIP"
    mode=HIP
fi

# Build the code
make -j `nproc`

echo "Running hipBone..."

mpirun -np $np hipBone -m $mode -nx 126 -ny 126 -nz 126 -p 1
mpirun -np $np hipBone -m $mode -nx  63 -ny  63 -nz  63 -p 2
mpirun -np $np hipBone -m $mode -nx  42 -ny  42 -nz  42 -p 3
mpirun -np $np hipBone -m $mode -nx  32 -ny  32 -nz  32 -p 4
mpirun -np $np hipBone -m $mode -nx  26 -ny  26 -nz  26 -p 5
mpirun -np $np hipBone -m $mode -nx  21 -ny  21 -nz  21 -p 6
mpirun -np $np hipBone -m $mode -nx  18 -ny  18 -nz  18 -p 7
mpirun -np $np hipBone -m $mode -nx  16 -ny  16 -nz  16 -p 8
mpirun -np $np hipBone -m $mode -nx  14 -ny  14 -nz  14 -p 9
mpirun -np $np hipBone -m $mode -nx  13 -ny  13 -nz  13 -p 10
mpirun -np $np hipBone -m $mode -nx  12 -ny  12 -nz  12 -p 11
mpirun -np $np hipBone -m $mode -nx  11 -ny  11 -nz  11 -p 12
mpirun -np $np hipBone -m $mode -nx  10 -ny  10 -nz  10 -p 13
mpirun -np $np hipBone -m $mode -nx   9 -ny   9 -nz   9 -p 14
mpirun -np $np hipBone -m $mode -nx   9 -ny   9 -nz   9 -p 15

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
