#!/bin/sh 
#PBS -N mcmc_18
#PBS -m abe
#PBS -l nodes=4:ppn=32:fdr14 
#PBS -l walltime=168:00:00 
cd /home/ajones/2_PhD/MCMC/model_18/
/usr/mpi/gcc/mvapich2-1.7/bin/mpirun /home/mjh/mcmc/mcmc.wrapper wrapper.in
