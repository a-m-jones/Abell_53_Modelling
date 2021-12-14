#!/bin/sh 
#PBS -N bbody
#PBS -l nodes=1:ppn=1:fdr14 
#PBS -l walltime=00:30:00 
cd /home/ajones/2_PhD/Best_models/bbody/
/home/ajones/Cloudy2/c17.01/source/cloudy.exe bbody.in
