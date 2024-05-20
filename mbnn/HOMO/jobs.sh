#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N test-auto
#$ -l h_rt=99999:00:00
#$ -pe make96 96
#$ -q node2018.q

#echo $JOB_ID > nodefile

source /home2/shang/.bashrc
#source /home7/intel/compilers_and_libraries_2018.2.199/linux/mpi/intel64/bin/mpivars.sh
cat `echo $PE_HOSTFILE` |cut -d " " -f -2 | awk '{print $1}' > hostfile
#PEXEC=/home8/hsd/testcode/train/2018-8-20_merge_stru_weight_scale_dist/traincnn-1.0/trainCNN-1.0
#PEXEC=/home8/hsd/include/strucdescrip/newtrain/newtrain/trainCNN-1.0
PEXEC=./lasp
ln -s $PEXEC ./$JOB_ID
#B=$( grep Energy TrainStr.txt -c )
#sed -i '/^NNtrain/d' lasp.in; sed -i '/^Ntrain/d' lasp.in; sed -i '1a \Ntrain '$B lasp.in
mpirun $mpienv -np $NSLOTS ./$JOB_ID >output

exit 0
