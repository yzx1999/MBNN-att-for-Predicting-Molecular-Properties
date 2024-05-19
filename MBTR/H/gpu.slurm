#!/bin/bash
#SBATCH --job-name=wangzx  # 作业名
#SBATCH -N 1 -n 8
#SBATCH -t 144:00:00             # 作业运行时间
#SBATCH --output=%j.o            # 标准输出,其中%j是job的序号
#SBATCH --error=%j.e             # 错误输出
#SBATCH --partition=gpu          # 使用gpu队列
#SBATCH --gres=gpu:1             # 需要的gpu

ulimit -s unlimited
# load the environment              # 加载环境变量，根据作业需求调整 
module purge                        # 清空module内容
# module load compiler/cuda/11.4
# export PATH=/software/compiler/cuda/11.4/bin:$PATH
# export LD_LIBRARY_PATH=/software/compiler/cuda/11.4/lib64:$LD_LIBRARY_PATH    # cuda 11.4的环境变量
# export CUDA_HOME=/software/compiler/cuda/11.4

source /public/home/liuzhipan/wangzx/miniconda3/bin/activate # 激活conda
conda activate venv   # 激活环境
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/public/home/liuzhipan/gzx/1.PdPMe3_Conf/1.work_stage/1.from_zero/src/7.libs/
#echo LD_LIBRARY_PATH $LD_LIBRARY_PATH
export PYTHONPATH=/public/home/liuzhipan/gzx/1.PdPMe3_Conf/1.work_stage/1.from_zero:$PYTHONPATH # 针对采样脚本的，以让它可以访问到module
export PYTHONPATH=/public/home/liuzhipan/gzx/1.PdPMe3_Conf/1.work_stage/1.from_zero/src/8.laspy:$PYTHONPATH  # laspy


echo "============================================================"
env | grep "MKLROOT="
echo "============================================================"
echo "Job ID: $SLURM_JOB_NAME"
echo "Job name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of processors: $SLURM_NTASKS"
echo "Task is running on the following nodes:"
echo $SLURM_JOB_NODELIST
echo "OMP_NUM_THREADS = $SLURM_CPUS_PER_TASK"
echo "python path = `which python`"     # 获得python解释器路径
echo "availiable gpu = $SLURM_JOB_GPUS" # 可用的gpu编号, 测试了一下双卡，返回值为 :  availiable gpu = 4,5  注意，即便如此，也要通过cuda:0和cuda:1来访问gpu，因为NVIDIA-smi只能看到两张卡
echo "============================================================"
echo

python -u main.py > qm9.log
