#PBS -l select=1:ncpus=1:mpiprocs=1:mem=10gb:ngpus=1
#PBS -lwalltime=0:10:0

# Load modules for any applications
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate pytorch

# Change to the directory the job was submitted from
cd $PBS_O_WORKDIR

python hpc_test.py > testing.log



