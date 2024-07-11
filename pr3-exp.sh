#!/bin/bash
#
#SBATCH --job-name=knn
#SBATCH --partition=compute
#SBATCH --account=scu102
#SBATCH --cpus-per-task=56
#SBATCH --mem=10G
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=knn-%j.out
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email_address>
#

module purge
module load cpu
module load slurm
module load intel/19.1.3.304 
module load intel-mpi/2019.10.317
export OMP_BIND=cores;
export OMP_NUM_THREADS=28;

/usr/bin/time -v mpirun /<your_project_directory_path>/knn /<your_project_directory_path>/yelp.train.clu /<your_project_directory_path>/yelp.test.clu /<your_project_directory_path>/yelp.train.labels 5 /<your_project_directory_path>/knn.pred.5.txt 2>&1 > /<your_project_directory_path>/knn.5.log
