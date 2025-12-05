#!/bin/bash
#SBATCH --time=16:35:00            # Change your runtime settings
#SBATCH --partition=standard           # Change partition as needed
#SBATCH --account=project_462000290 #project_462000838        # Add your project name here
##SBATCH --cpus-per-task=1         # Change as needed
##SBATCH --mem-per-cpu=1G           # Increase as needed

# Load Nextflow module
module use /appl/local/csc/modulefiles
module load cp2k/2024.3
module load nextflow

# Actual Nextflow command here
nextflow run workflow_modules_five_iterations.nf -resume
# nf-core pipeline example:
# nextflow run nf-core/scrnaseq  -profile test,singularity -resume --outdir .
