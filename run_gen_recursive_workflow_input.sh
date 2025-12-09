#!/bin/bash
#SBATCH --time=00:05:00             # Change your runtime settings
#SBATCH --partition=standard        # Change partition as needed
#SBATCH --account=project_462000838 #project_462000838        # Add your project name here
##SBATCH --cpus-per-task=1          # Change as needed
##SBATCH --mem-per-cpu=1G           # Increase as needed

export PATH="/project/project_462000838/container_wrapper/mace_env_cueq/bin:\$PATH"

# Generate workflow script with maximum number of iterations as argument
python generate_nextflow_recursive_workflow.py 3 > workflow_recursive_generated.nf

# Load Nextflow modules
#module use /appl/local/csc/modulefiles
#module load cp2k/2024.3
#module load nextflow

# Run the generated workflow 
#nextflow run workflow_generated.nf #-resume
# nf-core pipeline example:
# nextflow run nf-core/scrnaseq  -profile test,singularity -resume --outdir .
