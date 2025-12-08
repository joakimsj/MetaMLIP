#!/usr/bin/env python3
"""
generate_nextflow_workflow.py

Usage:
    python generate_nextflow_workflow.py <num_iterations> > workflow_generated.nf
"""

import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <num_iterations>")
    sys.exit(1)

num_iter = int(sys.argv[1])

# Header
print("nextflow.enable.dsl=2\n")
for i in range(1, num_iter + 1):
    print(f"include {{ ITERATION_STEP as iteration{i} }} from './modules/iteration_step.nf'")
print("\nworkflow {\n")

# Initial dataset and models
print("    initial_dataset = file('growing_dataset/growing_retrain_dataset.xyz')\n")
print("    initial_models = [")
print("        file('results/TDMAS_Si/reTrainMACE/iter_5/seed_123/MACE_model_seed_123.model'),")
print("        file('results/TDMAS_Si/reTrainMACE/iter_5/seed_456/MACE_model_seed_456.model'),")
print("        file('results/TDMAS_Si/reTrainMACE/iter_5/seed_789/MACE_model_seed_789.model')")
print("    ]\n")

# Iteration blocks
for i in range(1, num_iter + 1):
    if i == 1:
        print(f"    iter{i}_input = Channel.of(tuple(initial_dataset, initial_models, \"iter_{i}\"))")
        print(f"    iter{i}_input.view {{ \"DEBUG iter{i}_input: $it\" }}")
        print(f"    iter{i}_out = iteration{i}(iter{i}_input)")
        print(f"    iter{i}_models_list_ch = iter{i}_out.new_models.collect()\n")
    else:
        prev = i - 1
        print(f"    iter{i}_input = iter{prev}_out.grown_dataset")
        print(f"        .map {{ dataset ->")
        print(f"            tuple(dataset, iter{prev}_models_list_ch.val, \"iter_{i}\")")
        print(f"        }}")
        print(f"    iter{i}_input.view {{ \"DEBUG iter{i}_input: $it\" }}")
        print(f"    iter{i}_out = iteration{i}(iter{i}_input)")
        print(f"    iter{i}_models_list_ch = iter{i}_out.new_models.collect()\n")

# Dataset debug views
for i in range(1, num_iter + 1):
    print(f"    iter{i}_out.grown_dataset.view {{ \"Iter{i} dataset: $it\" }}")
print()
# Models debug views
for i in range(1, num_iter + 1):
    print(f"    iter{i}_out.new_models.view {{ \"Iter{i} models: $it\" }}")

print("\n}")
