nextflow.enable.dsl=2

include { ITERATION_STEP as iteration1 } from './modules/iteration_step_naive.nf'
include { ITERATION_STEP as iteration2 } from './modules/iteration_step_naive.nf'
include { ITERATION_STEP as iteration3 } from './modules/iteration_step_naive.nf'

workflow {

    // Initial dataset + models
    initial_dataset = file('growing_dataset/growing_retrain_dataset.xyz')
    
    initial_models = [
        file('input/MACE_models/MACE_model_03_finetuned.model'),
        file('input/MACE_models/MACE_model_04_finetuned.model'),
        file('input/MACE_models/MACE_model_05_finetuned.model')
    ]

    // === Iteration 1 ===
    iter1_input = Channel.of(tuple(initial_dataset, initial_models, "iter_1"))
    iter1_input.view { "DEBUG iter1_input: $it" }
    iter1_out   = iteration1(iter1_input)
    iter1_models_list_ch = iter1_out.new_models.collect()

    // === Iteration 2 ===
    iter2_input = iter1_out.grown_dataset
        .map { dataset ->
            tuple(dataset, iter1_models_list_ch.val, "iter_2")
        }
    iter2_input.view { "DEBUG iter2_input: $it" }
    iter2_out = iteration2(iter2_input)

    iter2_models_list_ch = iter2_out.new_models.collect()
    
    // === Iteration 3 ===
    iter3_input = iter2_out.grown_dataset
        .map { dataset ->
            tuple(dataset, iter2_models_list_ch.val, "iter_3")
        }
    iter3_input.view { "DEBUG iter3_input: $it" }
    iter3_out = iteration3(iter3_input)

    // === Monitor outputs with debug prints ===
    iter1_out.grown_dataset.view  { "Iter1 dataset: $it" }
    iter2_out.grown_dataset.view  { "Iter2 dataset: $it" }
    iter3_out.grown_dataset.view  { "Iter3 dataset: $it" }

    iter1_out.new_models.view     { "Iter1 models: $it" }
    iter2_out.new_models.view     { "Iter2 models: $it" }
    iter3_out.new_models.view     { "Iter3 models: $it" }
}

