nextflow.enable.dsl=2

params.num_iterations = params.num_iterations ?: 3
params.initial_dataset = params.initial_dataset ?: 'growing_dataset/growing_retrain_dataset.xyz'
params.base_label = params.base_label ?: 'iteration'

include { iteration_step as iteration1 } from './modules/iteration_step.nf'
include { iteration_step as iteration2 } from './modules/iteration_step.nf'
include { iteration_step as iteration3 } from './modules/iteration_step.nf'

workflow {

    initial_models = [
        'input/MACE_models/MACE_model_03_finetuned.model',
        'input/MACE_models/MACE_model_04_finetuned.model',
        'input/MACE_models/MACE_model_05_finetuned.model'
    ]

    current_dataset = Channel.value(file(params.initial_dataset))

    // Iteration 1
    iter1_out = iteration1(tuple(file(params.initial_dataset), initial_models, "iteration_1"))

    // Iteration 2
    iter2_out = iteration2(tuple(iter1_out[0], iter1_out[1], "iteration_2"))

    // Iteration 3
    iter3_out = iteration3(tuple(iter2_out[0], iter2_out[1], "iteration_3"))

}

