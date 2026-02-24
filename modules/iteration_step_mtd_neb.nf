nextflow.enable.dsl=2

include { runMACE_MTD; runMACE_NEB; extractReact; collectNEBs; calcREF; updateDataset; reTrainMACE } from './processes.nf'

workflow ITERATION_STEP {
    take:
        iteration_in  // tuple: (dataset, models, run_label)

    main:
        dataset_ch = iteration_in.map { it[0] }
        models_ch  = iteration_in.map { it[1] }
        label_ch   = iteration_in.map { it[2] }

        // === Debug ===
        dataset_ch.view { "DEBUG: Dataset received: $it" }
        models_ch.view  { "DEBUG: Committee models received: $it" }
        label_ch.view   { "DEBUG: Label received: $it" }

        // === Run MACE MTD with committee (Discovery) ===
        mtd_out = runMACE_MTD(
            file('scripts/MTD_committee_plumed_MACE_system.py'),
            file('scripts/MACE_compare_descriptors.py'),
            file('input/TDMAS_SiO2_start.traj'),
            dataset_ch,
            models_ch,
            label_ch
        )

        // === Analyze and extract reactions from MTD trajectory ===
        extract_out = extractReact(
            file('scripts/extract_MTD_reactions.py'),
            mtd_out.mtd_trajectory,
            mtd_out.colvar_file,
            label_ch
        )

        // === Run MACE NEB with committee (PES refinement) ===
        neb_out = runMACE_NEB(
            file('scripts/NEB_committee_MACE.py'),
            extract_out.reaction_windows,
            dataset_ch,
            models_ch,
            label_ch
        )
        
         // === Collect and filter NEB windows (reactions) ===
        collect_out = collectNEBs(
            file('scripts/MACE_compare_descriptors.py'),
            neb_out.neb_uncertain_frames,
            dataset_ch,
            label_ch
        )

        // === DFT Refinement ===
        calcREF_out = calcREF(
            file('scripts/prepare_cp2k_farming_jobs.py'),
            file('scripts/parse_cp2k_farmed_to_extxyz.py'),
            file('input/template.inp'),
            collect_out.mace_frames,
            label_ch
        )

        // === Update dataset ===
        update_out = updateDataset(
            calcREF_out.new_data,
            dataset_ch,
            label_ch
        )
        updated_dataset_ch = update_out.updated_dataset
        //updated_dataset_ch = update_out.persistent_dataset
        
        // === Retraining step (3 jobs via each seed, randomSample() can be provided with a second seed parameter for the random seed generator) ===
        //seed_ch = Channel.of(123, 456, 789) (for fixed seeds) 
        seed_ch = Channel.of( 1..1000000 )
            .randomSample( 3 )
            .view()
        foundation_model = file('input/MACE_models/mace-mpa-0-medium.model')

        retrain_out = reTrainMACE(
            updated_dataset_ch,
            foundation_model,
            seed_ch,
            label_ch
        )

	retrained_models_ch = retrain_out.trained_models
    	    .flatten()
    	    .filter { !it.getFileName().toString().endsWith('_compiled.model') && !it.getFileName().toString().endsWith('_stagetwo.model') }
    	    //.toList()

        retrained_models_ch.view { "DEBUG: Retrained models emitted: $it" }

    emit:
        grown_dataset = updated_dataset_ch
        new_models    = retrained_models_ch
}

