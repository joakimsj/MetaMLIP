# Committee-MACE ALD workflow scaffold

This is a scaffold for merging stochastic precursor insertion with committee-MACE uncertainty monitoring and PLUMED metadynamics.

## Molecular connectivity and reaction diagnostics

The surface-agnostic workflow imports `ald_connectivity.py` to construct a
periodic molecular graph using ASE neighbor lists, covalent-radius-based
formation/breaking thresholds, and hysteresis. Connectivity changes are only
reported after the configured number of consecutive checks (`connectivity.persist_checks`).
Formed/broken edges are appended to `connectivity.jsonl`; attempt summaries
include the final graph edges and precursor-connected fragment summaries.

The default `connectivity.enabled: true` also makes the pulse surface inventory
count graph-confirmed precursor-center-to-surface-site edges instead of relying
only on a distance cutoff. The inventory labels remain `reacted_proxy` because
the underlying geometric edge is not a validated chemical bond. To use molecular graph edges for an attempt's
completion criterion, configure that precursor's `saturation.criterion` as
`molecular_graph` and set `target_surface_bonds`. Optionally set
`also_require_ligand_dissociation: true` to require graph-breaking of all
configured center/ligand-anchor bonds. The prior `surface_bonds`,
`ligands_dissociated`, `both`, and `legacy_cv` criteria remain available.

Insertion height is always measured from the highest atom in graph components
connected to the dominant component of the initial substrate, even when
`connectivity.enabled` is false for reaction tracking. This avoids letting
detached gas-phase molecules or atoms near the upper cell boundary raise the
insertion reference. All existing atoms, including detached fragments, are
still considered during clash checks. The substrate anchor indices are
initialized from the largest connected component before dynamics; check the
initial connectivity graph if the input contains several comparable substrate
components.

These graph edges are geometric hypotheses, not bond orders. Validate
`form_scale`, `break_scale`, and any `pair_cutoffs_A` overrides against the
chemistry and trajectories. By default, graph tracking is restricted to the
current inserted precursor plus candidate surface atoms; this targets the
reaction interface and avoids treating unrelated slab bonds as reaction events.

## Status and limitations

This is **not yet a validated production ALD simulation**. It provides configuration loading, explicit molecular insertion, configurable CV generation, committee diagnostic hooks, trajectory logging, and a two-precursor state-machine outline.

Before production use:
1. Replace every example atom index with verified atom mappings.
2. Supply multiple independently trained MACE models if committee disagreement is intended.
3. Validate the MACE version's `energy_var` and `forces_comm` result keys.
4. The second precursor template is now `water.xyz` (O=0, H=1, H=2; zero-based local indices). Verify its geometry/orientation and the slab insertion region for your setup.
5. Implement chemistry-aware purge. The code deliberately raises `NotImplementedError` rather than risk deleting chemisorbed fragments.
6. Replace the simple single-center CV stopping rule with a validated surface-wide reaction and saturation test.
7. Decide whether PLUMED bias history restarts for each pulse or is continued; this scaffold uses per-phase directories and fresh HILLS/COLVAR files.
8. Add a thermostat if an isothermal ensemble is intended; the current MD integrator is VelocityVerlet.
9. Test on short trajectories and inspect bonds, CVs, energy/temperature, constraints, and uncertainty before long runs.

## Install

Use the same environment as your ASE, PLUMED and MACE installation, and add PyYAML if needed:

```bash
pip install pyyaml
```

## Validate configuration

```bash
python ald_workflow_surface_agnostic.py --config ald_config.yaml --dry-run
```

## Run

Only after the config is fully populated and chemistry-specific purge/completion logic has been implemented:

```bash
python ald_workflow_surface_agnostic.py --config ald_config.yaml
```

The water-half-cycle CV thresholds are placeholders: an O–H coordination CV alone does not determine water dissociation or hydroxylation. Define and validate a reaction-specific endpoint before running production trajectories.

The current script intentionally stops at purge states. This is a safety barrier, not an implementation omission to bypass: purge must distinguish gas-phase precursor from chemisorbed fragments based on the chemistry and current bonding state.

## Index conventions

- `center_index` and `ligand_indices` are local, zero-based indices in the precursor molecule file.
- `surface_group.indices` are zero-based ASE indices in the combined structure.
- The PLUMED builder converts ASE indices to PLUMED's one-based indexing.
- Static surface indices are not sufficient if the relevant surface sites change during deposition. Track/refresh site identities between half-cycles.
