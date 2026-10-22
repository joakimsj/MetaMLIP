#!/usr/bin/env python3
"""
Configurable committee-MACE / PLUMED ALD workflow scaffold.

Important:
- This is a starting implementation, not yet a validated ALD protocol.
- Explicit atom-index mappings are typically preferred over element-based guessing.
- PLUMED atom indices are 1-based; ASE indices are 0-based.
- Reaction/saturation criteria must be validated for each chemistry.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from ase import units
from ase.calculators.plumed import Plumed
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import NeighborList
from mace.calculators import MACECalculator


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def plumed_indices(indices):
    """Convert ASE zero-based indices to PLUMED one-based indices."""
    if not indices:
        raise ValueError("PLUMED atom group is empty")
    return [int(i) + 1 for i in indices]


def comma_indices(indices):
    return ",".join(str(i) for i in indices)


def resolve_group(group_spec, atoms, label):
    """
    Resolve an atom group from explicit ASE indices or element symbols.
    Explicit indices are strongly preferred for chemistry-specific groups.
    """
    if group_spec is None:
        raise ValueError(f"Missing atom-group definition: {label}")
    if isinstance(group_spec, list) and all(isinstance(x, int) for x in group_spec):
        indices = group_spec
    elif isinstance(group_spec, dict) and "indices" in group_spec:
        indices = [int(x) for x in group_spec["indices"]]
    elif isinstance(group_spec, dict) and "elements" in group_spec:
        elements = set(group_spec["elements"])
        indices = [i for i, atom in enumerate(atoms) if atom.symbol in elements]
        if "z_min" in group_spec:
            indices = [i for i in indices if atoms.positions[i, 2] >= float(group_spec["z_min"])]
    else:
        raise ValueError(
            f"{label} must be a list of ASE indices or a mapping with indices/elements"
        )
    if not indices:
        raise ValueError(f"Atom group '{label}' resolved to zero atoms")
    if min(indices) < 0 or max(indices) >= len(atoms):
        raise IndexError(f"Atom group '{label}' contains an index outside the current structure")
    return sorted(set(indices))


def insert_molecule(atoms, molecule_file, height, xy_range, rng):
    """Insert a molecule above the current highest atom and return its ASE index range."""
    molecule = read(molecule_file)
    if len(molecule) == 0:
        raise ValueError(f"Empty molecule structure: {molecule_file}")
    old_n = len(atoms)
    com = molecule.get_center_of_mass()
    top = float(np.max(atoms.positions[:, 2]))
    target = np.array([
        rng.uniform(-xy_range, xy_range),
        rng.uniform(-xy_range, xy_range),
        top + height,
    ])
    molecule.translate(target - com)
    atoms.extend(molecule)
    return list(range(old_n, len(atoms)))


def build_plumed_input(center_idx, surface_indices, ligand_indices, cfg, phase_tag):
    """Build a two-CV metadynamics input for one reactive center."""
    p_center = plumed_indices([center_idx])[0]
    p_surface = plumed_indices(surface_indices)
    p_ligands = plumed_indices(ligand_indices)

    cv = cfg["plumed"]
    temp = float(cfg["system"]["temperature_K"])
    lines = [
        f"UNITS LENGTH=A TIME={1/(1000*units.fs)} ENERGY={units.mol/units.kJ}",
        f"c_surface: COORDINATION GROUPA={p_center} GROUPB={comma_indices(p_surface)} R_0={cv['surface_r0_A']}",
        f"c_ligand: COORDINATION GROUPA={p_center} GROUPB={comma_indices(p_ligands)} R_0={cv['ligand_r0_A']}",
        (
            f"metad: METAD ARG=c_surface,c_ligand "
            f"PACE={cv['pace']} HEIGHT={cv['height']} "
            f"SIGMA={cv['sigma_surface']},{cv['sigma_ligand']} "
            f"BIASFACTOR={cv['biasfactor']} TEMP={temp} FILE=HILLS"
        ),
        f"PRINT ARG=c_surface,c_ligand,metad.bias STRIDE={cv['print_stride']} FILE=COLVAR",
        "FLUSH STRIDE=1",
    ]
    # Optional ligand CV wall. This is not a universal chemical criterion.
    if "ligand_lower_wall" in cv:
        wall = cv["ligand_lower_wall"]
        lines.insert(-3, f"wall_ligand: LOWER_WALLS ARG=c_ligand AT={wall['at']} KAPPA={wall['kappa']}")
    return lines


def latest_colvar(path="COLVAR"):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append(line.split())
    if not rows:
        return None
    vals = rows[-1]
    # PRINT columns: time c_surface c_ligand metad.bias
    if len(vals) < 3:
        return None
    return {"time": float(vals[0]), "c_surface": float(vals[1]), "c_ligand": float(vals[2])}


def get_committee_uncertainty(atoms, monitor_calc, cutoff=5.0):
    """
    Calculate committee force uncertainty and neighbor-aggregated uncertainty.

    Returns:
        force_sigma: (n_atoms,) norm of component-wise force standard deviations
        s_atom:      (n_atoms,) mean component-wise force standard deviation
        s_local:     (n_atoms,) neighbor-averaged s_atom
        energy_var:  structure-level energy variance, if provided by MACE
        max_force_uncertainty: max(force_sigma)
        mean_force_uncertainty: mean(force_sigma)
        local_max_force_uncertainty: max(s_local)
        local_mean_force_uncertainty: mean(s_local)

    The local metric preserves the definition used by the old workflow.
    NeighborList cutoffs are half the requested pair distance because ASE
    adds the cutoffs of the two atoms.
    """
    monitor_calc.calculate(atoms)
    result = monitor_calc.results

    forces_comm = result.get("forces_comm")
    if forces_comm is None:
        raise RuntimeError(
            "MACE did not provide 'forces_comm'; cannot calculate "
            "committee force uncertainty. Check your MACE version and "
            "calculator configuration."
        )

    fc = np.asarray(forces_comm, dtype=float)

    if fc.ndim != 3 or fc.shape[1:] != (len(atoms), 3):
        raise ValueError(
            "Unexpected forces_comm shape: "
            f"{fc.shape}; expected (n_models, {len(atoms)}, 3)."
        )

    if fc.shape[0] < 2:
        raise ValueError(
            "At least two committee force predictions are required "
            "to estimate ensemble disagreement."
        )

    # Standard deviation across models, independently for x, y, z.
    force_std = fc.std(axis=0)  # (n_atoms, 3)

    # Preserve both uncertainty definitions from the old script.
    s_atom = force_std.mean(axis=1)
    force_sigma = np.linalg.norm(force_std, axis=1)

    # Aggregate atomic uncertainty over each atom's local neighborhood.
    cutoffs = [float(cutoff) / 2.0] * len(atoms)
    nl = NeighborList(
        cutoffs,
        self_interaction=True,
        bothways=True,
    )
    nl.update(atoms)

    s_local = np.zeros(len(atoms), dtype=float)
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)

        # Include the central atom explicitly to avoid relying on
        # neighbor-list self-interaction behavior.
        indices = np.unique(np.append(indices, i))
        s_local[i] = s_atom[indices].mean()

    energy_var_raw = result.get("energy_var")
    energy_var = (
        None
        if energy_var_raw is None
        else float(np.asarray(energy_var_raw).mean())
    )

    return {
        "energy_var": energy_var,
        "force_sigma": force_sigma,
        "s_atom": s_atom,
        "s_local": s_local,
        "max_force_uncertainty": float(force_sigma.max()),
        "mean_force_uncertainty": float(force_sigma.mean()),
        "local_max_force_uncertainty": float(s_local.max()),
        "local_mean_force_uncertainty": float(s_local.mean()),
    }


def write_uncertainty_frame(atoms, uncertainty, out_path, phase, step, cv=None):
    """Write per-atom uncertainties and structure-level metadata to extxyz."""
    frame = atoms.copy()
    frame.info["ald_phase"] = str(phase)
    frame.info["step"] = int(step)

    for key in (
        "energy_var",
        "max_force_uncertainty",
        "mean_force_uncertainty",
        "local_max_force_uncertainty",
        "local_mean_force_uncertainty",
    ):
        value = uncertainty.get(key)
        if value is not None:
            frame.info[key] = float(value)

    # Per-atom arrays available for visualization and analysis.
    frame.arrays["per_atom_force_uncertainty"] = np.asarray(
        uncertainty["force_sigma"]
    ).copy()
    frame.arrays["atomic_force_uncertainty"] = np.asarray(
        uncertainty["s_atom"]
    ).copy()
    frame.arrays["local_force_uncertainty"] = np.asarray(
        uncertainty["s_local"]
    ).copy()

    if cv is not None:
        frame.info["c_surface"] = float(cv["c_surface"])
        frame.info["c_ligand"] = float(cv["c_ligand"])

    write(out_path, frame, format="extxyz", append=True)


def should_select_uncertain_frame(uncertainty, config):
    """Return whether a frame exceeds any enabled uncertainty threshold."""
    settings = config.get("uncertainty", {})
    select = False

    energy_limit = settings.get("energy_variance_limit", 0.0015)
    local_limit = settings.get("local_force_uncertainty_limit", 0.0200)

    if settings.get("select_by_energy_variance", True):
        energy_var = uncertainty.get("energy_var")
        if energy_var is not None and energy_var >= float(energy_limit):
            select = True

    if settings.get("select_by_local_force_uncertainty", True):
        local_max = uncertainty.get("local_max_force_uncertainty")
        if local_max is not None and local_max >= float(local_limit):
            select = True

    return select


def run_phase(atoms, phase_name, precursor_cfg, config, md_calc, monitor_calc, out_dir, rng):
    """
    Run one precursor exposure phase.
    This first version inserts one molecule and uses a configured CV threshold.
    For full multi-molecule saturation, extend this loop to insert additional molecules
    until a validated surface-wide coverage metric plateaus.
    """
    phase_dir = out_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(phase_dir)

    molecule_file = precursor_cfg["molecule_file"]
    inserted_indices = insert_molecule(
        atoms,
        molecule_file,
        float(config["sampling"]["insertion_height_A"]),
        float(config["sampling"]["insertion_xy_range_A"]),
        rng,
    )

    # For robust use, center_index and groups should be supplied explicitly in config.
    # center_index is an ASE index relative to the molecule template.
    center_local = int(precursor_cfg["center_index"])
    if center_local < 0 or center_local >= len(inserted_indices):
        raise ValueError(f"{phase_name}: center_index outside precursor molecule")
    center = inserted_indices[center_local]

    surface_group = resolve_group(precursor_cfg["surface_group"], atoms, f"{phase_name}.surface_group")
    ligand_local = precursor_cfg.get("ligand_indices")
    if ligand_local is None:
        raise ValueError(
            f"{phase_name}: provide ligand_indices as local indices in the precursor template. "
            "Element-based ligand guessing is intentionally not used."
        )
    ligand_indices = [inserted_indices[int(i)] for i in ligand_local]
    if not ligand_indices:
        raise ValueError(f"{phase_name}: ligand_indices is empty")

    for stale in ("HILLS", "COLVAR"):
        if os.path.exists(stale):
            os.remove(stale)

    plumed_input = build_plumed_input(
        center, surface_group, ligand_indices, config, phase_name
    )
    atoms.calc = Plumed(
        calc=md_calc,
        input=plumed_input,
        timestep=float(config["system"]["timestep_fs"]),
        atoms=atoms,
        kT=float(config["system"]["temperature_K"]) * units.kB,
    )
    dyn = VelocityVerlet(
        atoms, timestep=float(config["system"]["timestep_fs"]) * units.fs
    )

    threshold = precursor_cfg["completion"]
    check_interval = int(config["sampling"]["check_interval"])
    max_steps = int(config["sampling"]["max_steps_per_pulse"])
    uncertainty_log = out_dir / "uncertainty.jsonl"
    traj_path = out_dir / "ald_trajectory.extxyz"
    completed = False

    for step in range(check_interval, max_steps + 1, check_interval):
        dyn.run(check_interval)
        write(traj_path, atoms, format="extxyz", append=True, write_results=False)
        uncertainty_cfg = config.get("uncertainty", {})
        cutoff = float(uncertainty_cfg.get("neighbor_cutoff_A", 5.0))
        uncertainty = get_committee_uncertainty(
            atoms,
            monitor_calc,
            cutoff=cutoff,
        )

        cv = latest_colvar()
        write_uncertainty_frame(
            atoms,
            uncertainty,
            str(out_dir / "uncertain_frames.extxyz"),
            phase_name,
            step,
            cv=cv,
        )

        # Append scalar diagnostics and CVs to a machine-readable log.
        record = {
            key: value
            for key, value in uncertainty.items()
            if key not in ("force_sigma", "s_atom", "s_local")
        }
        record.update({
            "phase": phase_name,
            "step": step,
            "c_surface": None if cv is None else cv["c_surface"],
            "c_ligand": None if cv is None else cv["c_ligand"],
        })

        with open(uncertainty_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        # Save frames exceeding either enabled uncertainty threshold.
        if should_select_uncertain_frame(uncertainty, config):
            selected = atoms.copy()
            selected.info["ald_phase"] = str(phase_name)
            selected.info["step"] = int(step)

            for key in (
                "energy_var",
                "max_force_uncertainty",
                "mean_force_uncertainty",
                "local_max_force_uncertainty",
                "local_mean_force_uncertainty",
            ):
                value = uncertainty.get(key)
                if value is not None:
                    selected.info[key] = float(value)

            selected.arrays["per_atom_force_uncertainty"] = (
                uncertainty["force_sigma"].copy()
            )
            selected.arrays["atomic_force_uncertainty"] = (
                uncertainty["s_atom"].copy()
            )
            selected.arrays["local_force_uncertainty"] = (
                uncertainty["s_local"].copy()
            )

            if cv is not None:
                selected.info["c_surface"] = float(cv["c_surface"])
                selected.info["c_ligand"] = float(cv["c_ligand"])

            write(
                out_dir / "frames_for_DFT_eval.extxyz",
                selected,
                format="extxyz",
                append=True,
            )

        if cv is None:
            continue
        surface_ok = cv["c_surface"] >= float(threshold["surface_cv_min"])
        ligand_ok = cv["c_ligand"] <= float(threshold["ligand_cv_max"])
        if surface_ok and ligand_ok:
            completed = True
            break

    os.chdir(out_dir.parent)
    return completed, inserted_indices


def purge_unbound_precursor(atoms, inserted_indices, precursor_cfg):
    """
    Conservative placeholder: no automatic atom deletion is performed.
    Deleting molecules safely requires validated connectivity/bond-state detection.
    This function deliberately raises until that chemistry-specific logic is implemented.
    """
    raise NotImplementedError(
        "Implement chemistry-aware purge: distinguish intact gas-phase precursor from "
        "chemisorbed fragments before deleting atoms. Do not delete by distance alone."
    )


def main():
    parser = argparse.ArgumentParser(description="Configurable committee-MACE ALD workflow scaffold")
    parser.add_argument("--config", default="ald_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print planned phases without MD")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get("output_dir", "ald_output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.get("random_seed", None))

    phases = cfg["ald"]["sequence"]
    if args.dry_run:
        print("Configuration loaded. Planned sequence:")
        for i in range(int(cfg["sampling"]["max_cycles"])):
            for phase in phases:
                print(f"cycle {i + 1}: {phase}")
        print("Dry-run only: no MACE/PLUMED dynamics were run.")
        return

    atoms = read(cfg["system"]["slab_file"])
    if len(atoms) == 0:
        raise ValueError("Initial slab is empty")

    z_fix = float(cfg["system"].get("fix_below_z_A", np.percentile(atoms.positions[:, 2], 20)))
    fixed = [i for i, atom in enumerate(atoms) if atom.position[2] < z_fix]
    atoms.set_constraint(FixAtoms(indices=fixed))
    MaxwellBoltzmannDistribution(atoms, temperature_K=float(cfg["system"]["temperature_K"]))

    committee_cfg = cfg["committee"]
    md_calc = MACECalculator(
        model_paths=committee_cfg["model_paths"],
        device=committee_cfg.get("device", "cuda"),
        default_dtype=committee_cfg.get("dtype", "float64"),
        head=committee_cfg.get("head", "default"),
    )
    # Separate calculator avoids replacing the PLUMED-wrapped live MD calculator
    # while diagnostics are evaluated.
    monitor_calc = MACECalculator(
        model_paths=committee_cfg["model_paths"],
        device=committee_cfg.get("device", "cuda"),
        default_dtype=committee_cfg.get("dtype", "float64"),
        head=committee_cfg.get("head", "default"),
    )

    # Purge is deliberately blocked until a chemistry-aware deletion/flow protocol is supplied.
    for cycle in range(int(cfg["sampling"]["max_cycles"])):
        for phase_name in phases:
            if phase_name in ("purge_A", "purge_B"):
                raise NotImplementedError(
                    "Purge phases are in the state machine but require chemistry-specific "
                    "unbound-molecule detection before deleting atoms."
                )
            if phase_name not in cfg["chemistry"]:
                raise KeyError(f"No chemistry configuration found for phase '{phase_name}'")
            success, inserted = run_phase(
                atoms, phase_name, cfg["chemistry"][phase_name], cfg,
                md_calc, monitor_calc, out_dir, rng
            )
            write(out_dir / f"after_{phase_name}_cycle_{cycle + 1}.traj", atoms)
            if not success:
                print(f"WARNING: {phase_name} did not meet configured CV criteria within the step budget.")
                if cfg.get("stop_on_incomplete_phase", True):
                    write(out_dir / "final_structure.traj", atoms)
                    raise RuntimeError(f"Stopping because phase {phase_name} was incomplete.")
    write(out_dir / "final_structure.traj", atoms)
    print(f"Workflow completed. Output: {out_dir}")


if __name__ == "__main__":
    main()
