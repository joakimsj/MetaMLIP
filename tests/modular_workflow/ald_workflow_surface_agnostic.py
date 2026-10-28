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

from ald_connectivity import (
    build_base_cutoffs,
    classify_fragments,
    detect_bonds_hysteresis,
    find_connected_components,
)
from ald_state import (
    ATOM_ID,
    ATOM_ROLE,
    IS_CENTER,
    IS_DEPOSITED_CENTER,
    MOLECULE_ID,
    ROLE_PRECURSOR_A,
    ROLE_WATER,
    classify_water_oh_endpoint,
    deposited_center_indices,
    ensure_atom_provenance,
    indices_for_atom_ids,
    purge_detached_inserted_components,
    register_inserted_molecule,
    water_occupancy_status,
)


def load_config(path):
    """Load YAML and resolve file paths relative to the config file directory."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    base_dir = config_path.parent

    def resolve_path(value):
        candidate = Path(value).expanduser()
        return str(candidate if candidate.is_absolute() else (base_dir / candidate).resolve())

    if isinstance(config.get("system", {}).get("slab_file"), str):
        config["system"]["slab_file"] = resolve_path(config["system"]["slab_file"])
    committee = config.get("committee", {})
    if isinstance(committee.get("model_paths"), list):
        committee["model_paths"] = [resolve_path(x) for x in committee["model_paths"]]
    for phase_cfg in config.get("chemistry", {}).values():
        if isinstance(phase_cfg, dict) and isinstance(phase_cfg.get("molecule_file"), str):
            phase_cfg["molecule_file"] = resolve_path(phase_cfg["molecule_file"])
    config["_config_path"] = str(config_path)
    return config


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


def build_connectivity_graph(atoms, connectivity_cfg=None):
    """Build a fresh hysteretic connectivity graph for the supplied structure."""
    cfg = connectivity_cfg or {}
    base_cutoffs = build_base_cutoffs(atoms, cfg.get("atomic_radii_A"))
    return detect_bonds_hysteresis(
        atoms,
        (),
        base_cutoffs,
        float(cfg.get("form_scale", 1.10)),
        float(cfg.get("break_scale", 1.70)),
        pair_cutoffs=cfg.get("pair_cutoffs_A"),
    )


def initialize_surface_anchor_indices(atoms, connectivity_cfg=None):
    """Store the largest connected component of the initial slab as substrate anchors."""
    ensure_atom_provenance(atoms)
    if len(atoms) == 0:
        raise ValueError("Cannot determine a surface anchor from an empty structure")
    bonds = build_connectivity_graph(atoms, connectivity_cfg)
    components = find_connected_components(len(atoms), bonds)
    anchor_component = max(
        components,
        key=lambda component: (
            len(component),
            float(np.ptp(atoms.positions[component, 2])) if len(component) > 1 else 0.0,
        ),
    )
    atoms.info["_ald_surface_anchor_indices"] = [int(i) for i in anchor_component]
    atoms.info["_ald_surface_anchor_atom_ids"] = [
        int(atoms.arrays[ATOM_ID][i]) for i in anchor_component
    ]
    return anchor_component


def highest_surface_bound_z(atoms, connectivity_cfg=None):
    """Return the highest z coordinate in any fragment graph-connected to substrate anchors."""
    cfg = connectivity_cfg or {}
    anchors = atoms.info.get("_ald_surface_anchor_indices")
    if not anchors:
        anchors = initialize_surface_anchor_indices(atoms, cfg)
    anchor_set = {int(i) for i in anchors if 0 <= int(i) < len(atoms)}
    if not anchor_set:
        raise ValueError("Stored surface anchor indices do not refer to atoms in the current structure")

    components = find_connected_components(len(atoms), build_connectivity_graph(atoms, cfg))
    bound_indices = [
        idx for component in components
        if anchor_set.intersection(component)
        for idx in component
    ]
    if not bound_indices:
        raise RuntimeError("No molecular graph fragment remains connected to the substrate anchors")
    return float(np.max(atoms.positions[bound_indices, 2])), bound_indices


def insert_molecule(atoms, molecule_file, height, xy_range, rng, temperature_K,
                    sampling_cfg=None, connectivity_cfg=None):
    """Place a precursor in a periodic XY cell and search upward in Z for a clash-free placement.

    Candidate lateral positions are sampled in fractional coordinates of the
    periodic surface cell. Insertion height is always referenced to the highest
    atom in components connected to the initial substrate graph; detached
    fragments do not raise this reference, regardless of whether connectivity
    tracking is enabled for reaction analysis. At each
    lateral position, the molecule is tested at increasing heights. A placement
    is accepted only when every molecule--system-atom pair is farther apart than
    ``clash_distance_A`` using ASE's
    minimum-image distance handling. This is a geometric overlap filter, not a
    guarantee of a chemically or dynamically suitable initial configuration.
    The accepted molecule receives Maxwell-Boltzmann momenta at the simulation
    temperature before it is appended, rather than inheriting zero momenta.
    """
    sampling_cfg = sampling_cfg or {}
    molecule_template = read(molecule_file)
    if len(molecule_template) == 0:
        raise ValueError(f"Empty molecule structure: {molecule_file}")

    n_xy_trials = int(sampling_cfg.get("insertion_xy_trials", 40))
    n_height_trials = int(sampling_cfg.get("insertion_height_trials", 12))
    height_step = float(sampling_cfg.get("insertion_height_step_A", 0.75))
    clash_distance = float(sampling_cfg.get("clash_distance_A", 1.8))
    if n_xy_trials < 1 or n_height_trials < 1:
        raise ValueError("insertion_xy_trials and insertion_height_trials must be >= 1")
    if height <= 0 or height_step <= 0 or clash_distance <= 0:
        raise ValueError("insertion height, height step, and clash distance must be positive")

    ensure_atom_provenance(atoms)
    old_n = len(atoms)
    # Keep placement anchored to the substrate even if reaction-graph tracking
    # is disabled: a detached gas-phase molecule must not move later insertions
    # toward the upper cell boundary.
    top, surface_bound_indices = highest_surface_bound_z(atoms, connectivity_cfg)
    overall_top = float(np.max(atoms.positions[:, 2]))
    if overall_top > top + 1e-6:
        print(
            f"Ignoring detached fragment(s) above substrate-connected component: "
            f"overall z_max={overall_top:.3f} A, surface-bound z_max={top:.3f} A"
        )
    print(
        f"Insertion reference: highest surface-bound atom z={top:.3f} A "
        f"({len(surface_bound_indices)} atoms in substrate-connected components)"
    )
    # Keep a copy of the template so every candidate starts from identical geometry.
    template = molecule_template.copy()
    template_com = template.get_center_of_mass()
    pbc_xy = bool(atoms.pbc[0]) and bool(atoms.pbc[1])
    cell = np.asarray(atoms.cell, dtype=float)
    cell2d = cell[:2, :2]
    periodic_xy_ok = pbc_xy and abs(float(np.linalg.det(cell2d))) >= 1e-10

    tested = 0
    best_min_distance = -np.inf
    for xy_trial in range(n_xy_trials):
        if periodic_xy_ok:
            # Fractional coordinates span the full repeating unit, including skew cells.
            frac = rng.random(2)
            xy_cart = frac @ cell2d
        else:
            xy_cart = np.array([
                rng.uniform(-xy_range, xy_range),
                rng.uniform(-xy_range, xy_range),
            ])

        for z_trial in range(n_height_trials):
            candidate = template.copy()
            target_com = np.array([
                float(xy_cart[0]), float(xy_cart[1]),
                top + float(height) + z_trial * height_step,
            ])
            candidate.translate(target_com - candidate.get_center_of_mass())
            tested += 1

            # Test against the original slab only. get_distance(..., mic=True)
            # handles periodic XY images when the cell's PBC flags are set.
            min_distance = np.inf
            for pos in candidate.positions:
                deltas = atoms.positions - pos
                if pbc_xy:
                    # Use ASE's MIC implementation for orthogonal or skewed cells.
                    from ase.geometry import find_mic
                    mic_deltas, _ = find_mic(deltas, cell, pbc=atoms.pbc)
                    pair_distances = np.linalg.norm(mic_deltas, axis=1)
                else:
                    pair_distances = np.linalg.norm(deltas, axis=1)
                if len(pair_distances):
                    min_distance = min(min_distance, float(np.min(pair_distances)))

            best_min_distance = max(best_min_distance, min_distance)
            if min_distance >= clash_distance:
                MaxwellBoltzmannDistribution(
                    candidate,
                    temperature_K=float(temperature_K),
                    rng=rng,
                )
                atoms.extend(candidate)
                inserted = list(range(old_n, len(atoms)))
                print(
                    f"Insertion accepted: XY trial {xy_trial + 1}/{n_xy_trials}, "
                    f"height trial {z_trial + 1}/{n_height_trials}, "
                    f"minimum slab distance={min_distance:.3f} A"
                )
                return inserted

    raise RuntimeError(
        f"Could not find a clash-free precursor placement after {tested} candidates. "
        f"Best minimum slab distance was {best_min_distance:.3f} A; "
        f"clash_distance_A={clash_distance:.3f}. Increase insertion_height_trials, "
        "insertion_xy_trials, or insertion_height_A, or review the molecule template."
    )



def configure_initial_cell(atoms, system_cfg, sampling_cfg=None, connectivity_cfg=None):
    """Apply optional XY supercell replication and explicit cell-height control.

    ``supercell_xy: [nx, ny]`` repeats the complete input structure before
    precursor insertion. ``cell_z_A`` sets the final cell vector length along
    Cartesian z without scaling atomic coordinates. The slab must have a
    z-aligned third cell vector for this simple cell-height operation.
    """
    repeats = system_cfg.get("supercell_xy", [1, 1])
    if not isinstance(repeats, (list, tuple)) or len(repeats) != 2:
        raise ValueError("system.supercell_xy must be [nx, ny]")
    nx, ny = (int(repeats[0]), int(repeats[1]))
    if nx < 1 or ny < 1:
        raise ValueError("system.supercell_xy values must both be >= 1")
    if nx != 1 or ny != 1:
        atoms = atoms.repeat((nx, ny, 1))
        print(f"Replicated slab in XY: {nx} x {ny}; atoms={len(atoms)}")

    surface_anchors = initialize_surface_anchor_indices(atoms, connectivity_cfg)
    surface_z_min = float(np.min(atoms.positions[surface_anchors, 2]))
    surface_z_max = float(np.max(atoms.positions[surface_anchors, 2]))
    overall_z_min = float(np.min(atoms.positions[:, 2]))
    overall_z_max = float(np.max(atoms.positions[:, 2]))

    sampling_cfg = sampling_cfg or {}
    requested_z = system_cfg.get("cell_z_A")
    if requested_z is not None:
        requested_z = float(requested_z)
        if requested_z <= 0:
            raise ValueError("system.cell_z_A must be positive")
        cell = np.asarray(atoms.cell, dtype=float).copy()
        # This scaffold assumes the surface normal is Cartesian z and the
        # third cell vector is parallel to z. Avoid silently damaging tilted cells.
        if abs(cell[2, 0]) > 1e-8 or abs(cell[2, 1]) > 1e-8:
            raise ValueError(
                "system.cell_z_A currently requires the third cell vector to be z-aligned; "
                "for a tilted cell, retain the input z vector or implement a cell-basis-aware resize."
            )
        if requested_z < overall_z_max - overall_z_min:
            raise ValueError(
                f"system.cell_z_A={requested_z:.3f} A is smaller than the slab's z extent "
                f"({overall_z_max - overall_z_min:.3f} A)"
            )
        # Basic guard against choosing a box too short for the initial COM
        # placement. Molecule extent and dynamics can require additional vacuum.
        insertion_height = float(sampling_cfg.get("insertion_height_A", 6.0))
        minimum_z_cell = overall_z_max + insertion_height + 2.0
        if requested_z <= minimum_z_cell:
            raise ValueError(
                f"system.cell_z_A={requested_z:.3f} A is too short for the current slab top "
                f"and insertion height; require > {minimum_z_cell:.3f} A as a basic minimum. "
                "Allow additional space for the precursor's molecular extent and dynamics."
            )
        cell[2] = [0.0, 0.0, requested_z]
        atoms.set_cell(cell, scale_atoms=False)
        print(
            f"Set cell z length to {requested_z:.3f} A without scaling atomic positions; "
            f"full structure z extent={overall_z_max - overall_z_min:.3f} A; "
            f"surface-connected substrate z extent={surface_z_max - surface_z_min:.3f} A"
        )
    return atoms

def detect_top_layer_sites(atoms, config):
    """Detect candidate surface atoms from the pre-pulse top z layer.

    Detection is geometry-only. Optional element filters can be supplied when
    not every topmost atom is a chemically eligible bonding site.
    """
    tolerance = float(config.get("top_layer_tolerance_A", 1.5))
    if tolerance < 0:
        raise ValueError("top_layer_tolerance_A must be non-negative")
    z = np.asarray(atoms.positions[:, 2], dtype=float)
    z_max = float(z.max())
    indices = [i for i, zi in enumerate(z) if zi >= z_max - tolerance]
    elements = config.get("elements")
    excluded = set(config.get("exclude_elements", []))
    if elements is not None:
        allowed = set(str(x) for x in elements)
        indices = [i for i in indices if atoms[i].symbol in allowed]
    if excluded:
        indices = [i for i in indices if atoms[i].symbol not in excluded]
    if not indices:
        raise ValueError(
            "Top-layer detector found no candidate surface atoms. Increase "
            "top_layer_tolerance_A or revise the element filters."
        )
    return indices, z_max


def count_center_surface_bonds(atoms, center_index, surface_indices, cutoff_A):
    """Count candidate surface atoms within a configured center-to-site cutoff."""
    cutoff = float(cutoff_A)
    if cutoff <= 0:
        raise ValueError("surface_bond_distance_A must be positive")
    bonded = []
    distances = {}
    for idx in surface_indices:
        distance = float(atoms.get_distance(int(center_index), int(idx), mic=True))
        distances[str(int(idx))] = distance
        if distance <= cutoff:
            bonded.append(int(idx))
    return bonded, distances


def advance_persistent_bond_graph(confirmed_bonds, current_bonds, pending, required_checks):
    """Accept graph edges only after their changed state persists across checks.

    ``current_bonds`` is the instantaneous graph from the hysteretic neighbor
    list. ``confirmed_bonds`` is the persistent molecular graph used by ALD
    completion and inventory logic. Pending edge flips are discarded if the
    instantaneous graph returns to its previously confirmed state.
    """
    confirmed = set(confirmed_bonds)
    current = set(current_bonds)
    required_checks = max(1, int(required_checks))

    for edge in list(pending):
        if (edge in current) == (edge in confirmed):
            del pending[edge]

    formed, broken = set(), set()
    for edge in confirmed.symmetric_difference(current):
        present = edge in current
        state = pending.get(edge)
        if state is None or state["present"] != present:
            state = {"present": present, "checks": 0}
            pending[edge] = state
        state["checks"] += 1
        if state["checks"] >= required_checks:
            if present:
                confirmed.add(edge)
                formed.add(edge)
            else:
                confirmed.discard(edge)
                broken.add(edge)
            del pending[edge]

    return confirmed, pending, formed, broken


def bond_records(atoms, bonds):
    """Serialize graph edges with ASE indices and element symbols."""
    return [
        {"indices": [int(i), int(j)], "elements": [atoms[i].symbol, atoms[j].symbol]}
        for i, j in sorted(bonds)
    ]

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


def _run_single_precursor_attempt(
    atoms, phase_name, precursor_cfg, config, md_calc, monitor_calc, out_dir, rng,
    surface_indices_override=None, attempt_index=1, pulse_id="pulse_001",
):
    """Run one exposure using a geometry-detected top-layer surface group.

    `saturation.criterion` can be `surface_bonds`, `ligands_dissociated`,
    `both`, or `legacy_cv`. The top-layer group is detected before insertion,
    so the denominator and atom group refer to the surface at pulse start.
    """
    phase_dir = out_dir / phase_name / str(pulse_id) / f"attempt_{int(attempt_index):03d}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(phase_dir)

    try:
        molecule_file = precursor_cfg["molecule_file"]
        detector_cfg = precursor_cfg.get("surface_site_detection", {"enabled": False})
        detected_surface_indices = None
        surface_z_max = None
        if surface_indices_override is not None:
            # Freeze the candidate set detected at pulse start across all attempts.
            surface_group = list(map(int, surface_indices_override))
        elif detector_cfg.get("enabled", False):
            detected_surface_indices, surface_z_max = detect_top_layer_sites(atoms, detector_cfg)
            surface_group = detected_surface_indices
            print(
                f"{phase_name}: detected {len(surface_group)} candidate top-layer atoms "
                f"(z_max={surface_z_max:.3f} A; tolerance="
                f"{float(detector_cfg.get('top_layer_tolerance_A', 1.5)):.3f} A)"
            )
        else:
            # Backward-compatible explicit group mode.
            surface_group = None

        inserted_indices = insert_molecule(
            atoms,
            molecule_file,
            float(config["sampling"]["insertion_height_A"]),
            float(config["sampling"]["insertion_xy_range_A"]),
            rng,
            temperature_K=float(config["system"]["temperature_K"]),
            sampling_cfg=config["sampling"],
            connectivity_cfg=config.get("connectivity", {}),
        )
        molecule_id = register_inserted_molecule(
            atoms, inserted_indices, phase_name, int(precursor_cfg["center_index"])
        )
        inserted_atom_ids = [int(atoms.arrays[ATOM_ID][i]) for i in inserted_indices]

        if surface_group is None:
            surface_group = resolve_group(
                precursor_cfg["surface_group"], atoms, f"{phase_name}.surface_group"
            )

        connectivity_cfg = config.get("connectivity", {})
        connectivity_enabled = bool(connectivity_cfg.get("enabled", True))
        connectivity_form_scale = float(connectivity_cfg.get("form_scale", 1.10))
        connectivity_break_scale = float(connectivity_cfg.get("break_scale", 1.70))
        connectivity_persist_checks = max(1, int(connectivity_cfg.get("persist_checks", 2)))
        connectivity_pair_cutoffs = connectivity_cfg.get("pair_cutoffs_A")
        connectivity_base_cutoffs = None
        instantaneous_bonds = set()
        confirmed_bonds = set()
        pending_bond_changes = {}
        connectivity_events = []
        graph_reactive_indices = sorted(set(map(int, inserted_indices + surface_group)))
        if connectivity_enabled:
            connectivity_base_cutoffs = build_base_cutoffs(
                atoms, connectivity_cfg.get("atomic_radii_A")
            )
            instantaneous_bonds = detect_bonds_hysteresis(
                atoms,
                (),
                connectivity_base_cutoffs,
                connectivity_form_scale,
                connectivity_break_scale,
                reactive_indices=graph_reactive_indices,
                pair_cutoffs=connectivity_pair_cutoffs,
            )
            confirmed_bonds = set(instantaneous_bonds)

        center_local = int(precursor_cfg["center_index"])
        if center_local < 0 or center_local >= len(inserted_indices):
            raise ValueError(f"{phase_name}: center_index outside precursor molecule")
        center = inserted_indices[center_local]

        # One index per ligand/ligand anchor is recommended. The CV is a smooth
        # aggregate coordination number, not an exact integer ligand counter.
        ligand_local = precursor_cfg.get("ligand_indices")
        if ligand_local is None:
            raise ValueError(
                f"{phase_name}: provide ligand_indices as local precursor-template indices; "
                "prefer one anchor atom per ligand for ligand-dissociation mode."
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

        legacy_threshold = precursor_cfg.get("completion", {})
        saturation = precursor_cfg.get("saturation", {})
        criterion = str(saturation.get("criterion", "legacy_cv")).lower()
        allowed_criteria = {
            "surface_bonds", "ligands_dissociated", "both", "legacy_cv",
            "molecular_graph", "water_oh",
        }
        if criterion not in allowed_criteria:
            raise ValueError(
                f"{phase_name}: saturation.criterion must be one of {sorted(allowed_criteria)}"
            )
        if criterion in ("surface_bonds", "both") and "target_surface_bonds" not in saturation:
            raise ValueError(
                f"{phase_name}: criterion '{criterion}' requires saturation.target_surface_bonds"
            )
        if criterion in ("surface_bonds", "both") and "surface_bond_distance_A" not in saturation:
            raise ValueError(
                f"{phase_name}: criterion '{criterion}' requires saturation.surface_bond_distance_A"
            )
        if criterion == "molecular_graph":
            if not connectivity_enabled:
                raise ValueError(
                    f"{phase_name}: saturation criterion 'molecular_graph' requires connectivity.enabled"
                )
            if "target_surface_bonds" not in saturation:
                raise ValueError(
                    f"{phase_name}: criterion 'molecular_graph' requires saturation.target_surface_bonds"
                )
            if int(saturation["target_surface_bonds"]) < 1:
                raise ValueError(
                    f"{phase_name}: saturation.target_surface_bonds must be >= 1"
                )
        if criterion == "water_oh":
            if not connectivity_enabled:
                raise ValueError(f"{phase_name}: water_oh requires connectivity.enabled")
            if phase_name != "precursor_B":
                raise ValueError("water_oh criterion is reserved for precursor_B water attempts")
        if criterion in ("ligands_dissociated", "both") and "ligand_dissociation_distance_A" not in saturation:
            raise ValueError(
                f"{phase_name}: criterion '{criterion}' requires saturation.ligand_dissociation_distance_A; "
                "ligand_indices should contain one anchor atom per ligand."
            )
        if criterion == "legacy_cv" and not legacy_threshold:
            raise ValueError(f"{phase_name}: legacy_cv criterion requires completion thresholds")

        check_interval = int(config["sampling"]["check_interval"])
        max_steps = int(config["sampling"]["max_steps_per_pulse"])
        uncertainty_log = out_dir / "uncertainty.jsonl"
        traj_path = out_dir / "ald_trajectory.extxyz"
        completed = False
        last_cv = None
        completed_step = None
        ligand_dissociation_streak = 0
        ligand_dissociation_required_checks = max(
            1, int(saturation.get("ligand_dissociation_consecutive_checks", 2))
        )
        surface_bond_streak = 0
        surface_bond_required_checks = max(
            1, int(saturation.get("surface_bond_consecutive_checks", 2))
        )
        last_ligand_anchor_distances = []
        last_bonded_surface_indices = []
        graph_bonded_surface_indices = []
        water_endpoint = None
        connectivity_log = out_dir / "connectivity.jsonl"

        for step in range(check_interval, max_steps + 1, check_interval):
            dyn.run(check_interval)
            write(traj_path, atoms, format="extxyz", append=True, write_results=False)
            graph_formed, graph_broken = set(), set()
            if connectivity_enabled:
                current_bonds = detect_bonds_hysteresis(
                    atoms,
                    instantaneous_bonds,
                    connectivity_base_cutoffs,
                    connectivity_form_scale,
                    connectivity_break_scale,
                    reactive_indices=graph_reactive_indices,
                    pair_cutoffs=connectivity_pair_cutoffs,
                )
                confirmed_bonds, pending_bond_changes, graph_formed, graph_broken = (
                    advance_persistent_bond_graph(
                        confirmed_bonds,
                        current_bonds,
                        pending_bond_changes,
                        connectivity_persist_checks,
                    )
                )
                instantaneous_bonds = current_bonds
                if graph_formed or graph_broken:
                    event = {
                        "phase": phase_name,
                        "step": int(step),
                        "formed_bonds": bond_records(atoms, graph_formed),
                        "broken_bonds": bond_records(atoms, graph_broken),
                    }
                    connectivity_events.append(event)
                    with open(connectivity_log, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event) + "\n")
            uncertainty_cfg = config.get("uncertainty", {})
            cutoff = float(uncertainty_cfg.get("neighbor_cutoff_A", 5.0))
            uncertainty = get_committee_uncertainty(atoms, monitor_calc, cutoff=cutoff)

            cv = latest_colvar()
            last_cv = cv
            write_uncertainty_frame(
                atoms, uncertainty, str(out_dir / "uncertain_frames.extxyz"),
                phase_name, step, cv=cv,
            )

            record = {
                key: value for key, value in uncertainty.items()
                if key not in ("force_sigma", "s_atom", "s_local")
            }
            record.update({
                "phase": phase_name,
                "step": step,
                "c_surface": None if cv is None else cv["c_surface"],
                "c_ligand": None if cv is None else cv["c_ligand"],
                "surface_candidate_count_at_pulse_start": len(surface_group),
                "saturation_criterion": criterion,
            })
            with open(uncertainty_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            if should_select_uncertain_frame(uncertainty, config):
                selected = atoms.copy()
                selected.info["ald_phase"] = str(phase_name)
                selected.info["step"] = int(step)
                for key in (
                    "energy_var", "max_force_uncertainty", "mean_force_uncertainty",
                    "local_max_force_uncertainty", "local_mean_force_uncertainty",
                ):
                    value = uncertainty.get(key)
                    if value is not None:
                        selected.info[key] = float(value)
                selected.arrays["per_atom_force_uncertainty"] = uncertainty["force_sigma"].copy()
                selected.arrays["atomic_force_uncertainty"] = uncertainty["s_atom"].copy()
                selected.arrays["local_force_uncertainty"] = uncertainty["s_local"].copy()
                if cv is not None:
                    selected.info["c_surface"] = float(cv["c_surface"])
                    selected.info["c_ligand"] = float(cv["c_ligand"])
                write(out_dir / "frames_for_DFT_eval.extxyz", selected,
                      format="extxyz", append=True)

            # Count direct center-to-surface contacts from a configured geometric
            # cutoff. This is more interpretable than treating a smooth coordination
            # CV as an integer bond count, but the cutoff still needs validation.
            if criterion in ("surface_bonds", "both"):
                last_bonded_surface_indices, _ = count_center_surface_bonds(
                    atoms, center, surface_group, float(saturation["surface_bond_distance_A"])
                )
                if len(last_bonded_surface_indices) >= int(saturation["target_surface_bonds"]):
                    surface_bond_streak += 1
                else:
                    surface_bond_streak = 0
            surface_bonds_reached = (
                surface_bond_streak >= surface_bond_required_checks
            )
            if connectivity_enabled:
                graph_bonded_surface_indices = sorted(
                    int(site_idx)
                    for site_idx in surface_group
                    if tuple(sorted((center, int(site_idx)))) in confirmed_bonds
                )

            # For the chemistry-aware ligand criterion, inspect each configured
            # ligand anchor independently instead of relying only on an aggregate CV.
            ligand_dissociated_all = False
            if criterion in ("ligands_dissociated", "both"):
                ligand_cutoff = float(saturation["ligand_dissociation_distance_A"])
                last_ligand_anchor_distances = [
                    float(atoms.get_distance(center, ligand_idx, mic=True))
                    for ligand_idx in ligand_indices
                ]
                if all(distance > ligand_cutoff for distance in last_ligand_anchor_distances):
                    ligand_dissociation_streak += 1
                else:
                    ligand_dissociation_streak = 0
                ligand_dissociated_all = (
                    ligand_dissociation_streak >= ligand_dissociation_required_checks
                )

            water_endpoint = None
            if criterion == "water_oh":
                oxygen_local = int(precursor_cfg["center_index"])
                hydrogen_local = [int(i) for i in precursor_cfg["ligand_indices"]]
                water_endpoint = classify_water_oh_endpoint(
                    confirmed_bonds,
                    inserted_indices[oxygen_local],
                    [inserted_indices[i] for i in hydrogen_local],
                    surface_group,
                )
            if cv is None and criterion not in ("molecular_graph", "water_oh"):
                continue
            ligand_max = saturation.get("ligand_cv_max", legacy_threshold.get("ligand_cv_max", 0.25))
            ligand_ok = cv is not None and cv["c_ligand"] <= float(ligand_max)
            if criterion == "surface_bonds":
                done = surface_bonds_reached
            elif criterion == "molecular_graph":
                graph_target = int(saturation["target_surface_bonds"])
                graph_surface_ok = len(graph_bonded_surface_indices) >= graph_target
                if saturation.get("also_require_ligand_dissociation", False):
                    graph_ligand_ok = all(
                        tuple(sorted((center, int(ligand_idx)))) not in confirmed_bonds
                        for ligand_idx in ligand_indices
                    )
                else:
                    graph_ligand_ok = True
                done = graph_surface_ok and graph_ligand_ok
            elif criterion == "water_oh":
                done = bool(water_endpoint["qualifies"])
            elif criterion == "ligands_dissociated":
                done = ligand_dissociated_all
                if saturation.get("also_require_surface_coordination", False):
                    required = float(saturation.get("surface_cv_min", 0.5))
                    done = done and cv["c_surface"] >= required
            elif criterion == "both":
                done = surface_bonds_reached and ligand_dissociated_all
            else:
                done = (
                    cv is not None
                    and
                    cv["c_surface"] >= float(legacy_threshold["surface_cv_min"])
                    and cv["c_ligand"] <= float(legacy_threshold["ligand_cv_max"])
                )

            if done:
                completed = True
                completed_step = step
                break

        # Optional geometric audit of center-to-surface contacts. This is a
        # diagnostic metric and requires a chemistry-appropriate distance cutoff.
        bonded_surface_indices = []
        distance_map = {}
        bond_cutoff = saturation.get("surface_bond_distance_A")
        if bond_cutoff is not None:
            bonded_surface_indices, distance_map = count_center_surface_bonds(
                atoms, center, surface_group, float(bond_cutoff)
            )
        summary = {
            "phase": phase_name,
            "completed": bool(completed),
            "completion_step": completed_step,
            "saturation_criterion": criterion,
            "inserted_molecule_id": int(molecule_id),
            "inserted_atom_ids": inserted_atom_ids,
            "water_oh_endpoint_at_end": water_endpoint if criterion == "water_oh" else None,
            "precursor_ligand_anchor_count": len(ligand_indices),
            "available_surface_atoms_at_pulse_start": len(surface_group),
            "detected_surface_atom_indices": list(map(int, surface_group)),
            "bonded_surface_atoms_at_end_by_distance": len(bonded_surface_indices),
            "surface_bond_distance_cutoff_A": None if bond_cutoff is None else float(bond_cutoff),
            "surface_bond_consecutive_checks_required": surface_bond_required_checks,
            "bonded_surface_fraction_of_initial_candidates": (
                None if bond_cutoff is None or not surface_group
                else len(bonded_surface_indices) / len(surface_group)
            ),
            "bonded_surface_atom_indices_at_end": bonded_surface_indices,
            "graph_connectivity_enabled": connectivity_enabled,
            "graph_form_scale": connectivity_form_scale,
            "graph_break_scale": connectivity_break_scale,
            "graph_persistent_checks_required": connectivity_persist_checks,
            "confirmed_graph_bonds_at_end": (
                bond_records(atoms, confirmed_bonds) if connectivity_enabled else None
            ),
            "graph_bonded_surface_atom_indices_at_end": graph_bonded_surface_indices,
            "graph_reaction_events": connectivity_events,
            "precursor_connected_fragments_at_end": (
                [
                    fragment for fragment in classify_fragments(
                        atoms, confirmed_bonds, surface_group
                    )
                    if set(fragment["atom_indices"]).intersection(inserted_indices)
                ]
                if connectivity_enabled else None
            ),
            "center_to_surface_distances_A": distance_map,
            "final_c_surface": None if last_cv is None else last_cv["c_surface"],
            "final_c_ligand": None if last_cv is None else last_cv["c_ligand"],
            "ligand_dissociation_distance_cutoff_A": saturation.get("ligand_dissociation_distance_A"),
            "final_ligand_anchor_distances_A": last_ligand_anchor_distances,
            "ligand_dissociation_consecutive_checks_required": ligand_dissociation_required_checks,
            "note": "Top-layer membership, distance-cutoff contacts, and ligand-anchor dissociation are geometry-based proxies; validate them against chemistry-specific bond analysis.",
        }
        (phase_dir / "pulse_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{phase_name}: completed={completed}; initial surface candidates={len(surface_group)}; "
            f"bonded candidates by cutoff={len(bonded_surface_indices) if bond_cutoff is not None else 'not calculated'}"
        )
        return completed, inserted_indices, confirmed_bonds if connectivity_enabled else None
    finally:
        os.chdir(previous_cwd)


def run_phase(atoms, phase_name, precursor_cfg, config, md_calc, monitor_calc, out_dir, rng,
              pulse_id="pulse_001"):
    """Run a pulse with a frozen pulse-start surface inventory.

    The denominator is the number of candidate atoms detected before the first
    insertion. Each successful precursor attempt can mark candidate atoms as
    ``reacted_proxy`` when a persistent graph edge connects that site to the
    precursor center. Atoms are counted only once across the pulse.

    Connectivity-derived inventory remains a geometric proxy, not a universal
    chemical bond classifier.
    Failed attempts stop the pulse because chemistry-aware removal of an unbound
    precursor is not implemented in this scaffold.
    """
    ensure_atom_provenance(atoms)
    if phase_name == "precursor_B":
        return run_water_phase(
            atoms, phase_name, precursor_cfg, config, md_calc, monitor_calc,
            out_dir, rng, pulse_id=pulse_id,
        )

    detector_cfg = precursor_cfg.get("surface_site_detection", {"enabled": False})
    inventory_cfg = precursor_cfg.get("surface_inventory", {})
    inventory_enabled = bool(inventory_cfg.get("enabled", False))

    # Preserve the old one-insertion behavior when no top-layer inventory is enabled.
    if not inventory_enabled:
        completed, inserted, bonds = _run_single_precursor_attempt(
            atoms, phase_name, precursor_cfg, config, md_calc, monitor_calc,
            out_dir, rng, pulse_id=pulse_id,
        )
        if completed and phase_name == "precursor_A":
            center = inserted[int(precursor_cfg["center_index"])]
            atoms.arrays[IS_DEPOSITED_CENTER][center] = 1
        return completed, inserted, bonds

    if not detector_cfg.get("enabled", False):
        raise ValueError(
            f"{phase_name}: surface_inventory requires surface_site_detection.enabled: true"
        )

    pulse_dir = out_dir / phase_name / str(pulse_id)
    pulse_dir.mkdir(parents=True, exist_ok=True)
    # Detect exactly once: later adsorbates must not change the denominator or
    # silently become new entries in this pulse's initial surface inventory.
    initial_surface_indices, surface_z_max = detect_top_layer_sites(atoms, detector_cfg)
    if not initial_surface_indices:
        raise ValueError(f"{phase_name}: no surface candidates detected at pulse start")

    saturation = precursor_cfg.get("saturation", {})
    inventory_cutoff = inventory_cfg.get(
        "bond_distance_A", saturation.get("surface_bond_distance_A")
    )
    if inventory_cutoff is None:
        raise ValueError(
            f"{phase_name}: surface_inventory needs bond_distance_A, or saturation.surface_bond_distance_A, "
            "to classify unique surface contacts"
        )
    inventory_cutoff = float(inventory_cutoff)
    if inventory_cutoff <= 0:
        raise ValueError("surface_inventory bond distance must be positive")

    max_attempts = max(1, int(inventory_cfg.get(
        "max_adsorption_attempts", config.get("sampling", {}).get("max_adsorption_attempts", 1)
    )))
    target_fraction = float(inventory_cfg.get("target_fraction", 1.0))
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("surface_inventory.target_fraction must be in (0, 1]")

    inventory = {
        "phase": phase_name,
        "pulse_id": str(pulse_id),
        "surface_z_max_at_pulse_start_A": float(surface_z_max),
        "top_layer_tolerance_A": float(detector_cfg.get("top_layer_tolerance_A", 1.5)),
        "candidate_count_at_pulse_start": len(initial_surface_indices),
        "target_fraction": target_fraction,
        "contact_distance_cutoff_A": inventory_cutoff,
        "max_adsorption_attempts": max_attempts,
        "sites": [
            {
                "atom_index": int(idx),
                "element": atoms[int(idx)].symbol,
                "initial_position_A": [float(x) for x in atoms.positions[int(idx)]],
                "status": "available",
                "first_reacted_attempt": None,
                "contacting_precursor_center_index": None,
                "contacting_precursor_center_atom_id": None,
                "contact_distance_A": None,
            }
            for idx in initial_surface_indices
        ],
        "attempts": [],
    }
    inventory_path = pulse_dir / "surface_inventory.json"

    def save_inventory():
        inventory["unique_reacted_site_count"] = sum(
            site["status"] == "reacted_proxy" for site in inventory["sites"]
        )
        inventory["surface_coverage_fraction"] = (
            inventory["unique_reacted_site_count"] / inventory["candidate_count_at_pulse_start"]
            if inventory["candidate_count_at_pulse_start"] else 0.0
        )
        inventory["target_reached"] = (
            inventory["surface_coverage_fraction"] >= target_fraction
        )
        inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    save_inventory()
    last_inserted = []
    last_confirmed_bonds = None
    pulse_complete = False

    print(
        f"{phase_name} [{pulse_id}]: pulse-start inventory contains "
        f"{len(initial_surface_indices)} candidate atoms; target coverage={target_fraction:.3f}"
    )

    for attempt in range(1, max_attempts + 1):
        if inventory["target_reached"]:
            pulse_complete = True
            break

        attempt_complete, inserted, confirmed_bonds = _run_single_precursor_attempt(
            atoms, phase_name, precursor_cfg, config, md_calc, monitor_calc,
            out_dir, rng,
            surface_indices_override=initial_surface_indices,
            attempt_index=attempt,
            pulse_id=pulse_id,
        )
        last_inserted = inserted
        last_confirmed_bonds = confirmed_bonds
        center_index = int(inserted[int(precursor_cfg["center_index"])])
        if confirmed_bonds is not None:
            bonded = [
                int(site["atom_index"])
                for site in inventory["sites"]
                if tuple(sorted((center_index, int(site["atom_index"])))) in confirmed_bonds
            ]
            distances = {
                str(idx): float(atoms.get_distance(center_index, idx, mic=True))
                for idx in bonded
            }
        else:
            bonded, distances = count_center_surface_bonds(
                atoms, center_index, initial_surface_indices, inventory_cutoff
            )

        # Only a completed precursor attempt contributes to the pulse inventory.
        # A site can be counted once even if several precursors contact it.
        newly_reacted = []
        if attempt_complete:
            if phase_name == "precursor_A":
                atoms.arrays[IS_DEPOSITED_CENTER][center_index] = 1
            for site in inventory["sites"]:
                idx = int(site["atom_index"])
                if idx in bonded and site["status"] == "available":
                    site["status"] = "reacted_proxy"
                    site["first_reacted_attempt"] = attempt
                    site["contacting_precursor_center_index"] = center_index
                    site["contacting_precursor_center_atom_id"] = int(
                        atoms.arrays[ATOM_ID][center_index]
                    )
                    site["contact_distance_A"] = float(distances[str(idx)])
                    newly_reacted.append(idx)

        attempt_record = {
            "attempt": attempt,
            "precursor_center_index": center_index,
            "precursor_center_atom_id": int(atoms.arrays[ATOM_ID][center_index]),
            "precursor_attempt_completed": bool(attempt_complete),
            "bonded_initial_surface_indices_at_end": bonded,
            "connectivity_source": "molecular_graph" if confirmed_bonds is not None else "distance_cutoff",
            "newly_counted_surface_indices": newly_reacted,
            "unique_reacted_site_count_after_attempt": sum(
                site["status"] == "reacted_proxy" for site in inventory["sites"]
            ),
            "note": (
                "reacted_proxy is based on a persistent molecular-graph edge after a completed attempt"
                if confirmed_bonds is not None else
                "reacted_proxy is based on a distance-cutoff contact after a completed attempt"
            ) + "; validate chemically.",
        }
        inventory["attempts"].append(attempt_record)
        save_inventory()

        print(
            f"{phase_name} [{pulse_id}] attempt {attempt}/{max_attempts}: "
            f"precursor_completed={attempt_complete}, newly_counted_sites={len(newly_reacted)}, "
            f"coverage={inventory['surface_coverage_fraction']:.3f}"
        )

        if not attempt_complete:
            purge_result = purge_detached_inserted_components(
                atoms, config.get("connectivity", {})
            )
            attempt_record["purge_after_incomplete_attempt"] = purge_result
            save_inventory()
        if inventory["target_reached"]:
            pulse_complete = True
            break

    if not pulse_complete and "stopped_reason" not in inventory:
        inventory["stopped_reason"] = "max_adsorption_attempts_reached_before_target_coverage"
    inventory["pulse_completed"] = bool(pulse_complete)
    inventory["surface_inventory_note"] = (
        "Candidate sites are detected geometrically at pulse start. reacted_proxy is based on "
        "persistent graph edges after a completed precursor attempt when connectivity is enabled; "
        "otherwise it uses the configured distance cutoff. This remains a chemistry proxy."
    )
    save_inventory()

    summary_path = pulse_dir / "pulse_summary.json"
    summary_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return pulse_complete, last_inserted, last_confirmed_bonds


def run_water_phase(atoms, phase_name, precursor_cfg, config, md_calc, monitor_calc,
                    out_dir, rng, pulse_id="pulse_001"):
    """Run water attempts and inventory distinct Si-bound OH molecules per A-Si."""
    if not config.get("connectivity", {}).get("enabled", True):
        raise ValueError("precursor_B water occupancy requires connectivity.enabled: true")
    ensure_atom_provenance(atoms)
    deposited_si = deposited_center_indices(atoms, ROLE_PRECURSOR_A)
    if not deposited_si:
        raise RuntimeError("precursor_B cannot run: no successfully deposited precursor_A Si centers")

    inventory_cfg = precursor_cfg.get("surface_inventory", {})
    max_attempts = max(1, int(inventory_cfg.get("max_adsorption_attempts", 10)))
    target_per_si = int(inventory_cfg.get("target_waters_per_si", 2))
    minimum_per_si = int(inventory_cfg.get("minimum_waters_per_si", 1))
    if target_per_si < 1 or minimum_per_si < 1 or minimum_per_si > target_per_si:
        raise ValueError("Water occupancy requires 1 <= minimum_waters_per_si <= target_waters_per_si")
    center_local = int(precursor_cfg["center_index"])
    hydrogen_local = [int(i) for i in precursor_cfg["ligand_indices"]]
    template = read(precursor_cfg["molecule_file"])
    if center_local < 0 or center_local >= len(template) or template[center_local].symbol != "O":
        raise ValueError("precursor_B.center_index must identify the water oxygen")
    if len(hydrogen_local) != 2 or any(
        i < 0 or i >= len(template) or template[i].symbol != "H" for i in hydrogen_local
    ):
        raise ValueError("precursor_B.ligand_indices must identify exactly two water hydrogen atoms")

    pulse_dir = out_dir / phase_name / str(pulse_id)
    pulse_dir.mkdir(parents=True, exist_ok=True)
    si_ids = [int(atoms.arrays[ATOM_ID][i]) for i in deposited_si]
    inventory = {
        "phase": phase_name,
        "pulse_id": str(pulse_id),
        "deposited_si_atom_ids": si_ids,
        "minimum_waters_per_si": minimum_per_si,
        "target_waters_per_si": target_per_si,
        "max_adsorption_attempts": max_attempts,
        "occupancy": {str(atom_id): [] for atom_id in si_ids},
        "attempts": [],
    }
    # Carry qualifying waters from earlier cycles forward instead of resetting
    # coverage at every B half-cycle. The full graph is a geometry-based snapshot;
    # new-water acceptance below additionally uses per-attempt persistent edges.
    connectivity_cfg = config.get("connectivity", {})
    base_cutoffs = build_base_cutoffs(atoms, connectivity_cfg.get("atomic_radii_A"))
    existing_bonds = detect_bonds_hysteresis(
        atoms,
        (),
        base_cutoffs,
        float(connectivity_cfg.get("form_scale", 1.10)),
        float(connectivity_cfg.get("break_scale", 1.70)),
        pair_cutoffs=connectivity_cfg.get("pair_cutoffs_A"),
    )
    water_molecules = {}
    for index, role in enumerate(atoms.arrays[ATOM_ROLE]):
        if int(role) != ROLE_WATER or int(atoms.arrays[IS_CENTER][index]) != 1:
            continue
        molecule_id = int(atoms.arrays[MOLECULE_ID][index])
        molecule_indices = [
            j for j, candidate_id in enumerate(atoms.arrays[MOLECULE_ID])
            if int(candidate_id) == molecule_id and atoms[j].symbol == "H"
        ]
        water_molecules[molecule_id] = (index, molecule_indices)
    for molecule_id, (oxygen_index, hydrogen_indices) in water_molecules.items():
        endpoint = classify_water_oh_endpoint(
            existing_bonds, oxygen_index, hydrogen_indices, deposited_si
        )
        if not endpoint["qualifies"]:
            continue
        si_atom_id = int(atoms.arrays[ATOM_ID][endpoint["si_index"]])
        records = inventory["occupancy"].get(str(si_atom_id))
        if records is None or molecule_id in {int(x["molecule_id"]) for x in records}:
            continue
        records.append({
            "molecule_id": molecule_id,
            "oxygen_atom_id": int(atoms.arrays[ATOM_ID][oxygen_index]),
            "hydrogen_atom_ids": [int(atoms.arrays[ATOM_ID][i]) for i in hydrogen_indices],
            "retained_oh_hydrogen_atom_ids": [
                int(atoms.arrays[ATOM_ID][i])
                for i in endpoint["bonded_original_hydrogen_indices"]
            ],
            "attempt": "previous_cycle",
        })
    inventory_path = pulse_dir / "water_occupancy.json"

    def save_inventory():
        status = water_occupancy_status(
            {si: [item["molecule_id"] for item in waters]
             for si, waters in inventory["occupancy"].items()},
            minimum_per_si,
            target_per_si,
        )
        inventory.update(status)
        inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    save_inventory()
    last_inserted, last_bonds = [], None
    for attempt in range(1, max_attempts + 1):
        if inventory["full_occupancy"]:
            break
        # Target only Si centers still below the configured occupancy cap.
        current_map = indices_for_atom_ids(atoms, si_ids)
        eligible_si_ids = [
            atom_id for atom_id in si_ids
            if len(inventory["occupancy"][str(atom_id)]) < target_per_si
            and atom_id in current_map
        ]
        if not eligible_si_ids:
            break
        eligible_si_indices = [current_map[atom_id] for atom_id in eligible_si_ids]
        complete, inserted, bonds = _run_single_precursor_attempt(
            atoms, phase_name, precursor_cfg, config, md_calc, monitor_calc,
            out_dir, rng, surface_indices_override=eligible_si_indices,
            attempt_index=attempt, pulse_id=pulse_id,
        )
        last_inserted, last_bonds = inserted, bonds
        oxygen_index = inserted[int(precursor_cfg["center_index"])]
        h_indices = [inserted[int(i)] for i in precursor_cfg["ligand_indices"]]
        endpoint = classify_water_oh_endpoint(
            bonds or (), oxygen_index, h_indices, eligible_si_indices
        )
        endpoint["original_hydrogen_neighbor_atom_ids"] = {
            str(atoms.arrays[ATOM_ID][int(h_idx)]): [
                int(atoms.arrays[ATOM_ID][int(neighbor)])
                for neighbor in neighbors
            ]
            for h_idx, neighbors in endpoint["original_hydrogen_neighbors"].items()
        }
        molecule_id = int(atoms.arrays[MOLECULE_ID][oxygen_index])
        water_atom_ids = [int(atoms.arrays[ATOM_ID][i]) for i in inserted]
        assigned_si_atom_id = None
        if complete and endpoint["qualifies"]:
            si_index = int(endpoint["si_index"])
            assigned_si_atom_id = int(atoms.arrays[ATOM_ID][si_index])
            occupancy = inventory["occupancy"][str(assigned_si_atom_id)]
            if molecule_id not in {int(item["molecule_id"]) for item in occupancy}:
                occupancy.append({
                    "molecule_id": molecule_id,
                    "oxygen_atom_id": int(atoms.arrays[ATOM_ID][oxygen_index]),
                    "hydrogen_atom_ids": [int(atoms.arrays[ATOM_ID][i]) for i in h_indices],
                    "retained_oh_hydrogen_atom_ids": [
                        int(atoms.arrays[ATOM_ID][i])
                        for i in endpoint["bonded_original_hydrogen_indices"]
                    ],
                    "other_original_hydrogen_atom_ids": [
                        int(atoms.arrays[ATOM_ID][i])
                        for i in h_indices
                        if i not in endpoint["bonded_original_hydrogen_indices"]
                    ],
                    "original_hydrogen_neighbor_atom_ids": endpoint[
                        "original_hydrogen_neighbor_atom_ids"
                    ],
                    "attempt": attempt,
                })
        record = {
            "attempt": attempt,
            "completed": bool(complete),
            "water_molecule_id": molecule_id,
            "water_atom_ids": water_atom_ids,
            "endpoint": endpoint,
            "assigned_deposited_si_atom_id": assigned_si_atom_id,
            "qualifies": bool(complete and endpoint["qualifies"]),
        }
        inventory["attempts"].append(record)
        if not record["qualifies"]:
            record["purge"] = purge_detached_inserted_components(
                atoms, config.get("connectivity", {})
            )
        save_inventory()
        print(
            f"{phase_name} [{pulse_id}] attempt {attempt}/{max_attempts}: "
            f"qualifies={record['qualifies']}, occupancy={inventory['occupancy_counts']}"
        )

    save_inventory()
    inventory["pulse_completed"] = bool(inventory["minimum_success"])
    inventory["termination_reason"] = (
        "full_occupancy_reached" if inventory["full_occupancy"] else
        "minimum_success_attempt_budget_exhausted" if inventory["minimum_success"] else
        "minimum_occupancy_not_reached"
    )
    inventory["chemistry_note"] = (
        "Water qualifies only when a persistent graph contains exactly one deposited-Si/O edge "
        "and exactly one original water O/H edge. Geometry-based graph edges require validation."
    )
    save_inventory()
    (pulse_dir / "pulse_summary.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    return bool(inventory["minimum_success"]), last_inserted, last_bonds


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
        print(f"Configuration loaded from: {cfg.get('_config_path', args.config)}")
        print("Resolved input files:")
        for label, filename in [("slab", cfg.get("system", {}).get("slab_file"))]:
            if filename:
                print(f"  {label}: {filename} (exists={Path(filename).is_file()})")
        for i, filename in enumerate(cfg.get("committee", {}).get("model_paths", []), start=1):
            print(f"  model_{i}: {filename} (exists={Path(filename).is_file()})")
        for name, phase_cfg in cfg.get("chemistry", {}).items():
            filename = phase_cfg.get("molecule_file")
            if filename:
                print(f"  {name} molecule: {filename} (exists={Path(filename).is_file()})")
        system_cfg = cfg.get("system", {})
        print("Initial cell setup:")
        print(f"  supercell_xy: {system_cfg.get('supercell_xy', [1, 1])}")
        print(f"  cell_z_A: {system_cfg.get('cell_z_A', 'use input cell')}")
        print("Planned sequence:")
        for i in range(int(cfg["sampling"]["max_cycles"])):
            for phase in phases:
                print(f"cycle {i + 1}: {phase}")
        print("Dry-run only: no MACE/PLUMED dynamics were run.")
        return

    # Validate that the configured water atom map matches the required O/H/OH
    # classifier before loading accelerator-backed calculators.
    water_cfg = cfg.get("chemistry", {}).get("precursor_B", {})
    if water_cfg.get("saturation", {}).get("criterion") == "water_oh":
        water_template = read(water_cfg["molecule_file"])
        oxygen_index = int(water_cfg["center_index"])
        hydrogen_indices = [int(i) for i in water_cfg.get("ligand_indices", [])]
        if (
            oxygen_index < 0 or oxygen_index >= len(water_template)
            or water_template[oxygen_index].symbol != "O"
        ):
            raise ValueError("precursor_B.center_index must identify the water oxygen")
        if len(hydrogen_indices) != 2 or any(
            i < 0 or i >= len(water_template) or water_template[i].symbol != "H"
            for i in hydrogen_indices
        ):
            raise ValueError("precursor_B.ligand_indices must identify exactly two water hydrogen atoms")

    atoms = read(cfg["system"]["slab_file"])
    if len(atoms) == 0:
        raise ValueError("Initial slab is empty")
    atoms = configure_initial_cell(
        atoms,
        cfg.get("system", {}),
        cfg.get("sampling", {}),
        cfg.get("connectivity", {}),
    )
    surface_anchors = atoms.info["_ald_surface_anchor_indices"]
    print(
        f"Initialized substrate anchor component: {len(surface_anchors)} atoms; "
        f"z range={np.min(atoms.positions[surface_anchors, 2]):.3f}–"
        f"{np.max(atoms.positions[surface_anchors, 2]):.3f} A"
    )

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

    for cycle in range(int(cfg["sampling"]["max_cycles"])):
        for phase_name in phases:
            if phase_name in ("purge_A", "purge_B"):
                purge_result = purge_detached_inserted_components(
                    atoms, cfg.get("connectivity", {})
                )
                purge_dir = out_dir / phase_name / f"cycle_{cycle + 1}"
                purge_dir.mkdir(parents=True, exist_ok=True)
                (purge_dir / "purge_summary.json").write_text(
                    json.dumps(purge_result, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"{phase_name}: removed {purge_result['removed_atom_count']} detached "
                    "inserted atoms/components"
                )
                write(out_dir / f"after_{phase_name}_cycle_{cycle + 1}.traj", atoms)
                continue
            if phase_name not in cfg["chemistry"]:
                raise KeyError(f"No chemistry configuration found for phase '{phase_name}'")
            success, inserted, _ = run_phase(
                atoms, phase_name, cfg["chemistry"][phase_name], cfg,
                md_calc, monitor_calc, out_dir, rng,
                pulse_id=f"cycle_{cycle + 1}",
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