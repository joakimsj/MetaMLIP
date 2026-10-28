#!/usr/bin/env python3
"""Reusable periodic bond-connectivity and fragment analysis for ALD/NEB.

Bond assignments are geometric hypotheses based on covalent radii, not quantum
chemical bond orders. Validate pair cutoffs and scales for the chemistry/model.
ASE atom indices are zero-based. Periodic image offsets are not stored in the
simple (i, j) graph, so this graph is for connectivity/species analysis rather
than crystallographic bond enumeration.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, Mapping, Optional

import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

Bond = tuple[int, int]


def build_base_cutoffs(atoms, pair_overrides: Optional[Mapping[str, float]] = None) -> np.ndarray:
    """Return ASE natural covalent radii, optionally overridden per element.

    ``pair_overrides`` keys can be element symbols (e.g. ``{"H": 0.31}``) or
    pair keys (e.g. ``{"Si-O": 1.80}``). Pair overrides are interpreted as
    the *sum* of radii for that element pair, in Angstrom.
    """
    radii = np.asarray(natural_cutoffs(atoms), dtype=float)
    # Store pair sums separately on the function result's caller when needed;
    # element-level keys conveniently override individual atomic radii.
    for i, atom in enumerate(atoms):
        if pair_overrides and atom.symbol in pair_overrides:
            radii[i] = float(pair_overrides[atom.symbol])
    if np.any(radii <= 0):
        raise ValueError("Covalent radii/cutoffs must be positive")
    return radii


def detect_bonds_hysteresis(
    atoms,
    prev_bonds: Iterable[Bond] = (),
    base_cutoffs: Optional[np.ndarray] = None,
    form_scale: float = 1.10,
    break_scale: float = 1.70,
    reactive_indices: Optional[Iterable[int]] = None,
    pair_cutoffs: Optional[Mapping[str, float]] = None,
) -> set[Bond]:
    """Build a periodic bond graph with hysteresis and ASE neighbor lists.

    ``pair_cutoffs`` maps e.g. ``"Si-O"`` to the unscaled pair-distance
    threshold in Angstrom. Otherwise summed natural covalent radii are used.
    Previously present bonds persist until the break threshold is exceeded;
    new bonds form below the formation threshold.
    """
    if form_scale <= 0 or break_scale < form_scale:
        raise ValueError("Require form_scale > 0 and break_scale >= form_scale")
    n = len(atoms)
    if base_cutoffs is None:
        base_cutoffs = build_base_cutoffs(atoms)
    base_cutoffs = np.asarray(base_cutoffs, dtype=float)
    if base_cutoffs.shape != (n,):
        raise ValueError("base_cutoffs must contain one radius per atom")
    prev = {tuple(sorted((int(i), int(j)))) for i, j in prev_bonds if i != j}
    reactive = set(range(n)) if reactive_indices is None else {int(i) for i in reactive_indices}
    if any(i < 0 or i >= n for i in reactive):
        raise IndexError("reactive_indices contains an invalid atom index")

    # NeighborList uses per-atom radii and handles periodic boundaries/cell offsets.
    # Use the largest possible break radius to obtain candidate pairs, then apply
    # exact pair-specific thresholds below.
    max_radius = float(np.max(base_cutoffs)) if n else 0.0
    nl = NeighborList(
        [max_radius * break_scale] * n,
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    nl.update(atoms)
    bonds: set[Bond] = set()
    for i in range(n):
        js, offsets = nl.get_neighbors(i)
        for j, offset in zip(js, offsets):
            j = int(j)
            if j <= i or (i not in reactive and j not in reactive):
                continue
            key = "-".join(sorted((atoms[i].symbol, atoms[j].symbol)))
            base = float(pair_cutoffs[key]) if pair_cutoffs and key in pair_cutoffs else float(base_cutoffs[i] + base_cutoffs[j])
            r_form = form_scale * base
            r_break = break_scale * base
            # offset is the periodic image of j relative to i, as returned by ASE.
            vector = atoms.positions[j] + np.asarray(offset) @ np.asarray(atoms.cell) - atoms.positions[i]
            distance = float(np.linalg.norm(vector))
            pair = (i, j)
            if pair in prev:
                if distance < r_break:
                    bonds.add(pair)
            elif distance < r_form:
                bonds.add(pair)
    return bonds


def diff_bonds(previous: Iterable[Bond], current: Iterable[Bond]):
    previous, current = set(previous), set(current)
    return current - previous, previous - current


def build_connectivity_matrix(n_atoms: int, bonds: Iterable[Bond]) -> np.ndarray:
    """Return a symmetric uint8 adjacency matrix."""
    matrix = np.zeros((int(n_atoms), int(n_atoms)), dtype=np.uint8)
    for i, j in bonds:
        if i == j or min(i, j) < 0 or max(i, j) >= n_atoms:
            raise IndexError(f"Invalid bond index {(i, j)} for {n_atoms} atoms")
        matrix[i, j] = matrix[j, i] = 1
    return matrix


def find_connected_components(n_atoms: int, bonds: Iterable[Bond]) -> list[list[int]]:
    """Return graph connected components, including isolated atoms."""
    adjacency = [[] for _ in range(int(n_atoms))]
    for i, j in bonds:
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))
    seen = set()
    components = []
    for root in range(int(n_atoms)):
        if root in seen:
            continue
        component, queue = [], deque([root])
        seen.add(root)
        while queue:
            i = queue.popleft()
            component.append(i)
            for j in adjacency[i]:
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
        components.append(sorted(component))
    return components


def classify_fragments(atoms, bonds: Iterable[Bond], substrate_indices: Iterable[int]) -> list[dict]:
    """Summarize components and flag whether each contains substrate atoms."""
    substrate = {int(i) for i in substrate_indices}
    fragments = []
    for component_id, indices in enumerate(find_connected_components(len(atoms), bonds)):
        attached = sorted(substrate.intersection(indices))
        composition = {}
        for i in indices:
            symbol = atoms[i].symbol
            composition[symbol] = composition.get(symbol, 0) + 1
        fragments.append({
            "fragment_id": component_id,
            "atom_indices": indices,
            "atom_count": len(indices),
            "composition": dict(sorted(composition.items())),
            "contains_substrate_atoms": bool(attached),
            "substrate_atom_indices": attached,
            "classification": "substrate_connected" if attached else "detached_or_free",
        })
    return fragments


def analyze_connectivity(atoms, substrate_indices, previous_bonds=(), config=None):
    """Build graph, bond deltas, adjacency matrix and fragment inventory."""
    cfg = config or {}
    radii = build_base_cutoffs(atoms, cfg.get("atomic_radii_A"))
    bonds = detect_bonds_hysteresis(
        atoms, previous_bonds, radii,
        float(cfg.get("form_scale", 1.10)),
        float(cfg.get("break_scale", 1.70)),
        reactive_indices=cfg.get("reactive_indices"),
        pair_cutoffs=cfg.get("pair_cutoffs_A"),
    )
    formed, broken = diff_bonds(previous_bonds, bonds)
    return {
        "bonds": sorted([list(x) for x in bonds]),
        "formed_bonds": sorted([list(x) for x in formed]),
        "broken_bonds": sorted([list(x) for x in broken]),
        "connectivity_matrix": build_connectivity_matrix(len(atoms), bonds),
        "fragments": classify_fragments(atoms, bonds, substrate_indices),
    }
