from typing import Dict, List, Set, Tuple

from rdkit import Chem

# =================================================================================================
# Mapping completion.
# =================================================================================================


def complete_reaction_mapping(mapped_reaction: str) -> str:
    """
    Ensure every atom on both sides carries an atom-map number, assigning fresh
    numbers to the unmapped atoms only if some atom is unmapped.

    An unmapped atom is invisible to the reaction-centre logic (map numbers <= 0
    are ignored), so it silently corrupts the template built around it — e.g. an
    unmapped ring carbon is left out of the centre, the ring is torn apart at
    small radius, and the reaction becomes inapplicable.

    The completion is done with RDKit (not the regex `add_missing_mappings_both_sides`)
    so the molecule is preserved exactly: bracketising a bare atom in text, e.g.
    ``c`` -> ``[c]``, would drop its implicit hydrogen (H1 -> H0) and change the
    chemistry, which then makes the template fail to match the real substrate.

    Returns the reaction unchanged if it is already fully mapped or cannot be
    parsed (idempotent).
    """
    try:
        substrates_smi, products_smi = mapped_reaction.split(">>")
    except ValueError:
        return mapped_reaction
    mol_substrates = Chem.MolFromSmiles(substrates_smi)
    mol_products = Chem.MolFromSmiles(products_smi)
    if mol_substrates is None or mol_products is None:
        return mapped_reaction
    fully_mapped = all(a.GetAtomMapNum() for a in mol_substrates.GetAtoms()) and all(
        a.GetAtomMapNum() for a in mol_products.GetAtoms()
    )
    if fully_mapped:
        return mapped_reaction

    used = {
        atom.GetAtomMapNum()
        for mol in (mol_substrates, mol_products)
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    next_map_num = max(used, default=0) + 1
    for mol in (mol_substrates, mol_products):
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(next_map_num)
                next_map_num += 1
    return Chem.MolToSmiles(mol_substrates) + ">>" + Chem.MolToSmiles(mol_products)


# =================================================================================================
# Atom signatures.
# =================================================================================================


def _atom_signature(atom: Chem.Atom, ring_counts: Tuple[int, int]) -> Tuple:
    """
    Local signature of a mapped atom covering every property that can move its own
    ECFP/Morgan invariant (formal charge, H count, valence, aromaticity, ring
    membership, degree), plus the (bond order, neighbor atom-map number) of each of
    its bonds. Two atoms sharing the same atom-map number but with a different
    signature on each side of a reaction have been affected by the reaction, even
    if none of their own bonds were formed or broken (e.g. an atom pulled into a
    newly closed ring, rearomatized, or dropped from one of two fused rings,
    elsewhere in the molecule).

    `ring_counts` is `(num_rings, num_aromatic_rings)`: how many SSSR rings the
    atom belongs to, and how many of those rings are fully aromatic.
    """
    neighbor_sig = tuple(sorted(
        (bond.GetBondTypeAsDouble(), bond.GetOtherAtom(atom).GetAtomMapNum())
        for bond in atom.GetBonds()
    ))
    return (
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetTotalValence(),
        atom.GetIsAromatic(),
        ring_counts,
        atom.GetDegree(),
        neighbor_sig,
    )


def _map_num_to_atom(mol: Chem.Mol) -> Dict[int, Chem.Atom]:
    return {atom.GetAtomMapNum(): atom for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}


def _ring_counts_by_map_num(mol: Chem.Mol) -> Dict[int, Tuple[int, int]]:
    """
    For each mapped atom in `mol`, count how many SSSR rings it belongs to, and
    how many of those rings are fully aromatic (every bond in the ring aromatic).
    """
    ring_info = mol.GetRingInfo()
    counts = {atom.GetAtomMapNum(): [0, 0] for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}
    for atom_ring, bond_ring in zip(ring_info.AtomRings(), ring_info.BondRings()):
        is_aromatic_ring = all(mol.GetBondWithIdx(b).GetIsAromatic() for b in bond_ring)
        for atom_idx in atom_ring:
            map_num = mol.GetAtomWithIdx(atom_idx).GetAtomMapNum()
            if map_num <= 0:
                continue
            counts[map_num][0] += 1
            if is_aromatic_ring:
                counts[map_num][1] += 1
    return {map_num: tuple(v) for map_num, v in counts.items()}


# =================================================================================================
# Reaction centre identification.
# =================================================================================================


def get_reaction_centre_map_nums(mapped_reaction: str) -> Set[int]:
    """
    Given a mapped reaction SMILES (e.g. the "mapped_rxn" output of
    `map_reaction_with_rxnmapper`), return the atom-map numbers of the atoms whose
    local environment changes between substrates and products: bonds formed or
    broken, bond order changes, charge/valence/H-count changes, ring or
    aromaticity changes, or atoms disappearing entirely (e.g. leaving groups).

    Returns an empty set if either side cannot be parsed (e.g. a malformed
    reaction SMILES), so batch processing is not interrupted by bad entries.
    """
    mapped_reaction = complete_reaction_mapping(mapped_reaction)
    substrates_smi, products_smi = mapped_reaction.split(">>")
    mol_substrates = Chem.MolFromSmiles(substrates_smi)
    mol_products = Chem.MolFromSmiles(products_smi)
    if mol_substrates is None or mol_products is None:
        return set()

    substrate_atoms = _map_num_to_atom(mol_substrates)
    product_atoms = _map_num_to_atom(mol_products)
    substrate_ring_counts = _ring_counts_by_map_num(mol_substrates)
    product_ring_counts = _ring_counts_by_map_num(mol_products)

    centre_map_nums = set()
    for map_num, atom in substrate_atoms.items():
        product_atom = product_atoms.get(map_num)
        if product_atom is None:
            centre_map_nums.add(map_num)
            continue
        substrate_sig = _atom_signature(atom, substrate_ring_counts[map_num])
        product_sig = _atom_signature(product_atom, product_ring_counts[map_num])
        if substrate_sig != product_sig:
            centre_map_nums.add(map_num)
    return centre_map_nums


def get_substrate_reaction_centre_atom_indices(mapped_reaction: str) -> List[int]:
    """
    Given a mapped reaction SMILES, return the atom indices (within the substrates
    side, as parsed by `Chem.MolFromSmiles` on the left-hand side of the reaction)
    of the atoms belonging to the reaction centre.

    Returns an empty list if the substrates cannot be parsed, so batch processing
    is not interrupted by malformed reactions.
    """
    mapped_reaction = complete_reaction_mapping(mapped_reaction)
    substrates_smi = mapped_reaction.split(">>")[0]
    mol_substrates = Chem.MolFromSmiles(substrates_smi)
    if mol_substrates is None:
        return []

    centre_map_nums = get_reaction_centre_map_nums(mapped_reaction)
    return sorted(
        atom.GetIdx()
        for atom in mol_substrates.GetAtoms()
        if atom.GetAtomMapNum() in centre_map_nums
    )


def get_substrate_reaction_centre_atom_map_nums(mapped_reaction: str) -> List[int]:
    """
    Given a mapped reaction SMILES, return the atom-map numbers (as assigned by the
    atom mapper, e.g. the `:7` in `[CH3:7]`) of the atoms belonging to the reaction
    centre, instead of their RDKit atom indices.
    """
    return sorted(get_reaction_centre_map_nums(mapped_reaction))
