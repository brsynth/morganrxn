from typing import List

from rdkit import Chem

from morganrxn.core.centre import complete_reaction_mapping, get_reaction_centre_map_nums
from morganrxn.core.molecule_utils import get_atoms_within_radius

# =================================================================================================
# Per-atom ECFP-invariant SMARTS.
# =================================================================================================


def _atom_ecfp_query(atom: Chem.Atom) -> str:
    """
    SMARTS primitive of a single atom, carrying every property that feeds RDKit's
    Morgan/ECFP atom invariant so the template atom matches exactly the atoms that
    would produce the same fingerprint bit:

        <element> ; X<total degree> ; D<heavy degree> ; H<total Hs> ; v<valence>
                  ; <formal charge> ; a (if aromatic) ; R | !R

    e.g. ``[c;X3;D3;H0;v4;+0;a;R]`` or ``[Cl;X0;D0;H0;v0;-1;!R]``.

    The formal charge is part of RDKit's Morgan invariant and is required for the
    template to reproduce charged species (e.g. a chloride leaving group ``[Cl-]``
    rather than a neutral ``[Cl]`` radical).

    The values are read from `atom` as it sits in its full molecule, so a template
    atom on the fragment boundary keeps its real degree/valence/H count rather than
    the (lower) ones it would have in the isolated fragment.
    """
    symbol = atom.GetSymbol()
    if atom.GetIsAromatic():
        symbol = symbol.lower()
    parts = [
        symbol,
        f"X{atom.GetTotalDegree()}",
        f"D{atom.GetDegree()}",
        f"H{atom.GetTotalNumHs()}",
        f"v{atom.GetTotalValence()}",
        f"{atom.GetFormalCharge():+d}",
    ]
    if atom.GetIsAromatic():
        parts.append("a")
    parts.append("R" if atom.IsInRing() else "!R")
    return "[" + ";".join(parts) + "]"


# =================================================================================================
# Fragment extraction around the reaction centre.
# =================================================================================================


def _centre_atom_indices(mol: Chem.Mol, centre_map_nums) -> List[int]:
    """Indices, within `mol`, of the atoms whose atom-map number is in the reaction centre."""
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() in centre_map_nums]


def _fragment_query_mol(mol: Chem.Mol, atoms_to_use: List[int]) -> Chem.Mol:
    """
    Build a query molecule holding `atoms_to_use` (indices in `mol`), where every
    atom is replaced by its ECFP-invariant SMARTS query and the bonds internal to
    the fragment are kept with their original bond order. Atom-map numbers are
    carried over.
    """
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(atoms_to_use)}
    rw = Chem.RWMol()
    for old_idx in atoms_to_use:
        atom = mol.GetAtomWithIdx(old_idx)
        query_atom = Chem.AtomFromSmarts(_atom_ecfp_query(atom))
        if atom.GetAtomMapNum():
            query_atom.SetAtomMapNum(atom.GetAtomMapNum())
        rw.AddAtom(query_atom)
    for bond in mol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin_idx in old_to_new and end_idx in old_to_new:
            rw.AddBond(old_to_new[begin_idx], old_to_new[end_idx], bond.GetBondType())
    return rw.GetMol()


def _grouped_fragment_smarts(mol: Chem.Mol, atom_indices: List[int]) -> str:
    """
    SMARTS of the selected `atom_indices`, grouped by the molecule (connected
    component of `mol`) they belong to: each originating molecule's fragments are
    wrapped in parentheses ``(...)`` and the groups are joined with ``.``.

    The parentheses tell RDKit that the enclosed (possibly disconnected) fragments
    belong to the *same* reactant molecule. This is what makes intramolecular
    templates applicable: when a single substrate molecule carries two reaction
    sites (e.g. a diester), both template fragments must match within that one
    molecule rather than being treated as two separate reactants. Two distinct
    molecules instead give ``(...).(...)`` (two reactant templates).
    """
    atom_set = set(atom_indices)
    groups = []
    for component in Chem.GetMolFrags(mol):
        selected = [idx for idx in component if idx in atom_set]
        if selected:
            groups.append(Chem.MolToSmarts(_fragment_query_mol(mol, selected)))
    return ".".join("({})".format(group) for group in groups)


def _mapped_map_nums(mol: Chem.Mol) -> set:
    return {atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}


# =================================================================================================
# Reaction template.
# =================================================================================================


def get_reaction_template(mapped_reaction: str, radius: int) -> str:
    """
    Build a reaction template (SMARTS) from a mapped reaction SMILES.

    The substrate side keeps every atom lying within `radius` bonds of an atom of
    the reaction centre (as identified by `get_reaction_centre_map_nums`). The
    product side is then restricted to *exactly* the atoms corresponding to those
    retained substrate atoms (same atom-map numbers), plus any atom that exists
    only on the product side (a genuine addition, e.g. an oxygen coming from
    water). Crucially, the product side is **not** independently radius-expanded:
    doing so would pull in product neighbours absent from the substrate, which
    RDKit would then have to *create* when the template is applied, yielding
    phantom atoms / nonsense products. Restricting to the substrate's atom-map set
    keeps the two sides balanced and the template applicable.

    Every template atom is annotated with the full set of ECFP/Morgan atom
    invariants (element, total degree X, heavy degree D, H count, valence v,
    formal charge, aromaticity, ring membership) so the template stays consistent
    with the fingerprint used elsewhere in the pipeline. Atom-map numbers are
    carried over so the two sides remain in correspondence.

    Fragments are grouped by the molecule they come from and each group is wrapped
    in parentheses (see `_grouped_fragment_smarts`), so an intramolecular reaction
    whose single substrate carries several reaction sites stays a single reactant
    template ``(...)`` and remains applicable, while genuinely multi-molecule sides
    become ``(...).(...)``.

    radius=0 keeps only the reaction centre atoms; radius=1 adds their immediate
    neighbours, and so on.

    The mapping is completed first (see `complete_reaction_mapping`): an unmapped
    atom would otherwise be left out of the centre and tear apart the structure
    around it (e.g. a partial aromatic ring), making even a radius-0 template
    inapplicable.
    """
    mapped_reaction = complete_reaction_mapping(mapped_reaction)
    substrates_smi, products_smi = mapped_reaction.split(">>")
    mol_substrates = Chem.MolFromSmiles(substrates_smi)
    mol_products = Chem.MolFromSmiles(products_smi)
    if mol_substrates is None or mol_products is None:
        raise ValueError(f"Could not parse mapped reaction: {mapped_reaction}")

    centre_map_nums = get_reaction_centre_map_nums(mapped_reaction)

    # Substrate side: reaction centre + radius-R neighbourhood.
    centre_indices = _centre_atom_indices(mol_substrates, centre_map_nums)
    substrate_atoms = [
        int(idx) for idx in get_atoms_within_radius(mol_substrates, centre_indices, radius)
    ]

    # Atom-map numbers retained on the substrate side, and all substrate maps.
    retained_maps = {
        mol_substrates.GetAtomWithIdx(i).GetAtomMapNum()
        for i in substrate_atoms
        if mol_substrates.GetAtomWithIdx(i).GetAtomMapNum() > 0
    }
    substrate_maps = _mapped_map_nums(mol_substrates)

    # Product side: the retained atoms (same map numbers) plus product-only atoms
    # (genuine additions, e.g. an O from water or a leaving ion). Charge is encoded
    # per atom, so charged heavy species (e.g. a chloride [Cl-]) are reproduced
    # faithfully. Bare protons are still skipped: RDKit drops explicit [H+] atoms
    # during application (leaving an empty fragment) even with the charge encoded,
    # and a spectator proton is only charge balance, never part of the ECFP.
    product_atoms = [
        atom.GetIdx()
        for atom in mol_products.GetAtoms()
        if atom.GetAtomMapNum() in retained_maps
        or (
            atom.GetAtomMapNum() > 0
            and atom.GetAtomMapNum() not in substrate_maps
            and atom.GetAtomicNum() != 1
        )
    ]

    substrate_template = _grouped_fragment_smarts(mol_substrates, substrate_atoms)
    product_template = _grouped_fragment_smarts(mol_products, product_atoms)
    return substrate_template + ">>" + product_template
