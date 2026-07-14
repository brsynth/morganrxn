import numpy as np
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdmolops

# =================================================================================================
# Sanitize functions.
# =================================================================================================


def strip_cxsmiles(smi: str) -> str:
    return smi.split("|")[0].strip()


@lru_cache(maxsize=200_000)
def _sanitize_component_cached(smi_comp, remove_stereo=True):
    mol = Chem.MolFromSmiles(smi_comp)
    if mol is None:
        return smi_comp
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def sanitize_smiles(smi, verbose=False, remove_stereo=True):
    try:
        smi = strip_cxsmiles(smi)
        comps = smi.split(".")
        # drop wildcard-containing components
        comps = [c for c in comps if "*" not in c]
        comps_san = [_sanitize_component_cached(c, remove_stereo) for c in comps]
        comps_san.sort()
        return ".".join(comps_san)
    except Exception:
        if verbose:
            print("pb sanitize_smiles", smi)
        return smi


def sanitize_list_of_smiles(list_smiles):
    sanitized = {sanitize_smiles(smi) for smi in list_smiles}
    return [smi for smi in sanitized if smi]


# =================================================================================================
# Local environment utilities.
# =================================================================================================


def get_atoms_within_radius(mol, list_atoms, radius):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    # Compute the distance matrix
    distance_matrix = rdmolops.GetDistanceMatrix(mol)
    # Find atoms within the given radius
    selected_atoms = set()
    for atom_idx in list_atoms:
        # Get all atoms within `n` bonds from `atom_idx`
        neighbors = np.where(distance_matrix[atom_idx] <= radius)[0]
        selected_atoms.update(neighbors)
    return sorted(selected_atoms)


# =================================================================================================
# ECFP computations.
# =================================================================================================


def get_mol_ecfp(mol, ecfp_params):
    if ecfp_params["custom"]:
        return get_mol_ecfp_custom(mol, ecfp_params)

    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    if ecfp_params["folded"]:
        fpgen = AllChem.GetMorganGenerator(radius=ecfp_params["radius"], fpSize=ecfp_params["fpSize"])
        ecfp = fpgen.GetCountFingerprint(mol).ToList()
        return ecfp
    else:
        unfolded_fp = AllChem.GetMorganFingerprint(
            mol, radius=ecfp_params["radius"], useCounts=True, useFeatures=False
        )
        ecfp_unfold = []
        for bit in unfolded_fp.GetNonzeroElements():
            ecfp_unfold = ecfp_unfold + [bit] * unfolded_fp.GetNonzeroElements()[bit]
        return sorted(ecfp_unfold)


def get_mol_ecfp_atom_to_bits(mol, ecfp_params):
    if ecfp_params["custom"]:
        return get_mol_ecfp_custom_atom_to_bits(mol, ecfp_params)

    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    if ecfp_params["folded"]:
        bits_info = rdFingerprintGenerator.AdditionalOutput()
        bits_info.AllocateAtomToBits()
        fpgen = AllChem.GetMorganGenerator(radius=ecfp_params["radius"], fpSize=ecfp_params["fpSize"])
        _ = fpgen.GetCountFingerprint(mol, additionalOutput=bits_info).ToList()
        atoms_morgan_bits = {
            i: bits_info.GetAtomToBits()[i] for i in range(len(bits_info.GetAtomToBits()))
        }
        return atoms_morgan_bits
    else:
        bitInfo = {}  # Stores which atoms contribute to which bits
        # Compute the unfolded fingerprint and capture bit information
        _ = AllChem.GetMorganFingerprint(mol, radius=ecfp_params["radius"], useCounts=True, bitInfo=bitInfo)
        # Create a dictionary mapping atoms to their corresponding bits
        atoms_morgan_bits = {i: [] for i in range(mol.GetNumAtoms())}  # Initialize
        for bit_id, atom_env_list in bitInfo.items():
            for center_atom, env_radius in atom_env_list:
                atoms_morgan_bits[center_atom].append(
                    (env_radius, bit_id)
                )  # Store radius for sorting
        # Sort bits by radius for each atom and extract bit IDs
        atoms_morgan_bits = {
            atom: tuple(bit_id for _, bit_id in sorted(bits))
            for atom, bits in atoms_morgan_bits.items()
        }
        return atoms_morgan_bits


# =================================================================================================
# ECFP custom.
# =================================================================================================


def make_custom_atom_invariants(
    mol,
    use_atomic_num=True,
    use_degree=False,
    use_formal_charge=False,
    use_num_h=False,
    use_valence=False,
    use_aromatic=False,
    use_ring=False,
):
    """
    Build one integer invariant per atom.

    By default: atomic number only.
    You can switch on extra atom-level properties if needed.
    """
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    invariants = []
    for atom in mol.GetAtoms():
        fields = []

        if use_atomic_num:
            fields.append(atom.GetAtomicNum())
        if use_degree:
            fields.append(atom.GetDegree())
        if use_formal_charge:
            fields.append(atom.GetFormalCharge())
        if use_num_h:
            fields.append(atom.GetTotalNumHs())
        if use_valence:
            fields.append(atom.GetTotalValence())
        if use_aromatic:
            fields.append(int(atom.GetIsAromatic()))
        if use_ring:
            fields.append(int(atom.IsInRing()))

        # convert tuple of selected properties into a stable integer
        invariants.append(hash(tuple(fields)) & 0xFFFFFFFF)

    return invariants


def make_custom_bond_invariants(
    mol,
    use_bond_type=True,
    use_conjugation=False,
    use_aromatic=False,
    use_ring=False,
    use_stereo=False,
):
    """
    Build one integer invariant per bond.

    By default: bond type only.
    """
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    invariants = []
    for bond in mol.GetBonds():
        fields = []

        if use_bond_type:
            fields.append(int(bond.GetBondTypeAsDouble()))
        if use_conjugation:
            fields.append(int(bond.GetIsConjugated()))
        if use_aromatic:
            fields.append(int(bond.GetIsAromatic()))
        if use_ring:
            fields.append(int(bond.IsInRing()))
        if use_stereo:
            fields.append(int(bond.GetStereo()))

        invariants.append(hash(tuple(fields)) & 0xFFFFFFFF)

    return invariants


atom_invariant_params = {
    "use_atomic_num":    True,   # C, N, O, P, S...
    "use_degree":        True,   # nombre de voisins lourds (connectivité locale)
    "use_formal_charge": True,  # trop spécifique, rarement utile en biosynthèse
    "use_num_h":         False,  # redondant avec degree + valence
    "use_valence":       False,  # trop spécifique
    "use_aromatic":      True,   # aromatique vs aliphatique : info structurale clé
    "use_ring":          False,  # cycle : info utile mais ajoute de la spécificité
}
bond_invariant_params = {
    "use_bond_type":   True,    # simple/double/triple/aromatique
    "use_conjugation": False,
    "use_aromatic":    False,   # déjà capturé par bond_type (aromatic bond)
    "use_ring":        False,
    "use_stereo":      False,
}


def get_mol_ecfp_custom(
    mol,
    ecfp_params,
):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    custom_atom_invariants = make_custom_atom_invariants(mol, **atom_invariant_params)
    custom_bond_invariants = make_custom_bond_invariants(mol, **bond_invariant_params)

    if ecfp_params["folded"]:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=ecfp_params["radius"],
            fpSize=ecfp_params["fpSize"],
            includeChirality=False,   # no stereo
            useBondTypes=True,        # required if you want bond topology considered
        )
        ecfp = fpgen.GetCountFingerprint(
            mol,
            customAtomInvariants=custom_atom_invariants,
            customBondInvariants=custom_bond_invariants,
        ).ToList()
        return ecfp

    else:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=ecfp_params["radius"],
            includeChirality=False,
            useBondTypes=True,
        )
        fp = fpgen.GetSparseCountFingerprint(
            mol,
            customAtomInvariants=custom_atom_invariants,
            customBondInvariants=custom_bond_invariants,
        )

        ecfp_unfold = []
        for bit, count in fp.GetNonzeroElements().items():
            ecfp_unfold.extend([bit] * count)
        return sorted(ecfp_unfold)


def get_mol_ecfp_custom_atom_to_bits(
    mol,
    ecfp_params,
):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    custom_atom_invariants = make_custom_atom_invariants(mol, **atom_invariant_params)
    custom_bond_invariants = make_custom_bond_invariants(mol, **bond_invariant_params)

    bits_info = rdFingerprintGenerator.AdditionalOutput()
    bits_info.AllocateAtomToBits()

    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=ecfp_params["radius"],
        fpSize=ecfp_params.get("fpSize", 2048) if ecfp_params["folded"] else 2048,
        includeChirality=False,
        useBondTypes=True,
    )

    if ecfp_params["folded"]:
        _ = fpgen.GetCountFingerprint(
            mol,
            additionalOutput=bits_info,
            customAtomInvariants=custom_atom_invariants,
            customBondInvariants=custom_bond_invariants,
        )
    else:
        _ = fpgen.GetSparseCountFingerprint(
            mol,
            additionalOutput=bits_info,
            customAtomInvariants=custom_atom_invariants,
            customBondInvariants=custom_bond_invariants,
        )

    atom_to_bits_raw = bits_info.GetAtomToBits()
    atoms_morgan_bits = {
        i: tuple(atom_to_bits_raw[i]) for i in range(len(atom_to_bits_raw))
    }
    return atoms_morgan_bits
