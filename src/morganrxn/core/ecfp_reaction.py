from collections import Counter
from rdkit import Chem

from morganrxn.core.molecule_utils import get_mol_ecfp, get_mol_ecfp_atom_to_bits
from morganrxn.core.vector_utils import l1_plus_l2, l1_minus_l2, bits_to_vector


def compute_ecfp_prod_minus_sub(smi_prod, smi_sub, ecfp_params):
    mol_sub = Chem.MolFromSmiles(smi_sub)
    mol_prod = Chem.MolFromSmiles(smi_prod)
    if mol_sub is None or mol_prod is None:
        return None
    morgan_sub = get_mol_ecfp(mol_sub, ecfp_params=ecfp_params)
    morgan_prod = get_mol_ecfp(mol_prod, ecfp_params=ecfp_params)
    return l1_minus_l2(morgan_prod, morgan_sub, folded=ecfp_params["folded"])


def get_ecfp_reaction_center(mol_sub, reaction_center_indices, ecfp_params):
    atoms_morgan_bits = get_mol_ecfp_atom_to_bits(mol_sub, ecfp_params=ecfp_params)

    ecfp_bits = []
    visited_indices = set()
    current_shell = set(reaction_center_indices)
    for r in range(ecfp_params["radius"] + 1):
        next_shell = set()
        for atom_idx in current_shell:
            if atom_idx in visited_indices:
                continue  # Skip already visited atoms
            if atom_idx in atoms_morgan_bits:
                # Collect bits corresponding to this radius
                ecfp_bits.extend(list(atoms_morgan_bits[atom_idx])[r:])
            visited_indices.add(atom_idx)
            # Queue neighbors for next shell
            atom = mol_sub.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx not in visited_indices:
                    next_shell.add(n_idx)
        current_shell = next_shell

    # Convert to vector or sorted list of negative bits
    if ecfp_params["folded"]:
        vec = bits_to_vector(ecfp_bits, size=ecfp_params["fpSize"])
        return [-1 * x for x in vec]
    else:
        return sorted([-1 * x for x in ecfp_bits])


def ecfp_reaction_center_applicable(ecfp_reaction_center, ecfp_smi, ecfp_params):
    if ecfp_params["folded"]:
        return all(x >= 0 for x in l1_plus_l2(ecfp_reaction_center, ecfp_smi, folded=ecfp_params["folded"]))
    else:
        ecfp_reaction_center = [-1 * x for x in ecfp_reaction_center]
        counter_ecfp_smi = Counter(ecfp_smi)
        counter_ecfp_reaction = Counter(ecfp_reaction_center)
        return all(counter_ecfp_smi[element] >= count for element, count in counter_ecfp_reaction.items())
