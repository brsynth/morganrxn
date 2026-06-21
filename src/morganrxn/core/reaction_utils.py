import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from rulesmith import map_reaction, Reaction, make_templates
from rulesmith.templating.centre import find_flat_changes
from rulesmith.templating.mapping import MapLxR

from morganrxn.core.ecfp_reaction import compute_ecfp_prod_minus_sub, get_ecfp_reaction_center
from morganrxn.core.molecule_utils import sanitize_smiles
from morganrxn.core.vector_utils import vector_to_bits
from morganrxn.core.visualization import plot_reaction



# =================================================================================================
# Sanitize reaction.
# =================================================================================================


def sanitize_reaction(reaction_rule):
    subs, prods = reaction_rule.split(">>")
    subs = sanitize_smiles(subs)
    prods = sanitize_smiles(prods)
    reaction_rule = subs + ">>" + prods
    return reaction_rule


# =================================================================================================
# Basic reaction manipulations.
# =================================================================================================


def invert_reaction(reaction_rule):
    subs, prods = reaction_rule.split(">>")
    reaction_rule_inverted = prods + ">>" + subs
    return reaction_rule_inverted


# =================================================================================================
# Apply reaction functions.
# =================================================================================================


def apply_reaction(rxn, smi):
    if isinstance(rxn, str):
        rxn = rdChemReactions.ReactionFromSmarts(rxn)
    mol = Chem.MolFromSmiles(smi)
    rxn.Initialize()
    products = rxn.RunReactants((mol,))
    product_sets = []
    for prod_tuple in products:
        # build "A.B.C" style SMILES string for each product set
        smiles_set = []
        for m in prod_tuple:
            if m is None:
                continue
            try:
                Chem.SanitizeMol(m)
                smi = sanitize_smiles(Chem.MolToSmiles(m))
                smiles_set.append(smi)
            except Exception:
                continue
        if smiles_set:  # avoid empty sets
            product_sets.append(".".join(sorted(smiles_set)))
    # deduplicate sets
    product_sets = list(set(product_sets))
    return product_sets


# =================================================================================================
# SMARTS syntax manipulation functions.
# =================================================================================================


def add_missing_mappings_both_sides(reaction_rule: str) -> str:
    reaction_rule_sub, reaction_rule_prod = reaction_rule.split(">>")
    # Collect all existing AAM numbers across both sides
    used_numbers = set(map(int, re.findall(r':(\d+)', reaction_rule)))
    next_map_num = max(used_numbers, default=0) + 1
    # 1) Pre-pass: bracketize any bare atoms (not already inside [])
    #    Includes common organic + aromatic symbols and a few multi-letter ones.
    #    Add more elements if you need them.
    ELEM_PATTERN = (
        r'(?<!\[)'                # not immediately after '[' (i.e., not already bracketed)
        r'(Cl|Br|Si|As|Se|Na|Li|Mg|Al|Ca|Fe|Zn|Cu|Mn|Ag|K|Ti|Cr|Co|Ni|Mo|Sn|Pb|Pt|Au|'
        r'B|C|N|O|P|S|F|I|c|n|o|p|s|\*)'
        r'(?![a-z])'              # don’t consume the next lowercase (avoid e.g. 'Si' + 'm' in 'Sim')
    )
    def bracketize_bare_atoms(smarts_str: str) -> str:
        return re.sub(ELEM_PATTERN, r'[\1]', smarts_str)
    def add_mappings(smarts_str: str) -> str:
        nonlocal next_map_num, used_numbers
        # bracketize first so every atom token is of the form [ ... ]
        smarts_str = bracketize_bare_atoms(smarts_str)
        # track numbers used on this side to avoid duplicates
        current_used = set(map(int, re.findall(r':(\d+)', smarts_str)))
        def replacer(match):
            nonlocal next_map_num
            atom_token = match.group(0)  # e.g. "[CH3:2]" or "[N]" or "[cH]"
            if ':' not in atom_token:    # missing mapping -> add one just before closing ']'
                while next_map_num in current_used or next_map_num in used_numbers:
                    next_map_num += 1
                updated = atom_token[:-1] + f":{next_map_num}]"
                used_numbers.add(next_map_num)
                current_used.add(next_map_num)
                next_map_num += 1
                return updated
            return atom_token
        # now add maps to any bracketed atom missing one
        return re.sub(r'\[[^\]]+?\]', replacer, smarts_str)
    reaction_rule_sub = add_mappings(reaction_rule_sub)
    reaction_rule_prod = add_mappings(reaction_rule_prod)
    return reaction_rule_sub + ">>" + reaction_rule_prod


def extract_aams_in_reaction_rule(reaction_rule):
    substrate, product = reaction_rule.split(">>")
    aams_substrate = re.findall(r':(\d+)', substrate)
    aams_product = re.findall(r':(\d+)', product)
    return sorted(set(int(aam) for aam in aams_substrate)), sorted(set(int(aam) for aam in aams_product))


# =================================================================================================
# Matter loss.
# =================================================================================================


def find_matter_loss(reaction_rule):
    reaction_rule = add_missing_mappings_both_sides(reaction_rule)
    lhs_aams, rhs_aams = extract_aams_in_reaction_rule(reaction_rule)
    lhs_set, rhs_set = set(lhs_aams), set(rhs_aams)
    missing = lhs_set - rhs_set
    return missing


def has_matter_loss(reaction_rule):
    return len(find_matter_loss(reaction_rule)) > 0


def _expected_max_valence(atomic_num: int, charge: int, is_aromatic: bool) -> int:
    """
    Heuristic 'max expected valence' used to decide if a query atom could
    accept more bonds than those explicitly specified in the template.
    Adjusts a bit for charge and common organics.
    """
    pt = Chem.GetPeriodicTable()
    # RDKit valence list (possible valences). Pick a sensible upper bound.
    vals = list(pt.GetValenceList(atomic_num))
    if not vals:
        return 4  # fallback for weird cases
    # Common-sense tweaks for frequent elements
    if atomic_num == 1:
        return 1
    if atomic_num == 6:  # carbon
        # Aromatic C usually degree 3 max (three bonds counting 1.5 won't map well to integers)
        return 4
    if atomic_num == 7:  # nitrogen
        # Neutral N often 3; quaternary ammonium is 4
        if charge > 0:
            return 4
        return 3
    if atomic_num == 8:  # oxygen
        return 2
    if atomic_num == 16:  # sulfur (keep generous)
        return 6
    if atomic_num == 15:  # phosphorus
        return 5
    # Fallback: take the max listed valence, but keep it reasonable
    vmax = max(vals)
    # Mild adjustment: for positive charge, allow one more if it seems plausible
    if charge > 0 and vmax < 4:
        vmax = min(4, vmax + 1)
    return int(vmax)

def _bond_order_sum(atom: Chem.Atom) -> float:
    """Sum the numeric bond orders (aromatic=1.5) for bonds present in the template."""
    s = 0.0
    for b in atom.GetBonds():
        s += b.GetBondTypeAsDouble()
    return s

def _explicit_H_in_query(atom: Chem.Atom) -> int:
    """
    For SMARTS/query atoms, GetTotalNumHs() returns specified H count if constrained (e.g., [NH2], [CH3]).
    If unspecified, it tends to be 0 (which is fine for our 'could accept more' check).
    """
    try:
        return int(atom.GetTotalNumHs())
    except Exception:
        return 0

def reactant_atoms_not_fully_specified(reaction_smarts: str):
    """
    Inputs:
        reaction_smarts: RDKit reaction SMARTS with atom maps on both sides.
    Returns:
        dict with:
          - 'by_reactant': {reactant_index: [mapped_atom_numbers_not_fully_specified]}
          - 'flat': sorted list of all mapped atoms (tuples (reactant_index, mapnum)) not fully specified
          - 'details': per atom diagnostic with used_valence and expected_max_valence
    """
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    if rxn is None:
        raise ValueError("Could not parse reaction SMARTS.")
    reactants = rxn.GetReactants()
    out_by_reactant = {}
    flat = []
    details = []  # (react_idx, mapnum, symbol, used, vmax, reason)
    for r_idx, tmpl in enumerate(reactants):
        not_full = []
        for atom in tmpl.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap <= 0:
                # Skip unmapped atoms (you said all are mapped, but just in case)
                continue
            sym = atom.GetSymbol()
            charge = atom.GetFormalCharge()
            is_arom = atom.GetIsAromatic()
            # what the template explicitly connects:
            used = _bond_order_sum(atom) + _explicit_H_in_query(atom)
            vmax = _expected_max_valence(atom.GetAtomicNum(), charge, is_arom)
            # If used < vmax, the template leaves room for more attachment(s)
            not_fully_specified = used < vmax - 1e-6  # small tolerance

            if not_fully_specified:
                not_full.append(amap)
                flat.append((r_idx, amap))
                details.append(
                    dict(reactant_index=r_idx, atom_map=amap, symbol=sym,
                         used_valence=used, expected_max_valence=vmax,
                         neighbors_in_template=[n.GetAtomMapNum() for n in atom.GetNeighbors()])
                )
        out_by_reactant[r_idx] = sorted(not_full)
    flat.sort(key=lambda x: (x[0], x[1]))
    return {"by_reactant": out_by_reactant, "flat": flat, "details": details}


def has_open_matter_loss(reaction_rule, verbose=False):
    reaction_rule = add_missing_mappings_both_sides(reaction_rule)
    atoms_disappearing = find_matter_loss(reaction_rule)
    if len(atoms_disappearing) == 0:
        return False
    atoms_not_fully_specified = [atom[1] for atom in reactant_atoms_not_fully_specified(reaction_rule)["flat"]]
    if verbose:
        print("atoms_disappearing", atoms_disappearing)
        print("atoms_not_fully_specified", atoms_not_fully_specified)
    intersection = [atom for atom in atoms_disappearing if atom in atoms_not_fully_specified]
    return len(intersection) > 0


# =================================================================================================
# Reaction deduplication.
# =================================================================================================


def deduplicate_reaction(reaction_rule):
    reaction_rule_deduplicated = []
    reactants_str, products_str = reaction_rule.split(">>", 1)
    reactants = [r.strip() for r in reactants_str.split(".") if r.strip()]
    products = [p.strip() for p in products_str.split(".") if p.strip()]
    # Extract atom mapping numbers for each product once
    prod_mapnums = {}
    for prod in products:
        mol = Chem.MolFromSmiles(prod)
        if mol is None:
            continue
        prod_mapnums[prod] = {a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum() > 0}
    for reac in reactants:
        mol_reac = Chem.MolFromSmiles(reac)
        if mol_reac is None:
            continue
        reac_maps = {a.GetAtomMapNum() for a in mol_reac.GetAtoms() if a.GetAtomMapNum() > 0}
        # keep only products that share at least one mapping number
        filtered_prods = [
            prod for prod, maps in prod_mapnums.items()
            if reac_maps & maps
        ]
        if filtered_prods:
            reaction_rule_deduplicated.append(f"{reac}>>{'.'.join(filtered_prods)}")
    return reaction_rule_deduplicated


# =================================================================================================
# Reaction validation.
# =================================================================================================


def valid_reaction(reaction_rule):
    if ">>" not in reaction_rule:
        return False
    subs, prods = reaction_rule.split(">>")
    return subs != "" or prods != "" or subs != prods


# =================================================================================================
# Cofactors cleaning.
# =================================================================================================


def _inchi_key_prefix(smiles: str) -> str | None:
    """Return the first block of the InChIKey for a SMILES (ignoring atom maps)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Remove atom-map numbers manually
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return ik.split('-')[0] if ik else None

def make_cofactor_set(df_cofactors) -> set[str]:
    return {str(x).strip() for x in df_cofactors["INCHIKEY_PREFIX"].dropna()}

def _clean_side(side: str, cofactor_keys: set[str]) -> str:
    if not side:
        return ""
    tokens = [t for t in side.split('.') if t]
    kept = []
    for smi in tokens:
        ik_pref = _inchi_key_prefix(smi)
        if ik_pref is None or ik_pref not in cofactor_keys:
            kept.append(smi)
    return '.'.join(kept)

def clean_rule_drop_cofactors(rule: str, df_cofactors=None) -> str:
    """
    Drop cofactors from the left side of a reaction rule.

    If df_cofactors is None, no cofactor filtering is applied.
    """
    if ">>" not in rule:
        raise ValueError("Rule must contain '>>'.")

    if df_cofactors is None:
        return rule

    cofactor_keys = make_cofactor_set(df_cofactors)

    left, right = rule.split(">>", 1)
    left_clean = _clean_side(left.strip(), cofactor_keys)
    right_clean = _clean_side(right.strip(), set())

    if left_clean == "" or right_clean == "":
        return ""

    return f"{left_clean}>>{right_clean}"


# =================================================================================================
# Reaction validation.
# =================================================================================================


def suppress_agent(reaction_rule):
    r, a, p = reaction_rule.split(">")
    return f"{r}>>{p}"


# =================================================================================================
# Reaction validation.
# =================================================================================================


def prepare_and_clean_reaction_pipeline(
    reaction_rule,
    df_cofactors=None,
    direction="forward",
    verbose=False,
):
    if not isinstance(reaction_rule, str) or not reaction_rule.strip():
        return set()

    reaction_rule = reaction_rule.strip()

    try:
        reaction_rule = suppress_agent(reaction_rule)
        reaction_rule = sanitize_reaction(reaction_rule)
    except Exception:
        return set()

    if direction == "forward":
        reaction_rules = {reaction_rule}
    elif direction == "backward":
        reaction_rules = {invert_reaction(reaction_rule)}
    elif direction == "both":
        reaction_rules = {reaction_rule, invert_reaction(reaction_rule)}
    else:
        raise ValueError("direction must be one of: 'forward', 'backward', 'both'")

    if verbose:
        print(f"Nb reaction direction {len(reaction_rules)}")
    if verbose == 2:
        [plot_reaction(reaction) for reaction in reaction_rules]

    reaction_rules_mapped = set()

    for reaction_rule in reaction_rules:
        try:
            reaction_rule_mapped, _ = map_reaction(reaction_rule)

            if not reaction_rule_mapped or ">>" not in reaction_rule_mapped:
                continue

            reaction_rule_mapped = add_missing_mappings_both_sides(reaction_rule_mapped)

            if valid_reaction(reaction_rule_mapped):
                reaction_rules_mapped.add(reaction_rule_mapped)

        except Exception:
            continue

    if verbose:
        print(f"Nb reaction mapped {len(reaction_rules_mapped)}")
    if verbose == 2:
        [plot_reaction(reaction) for reaction in reaction_rules_mapped]

    reaction_rules_deduplicated = set()

    for reaction_rule in reaction_rules_mapped:
        try:
            deduplicated_reactions = set(deduplicate_reaction(reaction_rule))
            reaction_rules_deduplicated |= deduplicated_reactions
        except Exception:
            continue

    if verbose:
        print(f"Nb reaction deduplicated {len(reaction_rules_deduplicated)}")
    if verbose == 2:
        [plot_reaction(reaction) for reaction in reaction_rules_deduplicated]

    reaction_rules_cleaned = set()

    for reaction_rule in reaction_rules_deduplicated:
        try:
            reaction_rule_cleaned = clean_rule_drop_cofactors(
                reaction_rule,
                df_cofactors=df_cofactors,
            )

            if reaction_rule_cleaned and valid_reaction(reaction_rule_cleaned):
                reaction_rules_cleaned.add(reaction_rule_cleaned)

        except Exception:
            continue

    if verbose:
        print(f"Nb reaction cleaned {len(reaction_rules_cleaned)}")
    if verbose == 2:
        [plot_reaction(reaction) for reaction in reaction_rules_cleaned]

    reaction_rules_no_open_matter = set()

    for reaction_rule in reaction_rules_cleaned:
        try:
            if not has_matter_loss(reaction_rule):
                reaction_rules_no_open_matter.add(reaction_rule)
            elif not has_open_matter_loss(reaction_rule):
                reaction_rules_no_open_matter.add(reaction_rule)
            else:
                if verbose:
                    print("OPEN MATTER", reaction_rule)

        except Exception:
            continue

    if verbose:
        print(f"Nb reaction no open matter {len(reaction_rules_no_open_matter)}")
    if verbose == 2:
        [plot_reaction(reaction) for reaction in reaction_rules_no_open_matter]

    return reaction_rules_no_open_matter


# =================================================================================================
# Test pathway.
# =================================================================================================


def test_pathway(smi, pathway_to_test):
    # work with sanitized fragment SMILES (supports single or dot-mixture input)
    frags = [smi]
    for step_idx, rxn_smart in enumerate(pathway_to_test):
        progressed = False
        # try to apply this reaction to any current fragment
        for i, frag_smi in enumerate(list(frags)):
            product_sets = apply_reaction(rxn_smart, frag_smi)  # returns ["A.B", "C.D.E", ...]
            if not product_sets:
                continue
            # pick one deterministic set (first after sort to avoid set-order randomness)
            picked_set = sorted(product_sets)[0]
            picked_frags = [p for p in picked_set.split(".") if p]
            # replace fragment i by all products of the picked set
            frags = frags[:i] + picked_frags + frags[i+1:]
            progressed = True
            break
        if not progressed:
            return None, step_idx  # this step couldn't be applied to any fragment
    # return a dot-joined canonical (sorted) SMILES of all fragments
    return ".".join(sorted(frags)), None


# =================================================================================================
# Process a reaction.
# =================================================================================================


def process_a_reaction(reaction_smiles, sp_min, ecfp_params, template_radius=1, verbose=False):
    rxn = Reaction.from_smiles(reaction_smiles)
    lefts, rights = rxn.to_mols()

    ##########
    # Find reaction centers
    ##########
    lxr = MapLxR(lefts, rights)
    # Get first reaction center only
    changes = find_flat_changes(lefts, rights)
    first_center = changes[0]
    if verbose:
        print("reaction center in substrate:", first_center)
    # Map left indices -> atom-map numbers
    left_mol_i = 0
    changed_maps = [lxr.get_map_from_left_index(left_mol_i, idx) for idx in first_center]
    if verbose:
        print("reaction center in mapping:", changed_maps)

    ##########
    # Template
    ##########
    templates = make_templates(lefts, rights, smarts_params=sp_min, radius=template_radius)
    template = templates[0]['template']
    if verbose:
        print("len(templates)", len(templates))
        print("template", template)
        plot_reaction(template)

    ##########
    # ECFP reaction center
    ##########
    ecfp_reaction_center = get_ecfp_reaction_center(lefts[0], first_center, ecfp_params=ecfp_params)
    if verbose:
        if ecfp_params["folded"]:
            print("ecfp_reaction_center", vector_to_bits(ecfp_reaction_center))
        else:
            print("ecfp_reaction_center", ecfp_reaction_center)

    ##########
    # ECFP reaction
    ##########
    smi_subs, smi_prods = reaction_smiles.split(">>")
    ecfp_reaction = compute_ecfp_prod_minus_sub(smi_prods, smi_subs, ecfp_params=ecfp_params)
    if verbose:
        if ecfp_params["folded"]:
            print("ecfp_reaction", vector_to_bits(ecfp_reaction))
        else:
            print("ecfp_reaction", ecfp_reaction)

    return template, ecfp_reaction_center, ecfp_reaction


# =================================================================================================
# One step.
# =================================================================================================


def one_step(smi_ecfp, ecfp_reactions, ecfp_reaction_centers):
    mask = np.all((smi_ecfp[None, :] + ecfp_reaction_centers) >= 0, axis=1)
    applicable_idxs = np.where(mask)[0]
    child_vecs = smi_ecfp + ecfp_reactions[applicable_idxs]
    child_vecs_unique, idx_unique = np.unique(child_vecs, axis=0, return_index=True)
    rxn_idxs_unique = applicable_idxs[idx_unique]
    return child_vecs_unique, rxn_idxs_unique
