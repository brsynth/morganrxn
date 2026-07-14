import re
from collections import Counter
from itertools import permutations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from morganrxn.core.ecfp_reaction import compute_ecfp_prod_minus_sub, get_ecfp_reaction_center
from morganrxn.core.mapping import add_missing_mappings_both_sides
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


def _canonical_without_maps(component: str):
    """Canonical SMILES of a single component with atom-map numbers stripped (or None)."""
    mol = Chem.MolFromSmiles(component)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def remove_constant_components(reaction_rule):
    """
    Drop the components that stay constant across a (mapped) reaction, i.e. the
    spectator molecules present unchanged on both sides. Atom maps are preserved on
    the components that are kept.

    A component is a spectator when the same molecule (canonical SMILES, atom maps
    ignored) appears on both the left and the right; each such match is cancelled
    once from each side (so stoichiometry is respected). Components appearing only
    on the right (compounds genuinely produced by the reaction) are always kept, as
    are left-only reactants.
    """
    subs_smi, prods_smi = reaction_rule.split(">>")
    sub_components = [c for c in subs_smi.split(".") if c]
    prod_components = [c for c in prods_smi.split(".") if c]
    sub_keys = [_canonical_without_maps(c) for c in sub_components]
    prod_keys = [_canonical_without_maps(c) for c in prod_components]

    # Spectators = molecules present on both sides (multiset intersection).
    to_cancel = (
        Counter(k for k in sub_keys if k is not None)
        & Counter(k for k in prod_keys if k is not None)
    )

    def keep_side(components, keys):
        budget = dict(to_cancel)
        kept = []
        for component, key in zip(components, keys):
            if key is not None and budget.get(key, 0) > 0:
                budget[key] -= 1  # cancel one spectator occurrence
                continue
            kept.append(component)
        return kept

    kept_subs = keep_side(sub_components, sub_keys)
    kept_prods = keep_side(prod_components, prod_keys)
    return ".".join(kept_subs) + ">>" + ".".join(kept_prods)


# =================================================================================================
# Apply reaction functions.
# =================================================================================================


def _ring_requirement(props):
    """
    Interpret the ring-membership primitives of a single SMARTS atom's property
    string. SMARTS primitives are joined by the logical operators &, ; and , so
    we inspect each token independently. Returns True if the atom must be in a
    ring, False if it must not be, or None if ring membership is unconstrained.

    Note: ``Chem.MolToSmarts`` always emits the ``&``-joined form (e.g.
    ``C&X3&R``), so ring detection must not treat a leading ``&`` as part of the
    ``!R`` negation.
    """
    for tok in re.split(r'[&;,]', props):
        tok = tok.strip()
        if tok == 'R' or re.fullmatch(r'R[1-9][0-9]*', tok):
            return True
        if tok == '!R' or tok == 'R0':
            return False
    return None


def _get_ring_constraints(rxn):
    """
    For each atom-map number in the reactant templates, extract R/!R constraint.
    Returns: {map_num: (reactant_idx, template_atom_idx, required_in_ring: bool)}
    """
    constraints = {}
    for r_idx in range(rxn.GetNumReactantTemplates()):
        tmpl = rxn.GetReactantTemplate(r_idx)
        smarts = Chem.MolToSmarts(tmpl)
        for match in re.finditer(r'\[([^\]]+):(\d+)\]', smarts):
            props, map_num = match.group(1), int(match.group(2))
            required = _ring_requirement(props)
            if required is None:
                continue
            for atom in tmpl.GetAtoms():
                if atom.GetAtomMapNum() == map_num:
                    constraints[map_num] = (r_idx, atom.GetIdx(), required)
                    break
    return constraints


def _get_product_query_atoms(rxn):
    """
    Map each product-template atom-map number to its SMARTS query atom.
    Returns: {map_num: query_atom}.

    A product query atom encodes every primitive the template constrains for that
    atom (element, X, D, H, valence, charge, ring membership, aromaticity) -- the
    same information an ECFP atom identifier carries. ``RunReactants`` ignores
    these query features when it builds products, so a generated product atom can
    end up with a different degree, H count, valence, charge, ring membership or
    aromaticity than its template declares. Keeping the query atoms lets the
    caller check that consistency after the fact.
    """
    query_atoms = {}
    for p_idx in range(rxn.GetNumProductTemplates()):
        tmpl = rxn.GetProductTemplate(p_idx)
        for atom in tmpl.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num:
                query_atoms[map_num] = atom
    return query_atoms


def _product_matches_template(prod_mol, query_atoms):
    """
    Check that every mapped atom of a generated product molecule satisfies the
    full SMARTS query of its product-template atom. Product atoms keep the
    template map number in the ``old_mapno`` property after ``RunReactants``, so
    each is compared against the query atom sharing that map number. Atoms without
    a mapping (created by the reaction but not described by the template) are
    unconstrained. Returns False as soon as one mapped atom diverges from what its
    template declares.
    """
    if not query_atoms:
        return True
    for atom in prod_mol.GetAtoms():
        if not atom.HasProp('old_mapno'):
            continue
        query_atom = query_atoms.get(int(atom.GetProp('old_mapno')))
        if query_atom is not None and not query_atom.Match(atom):
            return False
    return True


def _ring_counts_by_index(mol):
    """
    For each atom index in `mol`, count how many SSSR rings it belongs to, and how
    many of those rings are fully aromatic. Index-keyed counterpart of
    ``morganrxn.core.centre._ring_counts_by_map_num``.
    """
    ring_info = mol.GetRingInfo()
    counts = {atom.GetIdx(): [0, 0] for atom in mol.GetAtoms()}
    for atom_ring, bond_ring in zip(ring_info.AtomRings(), ring_info.BondRings()):
        is_aromatic_ring = all(mol.GetBondWithIdx(b).GetIsAromatic() for b in bond_ring)
        for atom_idx in atom_ring:
            counts[atom_idx][0] += 1
            if is_aromatic_ring:
                counts[atom_idx][1] += 1
    return {idx: tuple(v) for idx, v in counts.items()}


def _local_signature(atom, ring_counts, neighbor_identity):
    """
    Local signature of an atom: its own invariants (atomic number, formal charge,
    H count, valence, aromaticity, ring counts, degree) plus the sorted
    ``(bond order, neighbor identity)`` of each bond. Mirrors
    ``morganrxn.core.centre._atom_signature`` but identifies neighbors through a
    caller-supplied key (a substrate atom index shared across both reaction sides)
    instead of atom-map numbers.
    """
    neighbor_sig = tuple(sorted(
        (bond.GetBondTypeAsDouble(), neighbor_identity(bond.GetOtherAtom(atom)))
        for bond in atom.GetBonds()
    ))
    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetTotalValence(),
        atom.GetIsAromatic(),
        ring_counts[atom.GetIdx()],
        atom.GetDegree(),
        neighbor_sig,
    )


def _change_is_within_template(source_mol, prod_mols):
    """
    Return False if applying the reaction altered the local environment of a
    substrate atom that the template does not describe -- i.e. the change "leaked"
    beyond the template's own atoms (tautomerisation, ring or aromaticity
    reshuffle elsewhere) and would break the ECFP-additivity the rule assumes.

    Atoms are matched across the two sides by their substrate index: every product
    atom carries it in ``react_atom_idx``. A product atom that also carries
    ``old_mapno`` is one the template explicitly builds, so it may change freely;
    any preserved substrate atom without ``old_mapno`` whose signature differs
    between substrate and product is an unintended side effect. Atoms removed by
    the reaction are skipped: ``RunReactants`` only removes template atoms.
    """
    sub_rings = _ring_counts_by_index(source_mol)

    prod_by_src = {}  # substrate index -> (product atom, its owning mol's ring counts)
    for prod_mol in prod_mols:
        prod_rings = _ring_counts_by_index(prod_mol)
        for atom in prod_mol.GetAtoms():
            if atom.HasProp('react_atom_idx'):
                prod_by_src[int(atom.GetProp('react_atom_idx'))] = (atom, prod_rings)

    def sub_identity(a):
        return a.GetIdx()

    def prod_identity(a):
        # Product neighbors are identified by their substrate index; atoms newly
        # created by the reaction (no react_atom_idx) get a distinct key so that a
        # bond to such an atom registers as a change.
        return int(a.GetProp('react_atom_idx')) if a.HasProp('react_atom_idx') else ('new', a.GetIdx())

    for sub_atom in source_mol.GetAtoms():
        entry = prod_by_src.get(sub_atom.GetIdx())
        if entry is None:
            continue  # atom removed by the reaction (only template atoms are removed)
        prod_atom, prod_rings = entry
        if prod_atom.HasProp('old_mapno'):
            continue  # atom built by the template: allowed to change
        sub_sig = _local_signature(sub_atom, sub_rings, sub_identity)
        prod_sig = _local_signature(prod_atom, prod_rings, prod_identity)
        if sub_sig != prod_sig:
            return False
    return True


def _substrate_satisfies_ring_constraints(mol, rxn, constraints):
    """
    Check R/!R constraints on the substrate via substructure matching.
    Product atoms lose their atom map numbers after RunReactants, so we check here.
    Returns True if at least one match of the reactant template satisfies all constraints.
    """
    if not constraints:
        return True
    by_reactant: dict = {}
    for map_num, (r_idx, atom_idx, required) in constraints.items():
        by_reactant.setdefault(r_idx, []).append((atom_idx, required))

    for r_idx, atom_constraints in by_reactant.items():
        tmpl = rxn.GetReactantTemplate(r_idx)
        matches = mol.GetSubstructMatches(tmpl)
        if not matches:
            return False
        if any(
            all(mol.GetAtomWithIdx(match[tmpl_idx]).IsInRing() == required
                for tmpl_idx, required in atom_constraints)
            for match in matches
        ):
            return True
        return False
    return True


def apply_reaction(
    rxn,
    smi,
    filter_ring_consistency=False,
    check_product_consistency=True,
    check_change_within_template=True,
):
    if isinstance(rxn, str):
        rxn = rdChemReactions.ReactionFromSmarts(rxn)
    mol = Chem.MolFromSmiles(smi)
    rxn.Initialize()

    if filter_ring_consistency:
        constraints = _get_ring_constraints(rxn)
        if constraints and not _substrate_satisfies_ring_constraints(mol, rxn, constraints):
            return []

    product_query_atoms = _get_product_query_atoms(rxn) if check_product_consistency else {}

    # RunReactants needs exactly one molecule per reactant template. For a
    # multi-component template we split the substrate into its fragments and try
    # every assignment of fragments to templates (RDKit matches positionally).
    n_templates = rxn.GetNumReactantTemplates()
    if n_templates <= 1:
        reactant_tuples = [(mol,)]
    else:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) < n_templates:
            return []
        reactant_tuples = permutations(frags, n_templates)

    product_sets = []
    for reactants in reactant_tuples:
        # react_atom_idx on the products indexes into a single source molecule, so
        # the "change stayed within the template" check only applies when the
        # template has one reactant (the monosubstrate rules this pipeline builds).
        source_mol = reactants[0] if len(reactants) == 1 else None
        for prod_tuple in rxn.RunReactants(reactants):
            valid = True
            sanitized = []
            for m in prod_tuple:
                if m is None:
                    continue
                try:
                    Chem.SanitizeMol(m)
                except Exception:
                    valid = False
                    break
                if not _product_matches_template(m, product_query_atoms):
                    valid = False
                    break
                sanitized.append(m)

            if valid and check_change_within_template and source_mol is not None:
                if not _change_is_within_template(source_mol, sanitized):
                    valid = False

            if not valid:
                continue

            smiles_set = []
            for m in sanitized:
                try:
                    smiles_set.append(sanitize_smiles(Chem.MolToSmiles(m)))
                except Exception:
                    valid = False
                    break
            if valid and smiles_set:
                product_sets.append(".".join(sorted(smiles_set)))
    product_sets = list(set(product_sets))
    return product_sets


# =================================================================================================
# SMARTS syntax manipulation functions.
# =================================================================================================


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


def suppress_agent(reaction_rule):
    r, a, p = reaction_rule.split(">")
    return f"{r}>>{p}"


# =================================================================================================
# Process a reaction.
# =================================================================================================


def process_a_reaction(reaction_smiles, ecfp_params, template_radius=1, verbose=False):
    from morganrxn.core.centre import get_substrate_reaction_centre_atom_indices
    from morganrxn.core.templating import get_reaction_template

    # `reaction_smiles` is expected to be already atom-mapped: mapping is
    # performed once upstream (data_processing/map_reactions.py) and this
    # function does not map. The substrate atom indices and the template must be
    # computed on the exact same molecule, so we parse smi_subs directly and
    # rely on the (map-preserving) upstream mapping for the correspondence.
    smi_subs, smi_prods = reaction_smiles.split(">>")
    mol_sub = Chem.MolFromSmiles(smi_subs)

    ##########
    # Find reaction center (atom indices in the substrate)
    ##########
    first_center = get_substrate_reaction_centre_atom_indices(reaction_smiles)
    if verbose:
        print("reaction center in substrate:", first_center)

    ##########
    # Template
    ##########
    template = get_reaction_template(reaction_smiles, template_radius)
    if verbose:
        print("template", template)
        plot_reaction(template)

    ##########
    # ECFP reaction center
    ##########
    ecfp_reaction_center = get_ecfp_reaction_center(mol_sub, first_center, ecfp_params=ecfp_params)
    if verbose:
        if ecfp_params["folded"]:
            print("ecfp_reaction_center", vector_to_bits(ecfp_reaction_center))
        else:
            print("ecfp_reaction_center", ecfp_reaction_center)

    ##########
    # ECFP reaction
    ##########
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
