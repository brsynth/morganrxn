from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List

from morganrxn.core.centre import complete_reaction_mapping

# =================================================================================================
# Defaults.
# =================================================================================================

DEFAULT_MODEL_NAME = "original"  # original RXNMapper model (head 5, layer 10)
DEFAULT_BATCH_SIZE = 32


# =================================================================================================
# Model loading.
# =================================================================================================


@lru_cache(maxsize=None)
def _get_batched_mapper(model_name: str, batch_size: int) -> "BatchedMapper":
    # rxnmapper is imported lazily so that the pure-text helpers in this module
    # (e.g. add_missing_mappings_both_sides) can be used without rxnmapper/torch
    # installed -- rule creation imports those but never maps reactions here.
    from rxnmapper import BatchedMapper

    # rxnmapper's BatchedMapper does not take a model name: its defaults
    # (head=5, layer=10, model_type="albert") are exactly the "original" model
    # referenced by DEFAULT_MODEL_NAME. model_name is kept in the public API for
    # forward compatibility but is not forwarded here.
    return BatchedMapper(batch_size=batch_size)


# =================================================================================================
# Completion of missing atom mappings.
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
        # Only bracketize atoms that are OUTSIDE existing [...] brackets. Splitting
        # on bracketed groups first prevents corrupting multi-letter bracketed
        # elements whose second letter is also an aromatic symbol, e.g. [As:2]
        # (the 's') or [Co:1] (the 'o'), which a naive global re.sub would mangle
        # into [A[s:2]] / [C[o:1]].
        parts = re.split(r'(\[[^\]]*\])', smarts_str)
        for i, part in enumerate(parts):
            if not part.startswith('['):
                parts[i] = re.sub(ELEM_PATTERN, r'[\1]', part)
        return ''.join(parts)
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


def _complete_mapped_rxn(mapped_rxn: str) -> str:
    if not mapped_rxn or ">>" not in mapped_rxn:
        return mapped_rxn
    try:
        # RDKit-based completion (preserves implicit Hs); the regex
        # add_missing_mappings_both_sides would drop them when bracketising bare atoms.
        return complete_reaction_mapping(mapped_rxn)
    except Exception:
        return mapped_rxn


# =================================================================================================
# Atom mapping with RXNMapper_v2.
# =================================================================================================


def map_reactions_with_rxnmapper(
    reactions: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    detailed: bool = False,
    complete_mapping: bool = True,
) -> List[Dict[str, Any]]:
    """
    Atom-map a list of reaction SMILES with RXNMapper_v2's attention-guided model.

    Reactions are processed in chunks of `batch_size`; a reaction that fails to
    map (invalid SMILES, too many tokens, ...) is isolated and comes back as an
    empty dict instead of failing the whole batch.

    RXNMapper sometimes leaves atoms unmapped, typically ones that disappear
    between substrates and products. If `complete_mapping` is True (default),
    `complete_reaction_mapping` assigns map numbers to those leftover atoms
    (preserving the molecule exactly, unlike a text-level bracketisation).

    Returns one dict per input reaction (same order as `reactions`), each with:
        - "mapped_rxn": the atom-mapped reaction SMILES
        - "confidence": the model's confidence in the mapping
    """
    mapper = _get_batched_mapper(model_name, batch_size)
    results = list(mapper.map_reactions_with_info(reactions, detailed=detailed))
    if complete_mapping:
        for result in results:
            if result.get("mapped_rxn"):
                result["mapped_rxn"] = _complete_mapped_rxn(result["mapped_rxn"])
    return results


def map_reaction_with_rxnmapper(
    reaction: str,
    model_name: str = DEFAULT_MODEL_NAME,
    detailed: bool = False,
    complete_mapping: bool = True,
) -> Dict[str, Any]:
    """Atom-map a single reaction SMILES. See `map_reactions_with_rxnmapper`."""
    return map_reactions_with_rxnmapper(
        [reaction],
        model_name=model_name,
        batch_size=1,
        detailed=detailed,
        complete_mapping=complete_mapping,
    )[0]
