import math
import numpy as np
from collections import Counter
from rdkit import Chem

from morganrxn.core.paths import REACTION_RULES_DIR


# =================================================================================================
# Reaction Rules class.
# =================================================================================================


class ReactionRules:
    def __init__(self, database_name, ecfp_params):
        self.template_reaction = []
        self.ecfp_reaction = []
        self.ecfp_reaction_center = []
        self.smi_sub = []
        self.nb_prod = []
        self.score = []
        self.reaction_monocomp_id = []
        self.reaction_id = []
        self.database_name = database_name
        self.ecfp_params = ecfp_params

    # ---------------------------------------------------------
    # Add one rule
    # ---------------------------------------------------------
    def add(
        self,
        rule,
        ecfp_reaction,
        ecfp_reaction_center,
        smi_sub,
        nb_prod,
        reaction_monocomp_id,
        reaction_id,
    ):
        self.template_reaction.append(rule)
        self.ecfp_reaction.append(tuple(ecfp_reaction))
        self.ecfp_reaction_center.append(tuple(ecfp_reaction_center))
        self.smi_sub.append(smi_sub)
        self.nb_prod.append(nb_prod)
        self.reaction_monocomp_id.append(reaction_monocomp_id)
        self.reaction_id.append(reaction_id)

    # ---------------------------------------------------------
    # Length
    # ---------------------------------------------------------
    def __len__(self):
        return len(self.template_reaction)

    # ---------------------------------------------------------
    # Score
    # ---------------------------------------------------------
    def compute_score(self):
        counts = Counter(self.ecfp_reaction)
        n_rules = len(self.ecfp_reaction)
        n_templates = len(counts)

        alpha = 1.0
        denom = n_rules + alpha * n_templates

        nll_scores = []
        for tpl in self.ecfp_reaction:
            c = counts[tpl]
            p = (c + alpha) / denom
            nll_scores.append(-math.log(p))

        self.score = nll_scores

    # ---------------------------------------------------------
    # Drop duplicates
    # ---------------------------------------------------------
    def drop_duplicates(self, verbose=False):
        """
        Drop duplicate reaction rules while keeping track of all
        reaction_monocomp_id and reaction_id values associated with duplicates.

        Duplicates are defined by:
            - template_reaction
            - ecfp_reaction
            - ecfp_reaction_center
            - nb_prod

        For duplicated entries, reaction_monocomp_id and reaction_id are merged
        as unique "|" separated strings.
        """
        key_to_index = {}
        keep_indices = []

        merged_reaction_monocomp_ids = []
        merged_reaction_ids = []

        for i in range(len(self.template_reaction)):
            key = (
                self.template_reaction[i],
                self.ecfp_reaction[i],
                self.ecfp_reaction_center[i],
                self.nb_prod[i],
            )

            reaction_monocomp_id = self.reaction_monocomp_id[i]
            reaction_id = self.reaction_id[i]

            if key not in key_to_index:
                key_to_index[key] = len(keep_indices)
                keep_indices.append(i)

                merged_reaction_monocomp_ids.append([])
                merged_reaction_ids.append([])

            kept_pos = key_to_index[key]

            if reaction_monocomp_id is not None and reaction_monocomp_id not in merged_reaction_monocomp_ids[kept_pos]:
                merged_reaction_monocomp_ids[kept_pos].append(reaction_monocomp_id)

            if reaction_id is not None and reaction_id not in merged_reaction_ids[kept_pos]:
                merged_reaction_ids[kept_pos].append(reaction_id)

        len_before = len(self.template_reaction)

        self.template_reaction = [self.template_reaction[i] for i in keep_indices]
        self.ecfp_reaction = [self.ecfp_reaction[i] for i in keep_indices]
        self.ecfp_reaction_center = [self.ecfp_reaction_center[i] for i in keep_indices]
        self.smi_sub = [self.smi_sub[i] for i in keep_indices]
        self.nb_prod = [self.nb_prod[i] for i in keep_indices]

        self.reaction_monocomp_id = [
            "|".join(map(str, ids)) for ids in merged_reaction_monocomp_ids
        ]
        self.reaction_id = [
            "|".join(map(str, ids)) for ids in merged_reaction_ids
        ]

        if len(self.score) == len_before:
            self.score = [self.score[i] for i in keep_indices]

        if verbose:
            print(f"Removed {len_before - len(self)} duplicates")
            print(f"Remaining rules: {len(self)}")

    # ---------------------------------------------------------
    # Drop zero ECFPs
    # ---------------------------------------------------------
    def drop_zero_ECFPs(self, verbose=False):
        keep_indices = []

        for i in range(len(self.ecfp_reaction)):
            ecfp = self.ecfp_reaction[i]
            if len([comp for comp in ecfp if comp == 0]) != len(ecfp):
                keep_indices.append(i)

        len_before = len(self.template_reaction)

        self.template_reaction = [self.template_reaction[i] for i in keep_indices]
        self.ecfp_reaction = [self.ecfp_reaction[i] for i in keep_indices]
        self.ecfp_reaction_center = [self.ecfp_reaction_center[i] for i in keep_indices]
        self.smi_sub = [self.smi_sub[i] for i in keep_indices]
        self.nb_prod = [self.nb_prod[i] for i in keep_indices]
        self.reaction_monocomp_id = [self.reaction_monocomp_id[i] for i in keep_indices]
        self.reaction_id = [self.reaction_id[i] for i in keep_indices]

        if len(self.score) == len_before:
            self.score = [self.score[i] for i in keep_indices]

        if verbose:
            print(f"Removed {len_before - len(self)} zero ECFPs")
            print(f"Remaining rules: {len(self)}")


    # ---------------------------------------------------------
    # Filter by smi_sub atom count
    # ---------------------------------------------------------
    def filter_by_smi_sub_atoms(self, min_atoms: int = 3, verbose: bool = True):
        """
        Keep only rules where smi_sub has at least `min_atoms` heavy atoms.

        Invalid SMILES are removed.

        This modifies the ReactionRules object in place and returns self.
        """
        keep_indices = []

        for i, smi in enumerate(self.smi_sub):
            mol = Chem.MolFromSmiles(smi)

            # Drop invalid SMILES
            if mol is None:
                continue

            # Heavy atoms only; implicit H atoms are not counted
            if mol.GetNumAtoms() >= min_atoms:
                keep_indices.append(i)

        len_before = len(self)

        self.template_reaction = [
            self.template_reaction[i] for i in keep_indices
        ]
        self.ecfp_reaction = [
            self.ecfp_reaction[i] for i in keep_indices
        ]
        self.ecfp_reaction_center = [
            self.ecfp_reaction_center[i] for i in keep_indices
        ]
        self.smi_sub = [
            self.smi_sub[i] for i in keep_indices
        ]
        self.nb_prod = [
            self.nb_prod[i] for i in keep_indices
        ]
        self.reaction_monocomp_id = [
            self.reaction_monocomp_id[i] for i in keep_indices
        ]
        self.reaction_id = [
            self.reaction_id[i] for i in keep_indices
        ]

        if len(self.score) == len_before:
            self.score = [
                self.score[i] for i in keep_indices
            ]

        if verbose:
            print(
                f"Removed {len_before - len(self)} rules "
                f"with smi_sub < {min_atoms} atoms"
            )
            print(f"Remaining rules: {len(self)}")

        return self

    # ---------------------------------------------------------
    # Internal: ECFP folder name
    # ---------------------------------------------------------
    def _ecfp_stem(self):
        r = int(self.ecfp_params["radius"])
        fp = int(self.ecfp_params["fpSize"])
        folded = bool(self.ecfp_params.get("folded", False))
        custom = bool(self.ecfp_params.get("custom", False))

        suffix = "folded" if folded else "unfolded"
        suffix2 = "custom" if custom else "uncustom"

        return f"ecfp_r{r}_fp{fp}_{suffix}_{suffix2}"

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    def save(self):
        stem = self._ecfp_stem()

        save_dir = REACTION_RULES_DIR / self.database_name / stem
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "rules.npz"

        np.savez_compressed(
            save_path,
            template_reaction=np.array(self.template_reaction, dtype=object),
            ecfp_reaction=np.array(self.ecfp_reaction, dtype=np.int32),
            ecfp_reaction_center=np.array(self.ecfp_reaction_center, dtype=np.int32),
            smi_sub=np.array(self.smi_sub, dtype=object),
            nb_prod=np.array(self.nb_prod, dtype=np.int32),
            score=np.array(self.score, dtype=np.float32),
            reaction_monocomp_id=np.array(self.reaction_monocomp_id, dtype=object),
            reaction_id=np.array(self.reaction_id, dtype=object),
            ecfp_params=self.ecfp_params,
        )

        print(f"Saved {len(self)} rules to:")
        print(save_path)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------
    @classmethod
    def load(cls, database_name, ecfp_params):
        obj = cls(database_name, ecfp_params=ecfp_params)

        r = int(ecfp_params["radius"])
        fp = int(ecfp_params["fpSize"])
        folded = bool(ecfp_params.get("folded", False))
        custom = bool(ecfp_params.get("custom", False))

        suffix = "folded" if folded else "unfolded"
        suffix2 = "custom" if custom else "uncustom"

        stem = f"ecfp_r{r}_fp{fp}_{suffix}_{suffix2}"
        load_path = REACTION_RULES_DIR / database_name / stem / "rules.npz"

        data = np.load(load_path, allow_pickle=True)

        obj.template_reaction = data["template_reaction"].tolist()
        obj.ecfp_reaction = [tuple(x) for x in data["ecfp_reaction"]]
        obj.ecfp_reaction_center = [tuple(x) for x in data["ecfp_reaction_center"]]
        obj.smi_sub = data["smi_sub"].tolist()
        obj.nb_prod = data["nb_prod"].tolist()
        obj.score = data["score"].tolist()

        # Backward-compatible loading for old .npz files
        if "reaction_monocomp_id" in data.files:
            obj.reaction_monocomp_id = data["reaction_monocomp_id"].tolist()
        else:
            obj.reaction_monocomp_id = [None] * len(obj)

        if "reaction_id" in data.files:
            obj.reaction_id = data["reaction_id"].tolist()
        else:
            obj.reaction_id = [None] * len(obj)

        print(f"Loaded {len(obj)} rules from:")
        print(load_path)

        return obj

    # ---------------------------------------------------------
    # Chunks
    # ---------------------------------------------------------
    def save_chunk(self, chunk_id: int):
        """
        Save the CURRENT BUFFER as chunk_{chunk_id}.npz.
        """
        if len(self) == 0:
            print(f"[save_chunk] buffer empty -> skip (chunk_id={chunk_id})")
            return

        stem = self._ecfp_stem()
        base_dir = REACTION_RULES_DIR / self.database_name / stem
        chunk_dir = base_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_path = chunk_dir / f"chunk_{chunk_id:06d}.npz"

        np.savez_compressed(
            chunk_path,
            template_reaction=np.array(self.template_reaction, dtype=object),
            ecfp_reaction=np.array(self.ecfp_reaction, dtype=np.int32),
            ecfp_reaction_center=np.array(self.ecfp_reaction_center, dtype=np.int32),
            smi_sub=np.array(self.smi_sub, dtype=object),
            nb_prod=np.array(self.nb_prod, dtype=np.int32),
            reaction_monocomp_id=np.array(self.reaction_monocomp_id, dtype=object),
            reaction_id=np.array(self.reaction_id, dtype=object),
        )

        print(f"Saved chunk {chunk_id} ({len(self)} rules) -> {chunk_path}")

    # ---------------------------------------------------------
    # Clear in-memory buffer
    # ---------------------------------------------------------
    def clear(self):
        """
        Clear all in-memory rule buffers.
        Does NOT touch anything on disk.
        """
        self.template_reaction.clear()
        self.ecfp_reaction.clear()
        self.ecfp_reaction_center.clear()
        self.smi_sub.clear()
        self.nb_prod.clear()
        self.score.clear()
        self.reaction_monocomp_id.clear()
        self.reaction_id.clear()

    # ---------------------------------------------------------
    # Combine all chunks into one ReactionRules object
    # ---------------------------------------------------------
    @classmethod
    def combine_chunks(cls, database_name, ecfp_params):
        obj = cls(database_name, ecfp_params=ecfp_params)

        stem = obj._ecfp_stem()
        chunk_dir = REACTION_RULES_DIR / database_name / stem / "chunks"

        if not chunk_dir.exists():
            raise FileNotFoundError(f"No chunk directory: {chunk_dir}")

        chunk_paths = sorted(chunk_dir.glob("chunk_*.npz"))

        if not chunk_paths:
            raise FileNotFoundError(f"No chunk files found in: {chunk_dir}")

        for p in chunk_paths:
            d = np.load(p, allow_pickle=True)

            n = len(d["nb_prod"])

            obj.template_reaction.extend(d["template_reaction"].tolist())
            obj.ecfp_reaction.extend([tuple(x) for x in d["ecfp_reaction"]])
            obj.ecfp_reaction_center.extend([tuple(x) for x in d["ecfp_reaction_center"]])
            obj.smi_sub.extend(d["smi_sub"].tolist())
            obj.nb_prod.extend(d["nb_prod"].tolist())

            # Backward-compatible loading for old chunks
            if "reaction_monocomp_id" in d.files:
                obj.reaction_monocomp_id.extend(d["reaction_monocomp_id"].tolist())
            else:
                obj.reaction_monocomp_id.extend([None] * n)

            if "reaction_id" in d.files:
                obj.reaction_id.extend(d["reaction_id"].tolist())
            else:
                obj.reaction_id.extend([None] * n)

            print(f"Loaded {p.name} ({n} rules)")

        return obj

    # ---------------------------------------------------------
    # Merge two ReactionRules
    # ---------------------------------------------------------
    @classmethod
    def merge(
        cls,
        a: "ReactionRules",
        b: "ReactionRules",
        database_name: str,
        verbose: bool = False,
    ) -> "ReactionRules":
        """
        Merge two ReactionRules objects into a NEW ReactionRules.
        """
        if a.ecfp_params != b.ecfp_params:
            raise ValueError(
                "ecfp_params mismatch. "
                f"a.ecfp_params={a.ecfp_params} vs b.ecfp_params={b.ecfp_params}"
            )

        ecfp_params = a.ecfp_params

        obj = cls(database_name, ecfp_params=ecfp_params)

        obj.template_reaction = list(a.template_reaction) + list(b.template_reaction)
        obj.ecfp_reaction = list(a.ecfp_reaction) + list(b.ecfp_reaction)
        obj.ecfp_reaction_center = list(a.ecfp_reaction_center) + list(b.ecfp_reaction_center)
        obj.smi_sub = list(a.smi_sub) + list(b.smi_sub)
        obj.nb_prod = list(a.nb_prod) + list(b.nb_prod)
        obj.score = list(a.score) + list(b.score)
        obj.reaction_monocomp_id = list(a.reaction_monocomp_id) + list(b.reaction_monocomp_id)
        obj.reaction_id = list(a.reaction_id) + list(b.reaction_id)

        if verbose:
            print(
                f"[merge] a={len(a)} rules, b={len(b)} rules -> "
                f"merged={len(obj)} rules (pre-filter)"
            )

        obj.drop_zero_ECFPs(verbose=verbose)
        obj.drop_duplicates(verbose=verbose)

        if verbose:
            print(f"[merge] final merged size: {len(obj)}")

        return obj