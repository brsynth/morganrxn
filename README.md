# morganrxn

**Representing chemical and enzymatic reactions in fingerprint space for applicability filtering and classification.**

`morganrxn` represents chemical reactions as signed transformations between counted
molecular Extended-Connectivity Fingerprint (ECFP) vectors, and studies the link
between graph-level reaction templates and vector-space reaction operators.

For an ECFP-compatible reaction template, graph-level reaction application induces a
*constant* displacement in counted ECFP space, so graph transformations become affine
translations and their composition becomes vector addition. Reaction-center ECFPs encode
the local environments a reaction requires and provide a fast, coordinate-wise ($O(d)$)
necessary condition for applicability, used as a prefilter before graph-level validation.

This repository accompanies the manuscript *"Representing Chemical and Enzymatic Reactions
in Fingerprint Space for Applicability Filtering and Classification"* (Meyer, Duigou,
Gricourt, Faulon).

## Key concepts

For a reaction `r : S₁ + … + Sₘ → P₁ + … + Pₙ` at ECFP radius `h`:

- **Reaction ECFP** — the net difference vector, describing *what a reaction does*:

  ```text
  ECFP(r) = Σⱼ ECFP(Pⱼ) − Σᵢ ECFP(Sᵢ)
  ```

  Positive coordinates are generated environments; negative coordinates are consumed ones.

- **Reaction-center ECFP** — a non-positive vector encoding *what a reaction needs*: the
  local environments around the reaction center that must be present in a substrate.
  A reaction is ECFP-applicable to a molecule vector `v` when `ECFP_rc(r) + v ≥ 0`.

- **ECFP-compatible template** — a template whose reaction-center radius is at least `2h`,
  so that graph-level application induces a context-independent fingerprint translation.

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/brsynth/morganrxn.git
cd morganrxn
pip install -e .
```

Runtime dependencies (install if not pulled in automatically):

```bash
pip install rdkit numpy scipy scikit-learn pandas matplotlib openpyxl
```

Atom mapping (stage 2 of the pipeline) additionally relies on **RXNMapper_v2**
(transformers / PyTorch), vendored under [`external/`](external/).

## Repository structure

```
src/morganrxn/
├── core/                     # Library
│   ├── molecule_utils.py     #   molecule sanitization, ECFP computation
│   ├── ecfp_reaction.py      #   reaction & reaction-center ECFPs
│   ├── centre.py             #   reaction-center detection, atom-map completion
│   ├── templating.py         #   ECFP-compatible template extraction (SMARTS)
│   ├── reaction_rules.py     #   ReactionRules container (save/load .npz)
│   ├── reaction_utils.py     #   reaction parsing / deduplication helpers
│   ├── vector_utils.py       #   counted-fingerprint vector arithmetic
│   ├── mapping.py            #   atom-mapping utilities
│   ├── paths.py              #   central project paths
│   └── visualization.py      #   RDKit-based plotting (notebooks)
├── data_processing/          # Data pipeline (stages 1–3)
│   ├── uspto.py              #   stage 1: sanitize USPTO reactions
│   ├── metanetx.py           #   stage 1: sanitize MetaNetX reactions
│   ├── map_reactions.py      #   stage 2: atom mapping (RXNMapper_v2)
│   └── create_reactionrules.py  # stage 3: build ReactionRules for each radius
└── paper_results/            # Analyses & benchmarks
    ├── data_statistics.py    #   representation counts & cross-dataset overlap
    ├── t_sne.py              #   t-SNE projection of reaction vectors
    ├── applicability_accuracy.py  # reaction-center filter vs. graph-level application
    ├── uspto_prediction.py   #   USPTO reaction-class prediction
    └── metanetx_ec_prediction.py  # MetaNetX EC-number prediction

data/                         # Datasets (git-ignored)
├── uspto/                    #   raw + processed USPTO
├── metanetx/                 #   raw + processed MetaNetX
└── reaction_rules/           #   generated ReactionRules, per database & radius
```

## Data layout

Raw inputs are placed under `data/`, and generated reaction rules are written to
`data/reaction_rules/<database>/ecfp_r<h>_fp<d>_folded_uncustom/rules.npz`:

```
data/
├── uspto/
│   ├── datasetB.csv                     # raw USPTO-50k
│   └── processed/                       # stage 1 & 2 outputs
├── metanetx/
│   ├── chem_prop.tsv, reac_prop.tsv     # raw MetaNetX v4.5 tables
│   └── processed/                       # stage 1 & 2 outputs
└── reaction_rules/
    ├── uspto/ecfp_r{0..5}_fp1024_folded_uncustom/rules.npz
    └── metanetx/ecfp_r{0..5}_fp1024_folded_uncustom/rules.npz
```

The `data/`, `results/`, and `slurms/` directories are git-ignored. Datasets are
available on Zenodo: <https://doi.org/10.5281/zenodo.21509287>.

## Pipeline

Reaction rules are built in three stages, then analyzed. Scripts are runnable as modules
(`python -m morganrxn.…`).

### Stage 1 — sanitize reactions

Canonicalizes reaction SMILES, drops agents, removes atom maps and stereochemistry.

```bash
# USPTO (L2R only)
python -m morganrxn.data_processing.uspto

# MetaNetX (both directions, keeps EC annotations)
python -m morganrxn.data_processing.metanetx
```

### Stage 2 — atom mapping

Applies RXNMapper_v2 atom mapping; unmappable reactions are dropped.

```bash
python -m morganrxn.data_processing.map_reactions --data uspto
python -m morganrxn.data_processing.map_reactions --data metanetx
```

### Stage 3 — build reaction rules

Deduplicates to monosubstrate reactions and computes, for each radius `h ∈ {0..5}`, the
ECFP-compatible template, reaction ECFP, and reaction-center ECFP.

```bash
python -m morganrxn.data_processing.create_reactionrules --data uspto   --radii 0,1,2,3,4,5
python -m morganrxn.data_processing.create_reactionrules --data metanetx --radii 0,1,2,3,4,5
```

## Reproducing the paper results

Each script defaults to radii 0–5 and reads the generated `ReactionRules`.

```bash
# Table 1 — representation counts & MetaNetX/USPTO overlap
python -m morganrxn.paper_results.data_statistics

# Figure — t-SNE of reaction & reaction-center ECFPs
python -m morganrxn.paper_results.t_sne --datasets metanetx uspto

# Table 2 — reaction-center filter vs. graph-level applicability
python -m morganrxn.paper_results.applicability_accuracy

# Table 3 — USPTO reaction-class prediction (4 classifiers)
python -m morganrxn.paper_results.uspto_prediction

# Table 4 — MetaNetX EC-number prediction (extra-trees, EC levels 1–4)
python -m morganrxn.paper_results.metanetx_ec_prediction
```

Pass `-h` / `--help` to any script for the full set of options (radii, fingerprint size,
classifiers, output paths, etc.). SLURM launch scripts for a cluster are provided under
`slurms/`.

## Citation

If you use this code, please cite:

> Meyer P., Duigou T., Gricourt G., Faulon J.-L. *Representing Chemical and Enzymatic
> Reactions in Fingerprint Space for Applicability Filtering and Classification.*

(Full citation and DOI will be added upon publication.)

## Funding

Supported by a French government grant managed by the Agence Nationale de la Recherche
under the France 2030 program (ANR-22-PEBB-0008), with computing resources from the
Institut Français de Bioinformatique (IFB, ANR-11-INBS-0013).

## License

See the repository for license information.
