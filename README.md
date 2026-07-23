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

Reaction rules are built in three stages, then analyzed. The exact commands below are the
ones used to produce the paper results on a SLURM cluster (see [`slurms/`](slurms/)); each
`slurms/run_*.slurm` wraps one of them with a conda environment and
`PYTHONPATH=src`. The commands can be run directly once the package is installed.

### Stage 1 — sanitize reactions (default parameters)

Canonicalizes reaction SMILES, drops agents, removes atom maps and stereochemistry. No
dedicated SLURM script — run with defaults:

```bash
# USPTO (L2R only)
python src/morganrxn/data_processing/uspto.py

# MetaNetX (both directions, keeps EC annotations)
python src/morganrxn/data_processing/metanetx.py
```

### Stage 2 — atom mapping (default parameters)

Applies RXNMapper_v2 atom mapping (batch size 32); unmappable reactions are dropped. No
dedicated SLURM script — run with defaults:

```bash
python src/morganrxn/data_processing/map_reactions.py --data uspto
python src/morganrxn/data_processing/map_reactions.py --data metanetx
```

### Stage 3 — build reaction rules

Deduplicates to monosubstrate reactions and computes, for each radius `h ∈ {0..5}`, the
ECFP-compatible template, reaction ECFP, and reaction-center ECFP.
(`slurms/run_create_reactionrules_{uspto,metanetx}.slurm`)

```bash
python src/morganrxn/data_processing/create_reactionrules.py --data uspto    --radii 0,1,2,3,4,5
python src/morganrxn/data_processing/create_reactionrules.py --data metanetx --radii 0,1,2,3,4,5
```

## Reproducing the paper results

The commands below use the exact parameters of the SLURM scripts. On the cluster,
`--n-jobs` is set to `$SLURM_CPUS_PER_TASK` (8 for t-SNE, 16 for EC prediction).

**Table 1 — representation counts & MetaNetX/USPTO overlap** (`slurms/run_data_statistics.slurm`)

```bash
python src/morganrxn/paper_results/data_statistics.py \
    --radii 0,1,2,3,4,5 \
    --metanetx-database-name metanetx \
    --uspto-database-name uspto \
    --output-dir results/data_statistics \
    --output-name reaction_vector_overlap_by_radius.csv
```

**Figure — t-SNE of reaction & reaction-center ECFPs** (`slurms/run_t_sne.slurm`)

```bash
python src/morganrxn/paper_results/t_sne.py \
    --datasets metanetx uspto \
    --radii 0 1 2 3 4 5 \
    --encoding raw \
    --metric cosine \
    --n-jobs 8 \
    --output-dir results/t_sne \
    --format pdf \
    --save-coords
```

**Table 2 — reaction-center filter vs. graph-level applicability** (`slurms/run_applicability_accuracy.slurm`)

```bash
python src/morganrxn/paper_results/applicability_accuracy.py \
    --radii 0,1,2,3,4,5 \
    --n-samples 1000 \
    --benchmark-dataset metanetx=metanetx \
    --benchmark-dataset uspto=uspto \
    --paired-rules metanetx=metanetx \
    --paired-rules uspto=uspto \
    --out-xlsx results/one_step_accuracy/applicability_accuracy_morganrxn_formats.xlsx
```

**Table 3 — USPTO reaction-class prediction** (4 classifiers) (`slurms/run_uspto_prediction.slurm`)

```bash
python src/morganrxn/paper_results/uspto_prediction.py \
    --database-name uspto \
    --radii 0,1,2,3,4,5 \
    --models logistic_regression,random_forest,gradient_boosting,mlp \
    --output-dir results/uspto_prediction \
    --summary-output results/uspto_prediction/metrics_all_radii.csv \
    --save-meta
```

**Table 4 — MetaNetX EC-number prediction** (extra-trees, EC levels 1–4) (`slurms/run_metanetx_ec_prediction.slurm`)

```bash
python src/morganrxn/paper_results/metanetx_ec_prediction.py \
    --ec-levels 1,2,3,4 \
    --radii 0,1,2,3,4,5 \
    --min-label-count 5 \
    --max-labels 300 \
    --models sgd,et \
    --feature-sets reaction_ecfp,reaction_center_ecfp,both \
    --sample-mode unique_rules \
    --n-jobs 16 \
    --output-dir results/metanetx_ec_prediction \
    --summary-output results/metanetx_ec_prediction/metrics_all_ec_levels_all_radii.csv \
    --save-meta
```

Pass `-h` / `--help` to any script for the full set of options. Ready-to-submit SLURM
scripts for all of the above (plus per-dataset applicability variants) are in
[`slurms/`](slurms/).

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
