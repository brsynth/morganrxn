# morganrxn

**Representing chemical and enzymatic reactions in fingerprint space for applicability filtering and classification.**

The `morganrxn` package represents chemical reactions as signed transformations between
counted molecular Extended-Connectivity Fingerprint (ECFP) vectors, and studies the link
between graph-level reaction templates and vector-space reaction operators.

For an ECFP-compatible reaction template, graph-level reaction application induces a
*constant* displacement in counted ECFP space, so graph transformations become affine
translations and their composition becomes vector addition. Reaction-center ECFPs encode
the local environments a reaction requires and provide a fast, coordinate-wise (O(d))
necessary condition for applicability, used as a prefilter before graph-level validation.

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
    ├── graph_vs_vector_timing.py  # per-target runtime, graph-level vs. vector route
    ├── merge_graph_vs_vector.py   # aggregate the runtime CSVs
    ├── uspto_prediction.py   #   USPTO reaction-class prediction
    └── metanetx_ec_prediction.py  # MetaNetX EC-number prediction

cluster/                      # SLURM: submit_all.sh chains the full pipeline
data/                         # Datasets (git-ignored)
├── uspto/                    #   raw + processed USPTO
├── metanetx/                 #   raw + processed MetaNetX
└── reaction_rules/           #   generated ReactionRules, per database, radius & fp size
```

## Data layout

Raw inputs are placed under `data/`, and generated reaction rules are written to
`data/reaction_rules/<database>/ecfp_r<h>_fp<d>_folded_uncustom/rules.npz`:

```
data/
├── uspto/
│   ├── dataSetB.csv                     # raw USPTO-50k
│   └── processed/                       # stage 1 & 2 outputs
├── metanetx/
│   ├── chem_prop.tsv, reac_prop.tsv     # raw MetaNetX v4.5 tables
│   └── processed/                       # stage 1 & 2 outputs
└── reaction_rules/
    ├── uspto/ecfp_r{0..5}_fp{512,1024,2048}_folded_uncustom/rules.npz
    └── metanetx/ecfp_r{0..5}_fp{512,1024,2048}_folded_uncustom/rules.npz
```

The `data/` and `results/` directories are git-ignored. Datasets are
available on Zenodo: <https://doi.org/10.5281/zenodo.21509287>.

## Pipeline

Reaction rules are built in three stages, then analyzed. The commands below are the ones
used to produce the paper results, runnable directly once the package is installed.

### Stage 1 — sanitize reactions (default parameters)

Canonicalizes reaction SMILES, drops agents, removes atom maps and stereochemistry:

```bash
# USPTO (L2R only)
python src/morganrxn/data_processing/uspto.py

# MetaNetX (both directions, keeps EC annotations)
python src/morganrxn/data_processing/metanetx.py
```

### Stage 2 — atom mapping (default parameters)

Applies RXNMapper_v2 atom mapping with its default model (`alberta_uspto_2800k`, layer 10,
head 3; batch size 32); unmappable reactions are dropped:

```bash
python src/morganrxn/data_processing/map_reactions.py --data uspto
python src/morganrxn/data_processing/map_reactions.py --data metanetx
```

### Stage 3 — build reaction rules

Deduplicates to monosubstrate reactions and computes, for each radius `h ∈ {0..5}`, the
ECFP-compatible template, reaction ECFP, and reaction-center ECFP.

```bash
python src/morganrxn/data_processing/create_reactionrules.py --data uspto    --radii 0,1,2,3,4,5
python src/morganrxn/data_processing/create_reactionrules.py --data metanetx --radii 0,1,2,3,4,5
```

The fingerprint-size ablation uses the same command with `--fp-size 512` or `--fp-size 2048`
(default 1024); rules are stored per size.

## Reproducing the paper results

On a SLURM cluster, `cluster/submit_all.sh` runs the whole pipeline (stages 1–3, then every
table, figure and ablation below, for fingerprint sizes 512, 1024 and 2048) as jobs chained
by dependencies. Run it from the root of a clean clone containing the raw inputs:

```bash
bash cluster/submit_all.sh
```

The commands below are the per-step equivalents, for `--fp-size 1024` (add `--fp-size 512`
or `--fp-size 2048` for the ablation). Adjust `--n-jobs` to the available cores.

**Table 1 — representation counts & MetaNetX/USPTO overlap**

```bash
python src/morganrxn/paper_results/data_statistics.py \
    --radii 0,1,2,3,4,5 \
    --metanetx-database-name metanetx \
    --uspto-database-name uspto \
    --output-dir results/data_statistics/fp1024 \
    --output-name reaction_vector_overlap_by_radius.csv
```

**Figure — t-SNE of reaction & reaction-center ECFPs**

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

**Table 2 — reaction-center filter vs. graph-level applicability** (1000 target molecules
of at most 500 Da per database, deterministic sample; run once per database)

```bash
python src/morganrxn/paper_results/applicability_accuracy.py \
    --radii 0,1,2,3,4,5 \
    --n-samples 1000 \
    --max-mol-wt 500 \
    --applicability-modes reaction_center \
    --benchmark-dataset metanetx=metanetx \
    --paired-rules metanetx=metanetx \
    --out-xlsx results/table2_mw500/fp1024/applicability_metanetx.xlsx
```

**Runtime — graph-level vs. vector route** on the Table 2 targets (per-target CSV rows,
aggregated with `merge_graph_vs_vector.py`)

```bash
python src/morganrxn/paper_results/graph_vs_vector_timing.py \
    --database-name metanetx \
    --radii 0,1,2,3,4,5 \
    --n-samples 1000 \
    --max-mol-wt 500
python src/morganrxn/paper_results/merge_graph_vs_vector.py \
    --inputs results/graph_vs_vector_mw500/*.csv \
    --out-csv results/graph_vs_vector_mw500/summary.csv
```

**Table 3 — USPTO reaction-class prediction** (4 classifiers)

```bash
python src/morganrxn/paper_results/uspto_prediction.py \
    --database-name uspto \
    --radii 0,1,2,3,4,5 \
    --models logistic_regression,random_forest,gradient_boosting,mlp \
    --output-dir results/uspto_prediction/fp1024 \
    --summary-output results/uspto_prediction/fp1024/metrics_all_radii.csv \
    --save-meta
```

**Table 4 — MetaNetX EC-number prediction** (extra-trees, EC levels 1–4)

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
    --output-dir results/metanetx_ec_prediction/fp1024 \
    --summary-output results/metanetx_ec_prediction/fp1024/metrics_all_ec_levels_all_radii.csv \
    --save-meta
```

Pass `-h` / `--help` to any script for the full set of options.

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
