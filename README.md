# morganrxn

**Representing chemical and enzymatic reactions in fingerprint space for applicability filtering and classification.**

- **Paper**: Meyer, Duigou, Gricourt and Faulon, *Journal of Cheminformatics* (2026),
  <https://doi.org/10.1186/s13321-026-01322-2>
- **Code used for the paper** (release `v0.2.0`): <https://doi.org/10.5281/zenodo.23127885>
- **Data** (raw inputs, mapped reactions, reaction rules): <https://doi.org/10.5281/zenodo.23127919>

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

- **Reaction ECFP**: the net difference vector, describing *what a reaction does*:

  ```text
  ECFP(r) = Σⱼ ECFP(Pⱼ) − Σᵢ ECFP(Sᵢ)
  ```

  Positive coordinates are generated environments; negative coordinates are consumed ones.

- **Reaction-center ECFP**: a non-positive vector encoding *what a reaction needs*, i.e. the
  local environments around the reaction center that must be present in a substrate.
  A reaction is ECFP-applicable to a molecule vector `v` when `ECFP_rc(r) + v ≥ 0`.

- **ECFP-compatible template**: a template whose reaction-center radius is at least `2h`,
  so that graph-level application induces a context-independent fingerprint translation.

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/brsynth/morganrxn.git
cd morganrxn
git checkout v0.2.0   # exact version used for the paper (optional)
pip install -e .
pip install rdkit numpy scipy scikit-learn pandas matplotlib openpyxl
```

`pyproject.toml` does not declare the dependencies, hence the second `pip install`.

### Atom mapping (stage 2 only)

Stage 2 maps reactions with [RXNMapper_v2](https://github.com/yvsgrndjn/RXNMapper_v2)
(transformers / PyTorch). It is not needed to reproduce the paper from the Zenodo data,
which already contains the mapped reactions. To run it, install RXNMapper_v2 from GitHub,
preferably in a separate environment. The paper used commit `a57117d`:

```bash
pip install git+https://github.com/yvsgrndjn/RXNMapper_v2.git@a57117ded230a1fd0a3fe261fac41b843aec2220
```

(Environment of the paper run: Python 3.11, transformers 4.46.3, PyTorch 2.12.1 CPU, RDKit 2026.03.3.)

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
│   ├── mapping.py            #   atom-mapping utilities (RXNMapper_v2)
│   ├── cli_utils.py          #   helpers shared by the command-line scripts
│   ├── paths.py              #   central project paths
│   └── visualization.py      #   RDKit-based plotting (notebooks)
├── data_processing/          # Data pipeline (stages 1–3)
│   ├── uspto.py              #   stage 1: sanitize USPTO reactions
│   ├── metanetx.py           #   stage 1: sanitize MetaNetX reactions
│   ├── map_reactions.py      #   stage 2: atom mapping (RXNMapper_v2)
│   └── create_reactionrules.py  # stage 3: build ReactionRules for each radius
└── paper_results/            # Analyses & benchmarks
    ├── data_statistics.py    #   Table 1: representation counts & cross-dataset overlap
    ├── t_sne.py              #   Fig. 3: t-SNE projection of reaction vectors
    ├── applicability_accuracy.py  # Table 2: reaction-center filter vs. graph-level application
    ├── graph_vs_vector_timing.py  # Table 2 & S8: per-target runtime, graph vs. vector route
    ├── merge_graph_vs_vector.py   #   aggregate the runtime CSVs
    ├── uspto_prediction.py   #   Table 3: USPTO reaction-class prediction
    └── metanetx_ec_prediction.py  # Table 4: MetaNetX EC-number prediction

cluster/                      # SLURM: submit_all.sh chains the full pipeline (job.sbatch runs each step)
data/                         # Datasets (git-ignored, see below)
```

## Data

Download `data.zip` from Zenodo (<https://doi.org/10.5281/zenodo.23127919>) and unzip it at
the repository root. It creates `data/` with the raw inputs, the outputs of stages 1–2
(sanitized and atom-mapped reactions) and of stage 3 (reaction rules for every radius and
fingerprint size), so any stage can be rerun or skipped:

```
data/
├── uspto/
│   ├── dataSetB.csv                     # raw USPTO-50k
│   └── processed/                       # uspto_reactions.tsv (stage 1), uspto_mapped.tsv (stage 2)
├── metanetx/
│   ├── chem_prop.tsv, reac_prop.tsv     # raw MetaNetX v4.5 tables
│   └── processed/                       # metanetx_reactions.tsv (stage 1), metanetx_mapped.tsv (stage 2)
└── reaction_rules/                      # stage 3
    ├── uspto/ecfp_r{0..5}_fp{512,1024,2048}_folded_uncustom/rules.npz
    └── metanetx/ecfp_r{0..5}_fp{512,1024,2048}_folded_uncustom/rules.npz
```

The `data/` and `results/` directories are git-ignored.

## Pipeline

Reaction rules are built in three stages, then analyzed. The commands below are the ones
used to produce the paper results, runnable directly once the package is installed.

### Stage 1: sanitize reactions (default parameters)

Canonicalizes reaction SMILES, drops agents, removes atom maps and stereochemistry:

```bash
# USPTO (L2R only)
python src/morganrxn/data_processing/uspto.py

# MetaNetX (both directions, keeps EC annotations)
python src/morganrxn/data_processing/metanetx.py
```

### Stage 2: atom mapping (default parameters)

Applies RXNMapper_v2 atom mapping with its default model (`alberta_uspto_2800k`, layer 10,
head 3; batch size 32); unmappable reactions are dropped:

```bash
python src/morganrxn/data_processing/map_reactions.py --data uspto
python src/morganrxn/data_processing/map_reactions.py --data metanetx
```

### Stage 3: build reaction rules

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

`START_STAGE=3 bash cluster/submit_all.sh` starts from the atom-mapped reactions instead
(this is how the paper results were produced: stages 1–2 were run locally); it also expects
a `data/mapping_provenance.txt` file describing how the mapped files were obtained. Logs
go to `logs/<RUN_ID>/`, results to `results/`.

The commands below are the per-step equivalents, for `--fp-size 1024`. Add `--fp-size 512`
or `--fp-size 2048` for the ablation (Additional file 1, Tables S1–S6). Adjust `--n-jobs`
to the available cores.

**Table 1: representation counts & MetaNetX/USPTO overlap**

```bash
python src/morganrxn/paper_results/data_statistics.py \
    --radii 0,1,2,3,4,5 \
    --metanetx-database-name metanetx \
    --uspto-database-name uspto \
    --output-dir results/data_statistics/fp1024 \
    --output-name reaction_vector_overlap_by_radius.csv
```

**Fig. 3: t-SNE of reaction & reaction-center ECFPs**

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

**Table 2: reaction-center filter vs. graph-level applicability** (1000 target molecules
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

**Table 2 running times & Table S8: graph-level vs. vector route** on the Table 2 targets
(per-target CSV rows, aggregated with `merge_graph_vs_vector.py`)

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

**Table 3: USPTO reaction-class prediction** (4 classifiers)

```bash
python src/morganrxn/paper_results/uspto_prediction.py \
    --database-name uspto \
    --radii 0,1,2,3,4,5 \
    --models logistic_regression,random_forest,gradient_boosting,mlp \
    --output-dir results/uspto_prediction/fp1024 \
    --summary-output results/uspto_prediction/fp1024/metrics_all_radii.csv \
    --save-meta
```

**Table 4: MetaNetX EC-number prediction** (extra-trees, EC levels 1–4)

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

> Meyer P., Duigou T., Gricourt G., Faulon J.-L. Representing chemical and enzymatic
> reactions in fingerprint space for applicability filtering and classification.
> *Journal of Cheminformatics* (2026). <https://doi.org/10.1186/s13321-026-01322-2>

```bibtex
@article{meyer2026morganrxn,
  author  = {Meyer, Philippe and Duigou, Thomas and Gricourt, Guillaume and Faulon, Jean-Loup},
  title   = {Representing chemical and enzymatic reactions in fingerprint space for applicability filtering and classification},
  journal = {Journal of Cheminformatics},
  year    = {2026},
  doi     = {10.1186/s13321-026-01322-2}
}
```

## Funding

Supported by a French government grant managed by the Agence Nationale de la Recherche
under the France 2030 program (ANR-22-PEBB-0008), with computing resources from the
Core Cluster of the Institut Français de Bioinformatique (IFB, ANR-11-INBS-0013).

## License

Creative Commons Attribution 4.0 International (CC BY 4.0), as for the archived release on
Zenodo. See [`LICENSE`](LICENSE).
