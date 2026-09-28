#!/bin/bash
# Submit the whole pipeline, from reaction sanitization to the paper tables, as
# SLURM jobs chained by dependencies (a failed job cancels everything downstream).
#
# Run from the root of a clean clone, with the raw inputs copied in:
#   data/uspto/dataSetB.csv
#   data/metanetx/chem_prop.tsv  data/metanetx/reac_prop.tsv
#   bash cluster/submit_all.sh
#
# START_STAGE=3 starts from the reaction rules, using atom-mapped reactions produced
# elsewhere (stages 1-2), copied in with their provenance file:
#   data/uspto/processed/uspto_mapped.tsv
#   data/metanetx/processed/metanetx_mapped.tsv
#   data/mapping_provenance.txt
#   START_STAGE=3 bash cluster/submit_all.sh
#
# Logs go to logs/<RUN_ID>/, results to results/, rules to data/reaction_rules/.

set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M)}"
LOG_DIR="logs/${RUN_ID}"
RADII="0,1,2,3,4,5"
FP_SIZES="512 1024 2048"
DATABASES="uspto metanetx"
MAX_MOL_WT=500          # Table 2 and timing targets
N_TARGETS=1000
TIMING_CHUNK=100        # timing targets per job (the graph route is ~1 min per target and radius)
EXCLUDE_NODES="cpu-node-49"
START_STAGE="${START_STAGE:-1}"

if [[ "$START_STAGE" == 1 ]]; then
    INPUTS=(data/uspto/dataSetB.csv data/metanetx/chem_prop.tsv data/metanetx/reac_prop.tsv)
elif [[ "$START_STAGE" == 3 ]]; then
    INPUTS=(data/uspto/dataSetB.csv data/metanetx/reac_prop.tsv
            data/uspto/processed/uspto_mapped.tsv data/metanetx/processed/metanetx_mapped.tsv
            data/mapping_provenance.txt)
else
    echo "START_STAGE must be 1 or 3." >&2
    exit 1
fi
for f in "${INPUTS[@]}"; do
    [[ -f "$f" ]] || { echo "Missing input: $f" >&2; exit 1; }
done
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Uncommitted changes in the clone: commit first so that the run is traceable." >&2
    exit 1
fi
if [[ -d "data/reaction_rules" || -d "results" ]]; then
    echo "data/reaction_rules or results already exists: use a fresh clone." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
git rev-parse HEAD > "$LOG_DIR/commit.txt"
if [[ "$START_STAGE" == 3 ]]; then
    cp data/mapping_provenance.txt "$LOG_DIR/"
    md5sum data/*/processed/*_mapped.tsv > "$LOG_DIR/mapped_inputs.md5"
fi

# submit NAME DEPS RESOURCES COMMAND...   (DEPS: colon-separated job ids, or "")
submit() {
    local name=$1 deps=$2 resources=$3
    shift 3
    local dep_opts=()
    if [[ -n "$deps" ]]; then
        dep_opts=(--dependency="afterok:${deps}" --kill-on-invalid-dep=yes)
    fi
    # shellcheck disable=SC2086
    sbatch --parsable -J "$name" --exclude="$EXCLUDE_NODES" \
        ${dep_opts[@]+"${dep_opts[@]}"} $resources \
        --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" \
        cluster/job.sbatch "$@"
}

RES_STAGE1="--partition=fast --time=0-06:00:00 --cpus-per-task=4 --mem=64G"
RES_MAP="--partition=long --time=3-00:00:00 --cpus-per-task=8 --mem=64G --export=ALL,THREADS=8"
RES_RULES="--partition=fast --time=1-00:00:00 --cpus-per-task=4 --mem=64G"
RES_TABLE1="--partition=fast --time=0-06:00:00 --cpus-per-task=4 --mem=128G"
RES_TSNE="--partition=long --time=2-00:00:00 --cpus-per-task=8 --mem=128G --export=ALL,THREADS=8"
RES_TABLE2="--partition=long --time=3-00:00:00 --cpus-per-task=1 --mem=32G"
RES_TABLE3="--partition=long --time=3-00:00:00 --cpus-per-task=16 --mem=256G"
RES_TABLE4="--partition=long --time=3-00:00:00 --cpus-per-task=16 --mem=512G"
RES_TIMING="--partition=long --time=3-00:00:00 --cpus-per-task=1 --mem=32G"

declare -A MAP_JOB RULES_JOB

# ---- Stage 1 (sanitization) and stage 2 (atom mapping) ---------------------------
if [[ "$START_STAGE" == 1 ]]; then
    S1_USPTO=$(submit s1_uspto "" "$RES_STAGE1" \
        python -u src/morganrxn/data_processing/uspto.py --input data/uspto/dataSetB.csv)
    S1_MNX=$(submit s1_metanetx "" "$RES_STAGE1" \
        python -u src/morganrxn/data_processing/metanetx.py)
    MAP_JOB[uspto]=$(submit map_uspto "$S1_USPTO" "$RES_MAP" \
        python -u src/morganrxn/data_processing/map_reactions.py --data uspto --batch-size 32)
    MAP_JOB[metanetx]=$(submit map_metanetx "$S1_MNX" "$RES_MAP" \
        python -u src/morganrxn/data_processing/map_reactions.py --data metanetx --batch-size 32)
else
    MAP_JOB[uspto]=""
    MAP_JOB[metanetx]=""
fi

# ---- Stage 3: reaction rules for every database and fingerprint size -------------
for DB in $DATABASES; do
    for FP in $FP_SIZES; do
        RULES_JOB[${DB}_${FP}]=$(submit "rules_${DB}_fp${FP}" "${MAP_JOB[$DB]}" "$RES_RULES" \
            python -u src/morganrxn/data_processing/create_reactionrules.py \
            --data "$DB" --radii "$RADII" --fp-size "$FP")
    done
done

# ---- Table 1: representation counts and overlap ------------------------------
for FP in $FP_SIZES; do
    submit "table1_fp${FP}" "${RULES_JOB[uspto_${FP}]}:${RULES_JOB[metanetx_${FP}]}" "$RES_TABLE1" \
        python -u src/morganrxn/paper_results/data_statistics.py \
        --radii "$RADII" --fp-size "$FP" \
        --metanetx-database-name metanetx --uspto-database-name uspto \
        --output-dir "results/data_statistics/fp${FP}" \
        --output-name reaction_vector_overlap_by_radius.csv > /dev/null
done

# ---- Figure: t-SNE (fp 1024 only) ------------------------------------------------
submit tsne "${RULES_JOB[uspto_1024]}:${RULES_JOB[metanetx_1024]}" "$RES_TSNE" \
    python -u src/morganrxn/paper_results/t_sne.py \
    --datasets metanetx uspto --radii 0 1 2 3 4 5 --encoding raw --metric cosine \
    --n-jobs 8 --output-dir results/t_sne --format pdf --save-coords > /dev/null

# ---- Table 2: applicability (reaction-centre filter vs graph level) ----------------
for FP in $FP_SIZES; do
    for DB in $DATABASES; do
        OUT="results/table2_mw${MAX_MOL_WT}/fp${FP}"
        COMMON=(--n-samples "$N_TARGETS" --fp-size "$FP" --max-mol-wt "$MAX_MOL_WT"
                --applicability-modes reaction_center
                --benchmark-dataset "${DB}=${DB}" --paired-rules "${DB}=${DB}")
        # radius 0 has by far the most candidate pairs, so it runs as its own job
        submit "t2_fp${FP}_${DB}_r0" "${RULES_JOB[${DB}_${FP}]}" "$RES_TABLE2" \
            python -u src/morganrxn/paper_results/applicability_accuracy.py \
            --radii 0 "${COMMON[@]}" \
            --out-xlsx "${OUT}/applicability_${DB}_r0.xlsx" > /dev/null
        submit "t2_fp${FP}_${DB}_r1-5" "${RULES_JOB[${DB}_${FP}]}" "$RES_TABLE2" \
            python -u src/morganrxn/paper_results/applicability_accuracy.py \
            --radii 1,2,3,4,5 "${COMMON[@]}" \
            --out-xlsx "${OUT}/applicability_${DB}_r1-5.xlsx" > /dev/null
    done
done

# ---- Table 3: USPTO reaction-class prediction (one job per radius) ----------------
for FP in $FP_SIZES; do
    for R in ${RADII//,/ }; do
        submit "table3_fp${FP}_r${R}" "${RULES_JOB[uspto_${FP}]}" "$RES_TABLE3" \
            python -u src/morganrxn/paper_results/uspto_prediction.py \
            --database-name uspto --radii "$R" --fp-size "$FP" \
            --dataset-path data/uspto/dataSetB.csv \
            --models logistic_regression,random_forest,gradient_boosting,mlp \
            --output-dir "results/uspto_prediction/fp${FP}" \
            --summary-output "results/uspto_prediction/fp${FP}/metrics_r${R}.csv" \
            --save-meta > /dev/null
    done
done

# ---- Table 4: MetaNetX EC-number prediction ------------------------------------
for FP in $FP_SIZES; do
    submit "table4_fp${FP}" "${RULES_JOB[metanetx_${FP}]}" "$RES_TABLE4" \
        python -u src/morganrxn/paper_results/metanetx_ec_prediction.py \
        --ec-levels 1,2,3,4 --radii "$RADII" --fp-size "$FP" \
        --min-label-count 5 --max-labels 300 --models sgd,et \
        --feature-sets reaction_ecfp,reaction_center_ecfp,both \
        --sample-mode unique_rules --n-jobs 16 \
        --output-dir "results/metanetx_ec_prediction/fp${FP}" \
        --summary-output "results/metanetx_ec_prediction/fp${FP}/metrics_all_ec_levels_all_radii.csv" \
        --save-meta > /dev/null
done

# ---- Runtime: graph vs vector on the Table 2 targets (fp 1024) ---------------------
for DB in $DATABASES; do
    for RG in 0 1:2:3:4:5; do
        for ((S = 0; S < N_TARGETS; S += TIMING_CHUNK)); do
            submit "timing_${DB}_r${RG//:/}_t$(printf %04d "$S")" "${RULES_JOB[${DB}_1024]}" "$RES_TIMING" \
                python -u src/morganrxn/paper_results/graph_vs_vector_timing.py \
                --database-name "$DB" --radii "${RG//:/,}" --fp-size 1024 \
                --n-samples "$N_TARGETS" --max-mol-wt "$MAX_MOL_WT" \
                --target-start "$S" --target-end $((S + TIMING_CHUNK)) > /dev/null
        done
    done
done

echo "Submitted run ${RUN_ID} (commit $(cat "$LOG_DIR/commit.txt"))."
echo "Logs: ${LOG_DIR}/   Follow with: squeue -u \$USER"
