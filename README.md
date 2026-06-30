# MOUT

This repository contains the standalone core implementation of MOUT extracted
from `/home/xgd/propositionMultiObj/multiObj_area.py`.

The public algorithm class is `MOUT` in `code/mout.py`. The module also
contains the dataset problem, proposition mapping, reference-vector,
crossover, and mutation helpers required by the run loop.

## Project layout

```text
MOUT3/
|-- code/
|   |-- mout.py
|   `-- run_mout.py
|-- dataset/
|   |-- systems/<system>.csv
|   `-- propositions/Data_<system>.txt
|-- requirements.txt
`-- README.md
```

## Environment setup

Python 3.10 is recommended. The `.12` server already has the required
packages in the `multi` Conda environment:

```bash
cd /home/xgd/propositionMultiObj/MOUT3
conda activate multi
```

For a clean Python environment:

```bash
python -m pip install -r requirements.txt
```

## Input convention

`--system HSQLDB` loads:

```text
dataset/systems/HSQLDB.csv
dataset/propositions/Data_HSQLDB.txt
```

`--proposition data1` selects the Python variable named `data1` from the
proposition file. The supplied proposition files normally define
`data1` through `data10`.

Available systems:

```text
HSQLDB
brotli
lrzip
MariaDB
sac_nbody_others
LLVM
sac_srad_others
ExaStencils
MongoDB
spear17
VP8
```

System names are case-sensitive and must match the dataset filenames.

## Run one experiment

Run HSQLDB with `data1`, seed 1, population 10, and budget 300:

```bash
python code/run_mout.py \
  --system HSQLDB \
  --proposition data1 \
  --budget 300 \
  --population-size 10 \
  --seed 1
```

The final result is written to:

```text
results/HSQLDB/data1/seed_1.json
```

The progress trajectory is written below the same output directory under
`prop_results/seed_1/`.

Use the following command to see every option:

```bash
python code/run_mout.py --help
```

## Basic tests

Check syntax and imports:

```bash
python -m py_compile code/mout.py code/run_mout.py
python -c "import sys; sys.path.insert(0, 'code'); from mout import MOUT; print(MOUT.__name__)"
```

Run a short smoke test:

```bash
python code/run_mout.py \
  --system HSQLDB \
  --proposition data1 \
  --budget 2 \
  --seed 1 \
  --output-dir /tmp/mout3-smoke
```

Confirm that `/tmp/mout3-smoke/seed_1.json` exists and contains `hv`,
`evaluations`, and `front`.

## Run 100 repetitions

The following example runs one system/proposition for seeds 1 through 100:

```bash
SYSTEM=HSQLDB
PROP=data1
OUT=results/main/${SYSTEM}/${PROP}

mkdir -p logs/main/${SYSTEM}/${PROP}
for SEED in $(seq 1 100); do
  python code/run_mout.py \
    --system "$SYSTEM" \
    --proposition "$PROP" \
    --budget 300 \
    --seed "$SEED" \
    --output-dir "$OUT" \
    > "logs/main/${SYSTEM}/${PROP}/seed_${SEED}.log" 2>&1
done
```

To reproduce the complete experiment, repeat this loop for all 11 systems and
`data1` through `data10`. Store every experimental variant in a separate
output directory. Do not mix results produced by different source settings.

## RQ3: strategy ablation

The old source referred to this logic by line numbers around 499-510. Line
numbers change when the file is cleaned, so use the following markers in
`MOUT.run()` instead:

```text
# RQ3/RQ4 EXPERIMENT BLOCK START
...
# RQ3/RQ4 EXPERIMENT BLOCK END
```

Save a backup before editing:

```bash
cp code/mout.py code/mout.py.backup
```

Run the following three RQ3 variants separately.

### Normal

Comment out the complete marked experiment block. This disables proposition
changes after each generation.

Store results under:

```text
results/RQ3/Normal/<system>/<proposition>/
```

### Sustainable

Replace the complete marked block with:

```python
mode = "improving"
self.current_mode = mode
self.change_propositions(mode)
```

Store results under:

```text
results/RQ3/Sustainable/<system>/<proposition>/
```

### Temporary

Replace the complete marked block with:

```python
mode = "temp"
self.current_mode = mode
self.change_propositions(mode)
```

Store results under:

```text
results/RQ3/Temporary/<system>/<proposition>/
```

After each edit, run `py_compile` and the smoke test before starting the full
100-run experiment. Restore the main implementation after RQ3:

```bash
mv code/mout.py.backup code/mout.py
```

Example RQ3 command after selecting a variant:

```bash
VARIANT=Temporary
SYSTEM=HSQLDB
PROP=data1

for SEED in $(seq 1 100); do
  python code/run_mout.py \
    --system "$SYSTEM" \
    --proposition "$PROP" \
    --budget 300 \
    --seed "$SEED" \
    --output-dir "results/RQ3/${VARIANT}/${SYSTEM}/${PROP}"
done
```

## RQ4: stagnation-threshold sensitivity

RQ4 uses the complete marked experiment block. Change these two class
constants near the beginning of `class MOUT`:

```python
SUSTAINABLE_THRESHOLD = 1
TEMPORARY_THRESHOLD = 5
```

The four tested parameter pairs are:

| Variant | `SUSTAINABLE_THRESHOLD` | `TEMPORARY_THRESHOLD` |
|---|---:|---:|
| `1_5` | 1 | 5 |
| `1_10` | 1 | 10 |
| `3_8` | 3 | 8 |
| `5_10` | 5 | 10 |

For each pair:

1. Update both constants.
2. Run `py_compile` and the smoke test.
3. Run 100 seeds for every system/proposition.
4. Write results to a directory named after the pair.

Example:

```bash
PAIR=3_8
SYSTEM=HSQLDB
PROP=data1

for SEED in $(seq 1 100); do
  python code/run_mout.py \
    --system "$SYSTEM" \
    --proposition "$PROP" \
    --budget 300 \
    --seed "$SEED" \
    --output-dir "results/RQ4/${PAIR}/${SYSTEM}/${PROP}"
done
```

Do not change random seeds, budget, population size, or datasets between RQ4
variants. Only the two threshold constants should differ.

## HV convention

MOUT optimizes proposition satisfaction scores. The reported HV uses the
original proposition satisfaction values and reference point `0.1` in every
objective. Maximization values are negated before calling `pymoo`'s HV
indicator.
