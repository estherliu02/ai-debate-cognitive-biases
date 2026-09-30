# Debate

Supplementary material: code, data, and analysis for the detective debate study.

## Layout

```
code/      generation pipeline (configs, prompts, core, scripts)
data/      accusation results and the 68 study dialogues
results/   participant judgments (results.csv) and H1–H2b analysis notebooks
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=YOUR_KEY   # needed to build cases and generate dialogues
```

## Build detective cases

The detective cases are not included. Build them from the source dataset, [True Detective](https://github.com/MaksymDel/true-detective) (Del & Fishel, 2023):

```bash
git clone https://github.com/MaksymDel/true-detective data/true-detective
python code/convert_true_detective.py
```

Reads `data/true-detective/data/data.zip` and writes the four study cases to `data/detective_cases/`.
The suspect evidence analysis calls an LLM through OpenRouter, so this step needs `OPENROUTER_API_KEY`.

## Generate dialogues

Requires the detective cases above.

```bash
python code/scripts/run_detective_v2.py --step full
```

Settings are in `code/configs/generation_detective.yaml`. Outputs go to `outputs/`.
Run `python code/scripts/run_detective_v2.py --help` for all options.

## Analysis

`results/primary_analysis_H1.ipynb`, `_H2a`, and `_H2b` test hypotheses H1–H2b on `results/results.csv`.
Open them in VS Code or Jupyter.
Running `_H2a` also generates the power curve, `results/power_curve.png` (with its data in `results/power_curve.csv`).

`readme.ipynb` walks through all of the steps above in one notebook.
