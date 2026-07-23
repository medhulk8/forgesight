# Kaggle handoff (§13 step 8) — run these on your side

The M3 side is done: code pushed, dataset generated under `data/processed/`
(gitignored — 2964 examples: `train.jsonl`/`val.jsonl`/`test.jsonl` + `images/`,
~2.1 GB). Three manual steps, then Run-All the notebook.

## 1. Upload the dataset (private)
From the machine that has `data/processed/` (this M3), with the Kaggle CLI configured
(`~/.kaggle/kaggle.json`):

```bash
# edit the id first: replace the username placeholder
#   data/processed/dataset-metadata.json  ->  "id": "<your-kaggle-username>/forgesight-data"
kaggle datasets create -p data/processed --dir-mode zip
```
- Keep it **Private**. The slug must be **`forgesight-data`** so it mounts at
  `/kaggle/input/forgesight-data/`.
- To update later: `kaggle datasets version -p data/processed -m "rebuild" --dir-mode zip`.

## 2. GitHub PAT in Kaggle Secrets
- GitHub → Settings → Developer settings → fine-grained PAT, **read-only** on
  `medhulk8/forgesight`.
- Kaggle notebook → **Add-ons → Secrets** → add `GH_PAT` = the token.

## 3. Notebook settings
- Open `notebooks/forgesight_kaggle.ipynb` on Kaggle.
- Attach the `forgesight-data` dataset (Add Input).
- Accelerator = **GPU T4 ×2**, Internet = **On**.
- **Run All**, stop after the Overfit-8 cell.

## Gates to report back
- **Step 9:** 4-bit model loads on one T4; `print_trainable_parameters` shows only LoRA params.
- **Step 10:** overfit-8 train loss → ~0; cell 7 reproduces the 8 target JSONs verbatim.

If loss doesn't drop or JSONs don't reproduce → collator/masking/model wiring bug;
send the loss curve + cell-7 output and we debug before any full run.
