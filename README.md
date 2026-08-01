# Finding the Sweet Spot

Official code for the paper:

> **Finding the Sweet Spot: An Empirical Study on Dataset Size, Performance, and Efficiency in Relation Extraction**  
> Sohrab Pirhadi, Babak Niroomand, Ebrahim Ansari  
> Institute for Advanced Studies in Basic Sciences (IASBS)

This repository implements the experimental pipeline used in the paper: dataset partitioning into eight training subsets (SE.1–SE.8), training of four relation extraction models, and evaluation with Accuracy, Precision, Recall, and F-score.

A PDF copy of the paper is included as `[paper.pdf](paper.pdf)`.

---



## Citation

If you use this code or refer to the study, please cite:

```bibtex
@article{pirhadi2024sweetspot,
  title={Finding the Sweet Spot: An Empirical Study on Dataset Size, Performance, and Efficiency in Relation Extraction},
  author={Sohrab Pirhadi, Babab Niroomand,Ebrahim Ansari},
  journal={IEEE},
  year={2025}
}
```

Please also cite the original model and dataset papers listed in the Acknowledgements.

---



## Requirements

- Python 3.10+
- PyTorch with CUDA (recommended) or CPU
- SemEval-2010 Task 8 (downloaded automatically)

Install dependencies:

```bash
# CUDA 12.4 build (recommended for NVIDIA GPUs)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Verify the GPU (optional):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---



## Dataset Preparation

The SemEval-2010 Task 8 corpus is downloaded and parsed automatically:

```bash
python reproduce.py --stage data
```

This step:

1. Downloads the shared-task files into `data/raw/`
2. Parses train (8,000) and test (2,717) instances
3. Creates training subsets SE.1–SE.8 via Algorithm 1 (fixed seed)
4. Writes subset index files under `data/processed/subsets/`

Subset sizes follow Table I of the paper:


| Subset | Portion | Instances |
| ------ | ------- | --------- |
| SE.1   | 1/128   | 62        |
| SE.2   | 1/64    | 125       |
| SE.3   | 1/32    | 250       |
| SE.4   | 1/16    | 500       |
| SE.5   | 1/8     | 1,000     |
| SE.6   | 1/4     | 2,000     |
| SE.7   | 1/2     | 4,000     |
| SE.8   | 1       | 8,000     |


Word embeddings for Att-CNN / Att-BiLSTM are loaded via `gensim` (`word2vec-google-news-300`) on first use.

---



## Models


| Model      | Description                                                         | Module                 |
| ---------- | ------------------------------------------------------------------- | ---------------------- |
| Att-CNN    | Attention-based CNN (Shen & Huang, COLING 2016)                     | `models/att_cnn.py`    |
| Att-BiLSTM | Attention-based BiLSTM (Zhou et al., ACL 2016)                      | `models/att_bilstm.py` |
| R-BERT     | Entity-aware BERT (Wu & He, CIKM 2019)                              | `models/rbert.py`      |
| MTB        | BERT entity markers / supervised MTB head (Soares et al., ACL 2019) | `models/mtb.py`        |


Hyperparameters are specified in `configs/`.

---



## Training

Train all models on all subsets (default device: CUDA):

```bash
python reproduce.py --stage train
```

Train a single model or subset:

```bash
python reproduce.py --stage train --models rbert --subsets SE.8
python reproduce.py --stage train --models att_cnn att_bilstm
```

Useful flags:


| Flag                | Meaning                          |
| ------------------- | -------------------------------- |
| `--device cuda`     | Use GPU (default)                |
| `--device cpu`      | Force CPU                        |
| `--seed 42`         | Global random seed               |
| `--quick`           | Fewer epochs (debug only)        |
| `--skip-embeddings` | Random word vectors (debug only) |


Checkpoints are written to `results/checkpoints/` (gitignored).

---



## Evaluation

Metrics are computed on the official SemEval-2010 Task 8 test set after each training run and saved under `results/metrics/`.

Regenerate Table I, Table II, and Figures 1–2 from saved metrics:

```bash
python reproduce.py --stage tables
```

Outputs:


| Artifact                     | Path                                    |
| ---------------------------- | --------------------------------------- |
| Table I                      | `results/tables/table_i_subsets.*`      |
| Table II                     | `results/tables/table_ii_performance.*` |
| Figure 1 (Accuracy)          | `results/figures/figure1_accuracy.png`  |
| Figure 2 (F-score)           | `results/figures/figure2_fscore.png`    |
| Comparison to paper Table II | `results/tables/comparison_vs_paper.*`  |


---



## Full Reproduction

End-to-end pipeline (data → train → tables/figures):

```bash
python reproduce.py
```

A full run (4 models × 8 subsets) typically takes several hours on a modern NVIDIA GPU.

Published metric JSON files and regenerated tables/figures are included under `results/` so readers can inspect outputs without retraining.

---



## Repository Structure

```
.
├── configs/               # Global and per-model hyperparameters
├── data/
│   ├── raw/               # SemEval download (auto-populated)
│   └── processed/
│       └── subsets/       # SE.1–SE.8 sampling indices
├── models/                # Att-CNN, Att-BiLSTM, R-BERT, MTB
├── src/                   # Data, training, evaluation, visualization
├── results/
│   ├── figures/           # Figures 1–2
│   ├── tables/            # Tables I–II
│   └── metrics/           # Per-run metrics (JSON)
├── paper.pdf
├── reproduce.py           # Main entry point
├── requirements.txt
├── LICENSE
└── README.md
```

---



## Reproducibility Notes

- Default random seed: **42** (`configs/default.yaml`)
- Seeds are set for Python, NumPy, PyTorch, CUDA, HuggingFace Transformers, and DataLoader workers
- Subset indices under `data/processed/subsets/` fix Algorithm 1 sampling for bit-stable splits
- MTB in this codebase fine-tunes BERT with entity-marker inputs (BERT EM). Full unsupervised Matching-the-Blanks pretraining on entity-linked Wikipedia is not included

---



## Acknowledgements

This work builds on:

- SemEval-2010 Task 8 (Hendrickx et al., 2010)
- Att-CNN (Shen & Huang, COLING 2016)
- Att-BiLSTM (Zhou et al., ACL 2016)
- R-BERT (Wu & He, CIKM 2019)
- Matching the Blanks (Baldini Soares et al., ACL 2019)

We thank the authors of the open-source reference implementations that informed the model ports used here.

---



## License

This code is released under the MIT License. See `[LICENSE](LICENSE)`.
The SemEval-2010 Task 8 dataset remains subject to its original distribution terms.