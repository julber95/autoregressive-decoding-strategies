# Evaluating Autoregressive Decoding Strategies

This repository contains the code and resources for a project comparing different decoding strategies (greedy, beam search, top-k, nucleus) on an LSTM‐based text generation model.

## Structure

- **models/**  
  Saved PyTorch checkpoints (`.pt`) of the trained LSTM model (too large for github).
- **utils/**  
  Generated figures (bar charts, scatter plots, heatmaps, beam‐search trees) used in the report.  
- **wikitext2/**  
  Raw WikiText-2 dataset splits: `train.txt`, `valid.txt`, `test.txt`.  
- **data_processing.py**  
  Cleans the corpus, filters `<unk>` lines, tokenizes, builds vocabulary, and creates training tensors.  
- **decoding_strategies.py**  
  Implements four decoding algorithms:  
  - Greedy  
  - Beam Search  
  - Top-k Sampling  
  - Nucleus (Top-p) Sampling  
- **evaluation.py**  
  Computes automatic metrics: BLEU, ROUGE.
- **generate_evaluations_plots.py**  
  Generates comparative performance plots for each decoding strategy.  
- **generate_text.py**  
  Produces sample text continuations from the trained model for any chosen strategy.  
- **generate_visualizations.py**  
  Creates illustrative plots (probability distributions, dynamic cutoffs, cumulative curves) for each method.  
- **model.py**  
  Defines the LSTM architecture: embedding layer, stacked LSTM cells, dropout, and output projection.  
- **train.py**  
  Main training script: data loading, training loop, checkpoint saving, and logging.  
- **plot_utils.py**  
  Utility functions for plotting (bar charts, scatter plots, heatmaps).  
- **notebook_visualization.ipynb**  
  Jupyter notebook for rapid prototyping.
