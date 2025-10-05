# Deep Learning for Text Classification — CNN, RNN, LSTM, GRU

**Problem.** Build and compare deep learning models for text classification to analyze sentiment in book reviews, supporting the PyBooks recommendation engine.

**Data.** Custom book review samples and `fetch_20newsgroups` dataset (categories: rec.autos, sci.med, comp.graphics).

**Approach.**
- Implemented multiple deep learning models for text classification:
  - **CNN** — captures local word patterns using 1D convolutions.
  - **RNN** — sequential modeling for contextual dependencies.
  - **LSTM** — handles long-term dependencies and vanishing gradients.
  - **GRU** — simplified recurrent structure for efficient training.
- Used embeddings to represent words numerically.
- Compared architecture behaviors without focusing on raw metrics.

**Results (qualitative).**
- CNN effectively captured localized text patterns.
- LSTM and GRU demonstrated smoother learning for sequential context.
- RNN served as a conceptual baseline for understanding recurrence.

**What I Learned.**
- Fundamental architecture differences between CNNs and RNN-family models for NLP.
- How embeddings and tokenization shape model performance.
- Practical trade-offs between accuracy, complexity, and interpretability in deep learning.

## Quick Start
```bash
git clone https://github.com/Joe-Naz01/text-classification-nlp.git
cd text-classification-nlp
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
