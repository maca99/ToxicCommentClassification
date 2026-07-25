# Toxic Comment Classification

Progetto di Natural Language Processing per la classificazione automatica di commenti tossici, sviluppato sul dataset della [Jigsaw/Conversation AI Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge). Il progetto confronta diversi approcci di modellazione, da metodi classici di machine learning a modelli di deep learning basati su word embedding.

## Descrizione

L'obiettivo è classificare commenti testuali secondo diverse categorie di tossicità (es. tossico, osceno, minaccioso, offensivo, odio identitario), analizzando e confrontando le prestazioni di più famiglie di modelli:

- **Modelli classici di machine learning**, basati su rappresentazioni testuali tradizionali (es. TF-IDF / bag-of-words).
- **Modelli di deep learning**, basati su reti neurali (es. LSTM/GRU) che sfruttano word embedding pre-addestrati **GloVe**.

Il lavoro è documentato nel report `MLFMDE_Report.pdf`.

## Struttura del repository

```
ToxicCommentClassification/
├── data/                    # Dataset (training/test) del progetto
├── models/                  # Modelli addestrati / salvati
├── DataAnalisys.ipynb       # Analisi esplorativa del dataset
├── Models.ipynb             # Costruzione e addestramento dei modelli
├── Testing.ipynb            # Valutazione e confronto dei modelli
├── tokenizer.json           # Tokenizer salvato per il preprocessing del testo
├── MLFMDE_Report.pdf        # Report del progetto
└── README.md
```

## Dataset

Il progetto utilizza il dataset della [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge), contenente commenti Wikipedia etichettati secondo diverse categorie di tossicità.

## Word Embedding

Per i modelli di deep learning è richiesto **GloVe** (versione con **6 miliardi di token**, `glove.6B`), da scaricare separatamente:

1. Scaricare gli embedding da [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/) (versione `glove.6B`).
2. Posizionare il file scaricato nella cartella del progetto (es. `data/`) prima di eseguire i notebook di training.

## Prerequisiti

- Python 3
- Jupyter Notebook / JupyterLab
- Principali librerie: `numpy`, `pandas`, `scikit-learn`, un framework di deep learning (es. `tensorflow`/`keras` o `pytorch`, a seconda dell'implementazione nei notebook)

## Utilizzo

1. Clonare il repository:
   ```bash
   git clone https://github.com/maca99/ToxicCommentClassification.git
   cd ToxicCommentClassification
   ```
2. Scaricare il dataset Kaggle e gli embedding GloVe come indicato sopra.
3. Eseguire i notebook nell'ordine:
   - `DataAnalisys.ipynb` — esplorazione e preprocessing del dataset
   - `Models.ipynb` — addestramento dei modelli (classici e deep learning)
   - `Testing.ipynb` — valutazione e confronto delle prestazioni

## Report

Per l'analisi dettagliata della metodologia, dei modelli confrontati e dei risultati ottenuti, consultare `MLFMDE_Report.pdf`.
