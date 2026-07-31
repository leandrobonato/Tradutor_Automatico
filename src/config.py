"""Configuração central do projeto: caminhos e hiperparâmetros."""
from pathlib import Path

# --- Caminhos -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, TOKENIZER_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Dataset --------------------------------------------------------------
HF_DATASET_NAME = "Helsinki-NLP/opus-100"
HF_DATASET_CONFIG = "en-pt"
SRC_LANG = "en"
TGT_LANG = "pt"

# Meta de negócio: ~100.000 pares de treino (subamostrados do OPUS-100, que
# já reúne fontes como Tatoeba, Europarl e OpenSubtitles via o projeto OPUS).
TRAIN_SAMPLE_SIZE = 100_000
MAX_SENTENCE_LEN_CHARS = 200  # filtra frases muito longas (ruído/outliers)
MIN_SENTENCE_LEN_CHARS = 1
MAX_LEN_RATIO = 2.5  # descarta pares com razão de tamanho EN/PT muito desbalanceada

RANDOM_SEED = 42

# --- Tokenização (SentencePiece BPE) --------------------------------------
VOCAB_SIZE = 16_000
SPM_MODEL_TYPE = "bpe"
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]

# --- Modelo -----------------------------------------------------------
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
ENCODER_LAYERS = 1
DROPOUT = 0.3
MAX_DECODE_LEN = 60

# --- Treino -----------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5
GRAD_CLIP = 1.0
BEAM_WIDTH = 3
