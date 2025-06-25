from functools import lru_cache

# Suppress warnings from thinc/spaCy about deprecated torch.cuda.amp.autocast
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch*")

# Lazy, cached loading of heavy ML models used across the application
# ---------------------------------------------------------------
# We centralise this here so that ingest.py, query.py and any other
# modules only import the models once.  Subsequent calls just return the
# cached singleton, eliminating redundant initialisation and memory use.

from sentence_transformers import SentenceTransformer
import spacy


@lru_cache(maxsize=1)
def get_spacy_model():
    """Return a cached spaCy transformer model.

    The first call loads the heavyweight `en_core_web_trf` pipeline.  All
    subsequent calls return the already instantiated model, ensuring we
    pay the loading cost only once per interpreter lifetime.
    """
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        # Fallback: the model might not be installed.  We raise a clear
        # error so that the caller can handle / notify appropriately.
        raise RuntimeError(
            "spaCy model 'en_core_web_trf' is not installed. Install it with: "
            "python -m spacy download en_core_web_trf"
        )
    return nlp


@lru_cache(maxsize=1)
def get_embedding_model():
    """Return a cached SentenceTransformer model for embeddings."""
    return SentenceTransformer("all-MiniLM-L6-v2") 