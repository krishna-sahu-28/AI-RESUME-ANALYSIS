"""
Machine Learning Classifier & Semantic Matching Pipeline for AI Resume Analysis.

Provides:
1. Singleton loader for the local BERT SentenceTransformer model (ml/models/bert_resume_matcher)
   ensuring the neural network loads into memory only once across requests.
2. BERT-based semantic cosine similarity and ATS Match Score calculation.
3. Scikit-learn TF-IDF + Logistic Regression career category prediction with confidence scoring.
"""

import os
import logging
import threading
import numpy as np
import joblib

logger = logging.getLogger(__name__)

# Base paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_DIR = os.path.join(CURRENT_DIR, 'models', 'bert_resume_matcher')
SKLEARN_MODEL_PATH = os.path.join(CURRENT_DIR, 'models', 'resume_classifier.joblib')


# =====================================================================
# 1. Thread-Safe Singleton Loader for Local BERT SentenceTransformer
# =====================================================================

class BertModelSingleton:
    """
    Thread-safe Singleton container for the local BERT SentenceTransformer model.
    Guarantees that large model weights (90MB+) are loaded into memory only once
    at startup or upon first request, preventing memory leaks and request latency.
    """
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BertModelSingleton, cls).__new__(cls)
        return cls._instance

    def get_model(self):
        """
        Retrieves the cached SentenceTransformer instance or loads it if not yet loaded.
        """
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            if not os.path.isdir(BERT_MODEL_DIR):
                logger.warning("BERT model directory not found at: %s", BERT_MODEL_DIR)
                return None

            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading local BERT SentenceTransformer model from %s...", BERT_MODEL_DIR)
                self._model = SentenceTransformer(BERT_MODEL_DIR, device='cpu')
                logger.info("Local BERT SentenceTransformer model loaded successfully.")
                return self._model
            except Exception as e:
                logger.error("Failed to load BERT SentenceTransformer model: %s", e)
                return None

    def compute_embedding(self, text):
        """
        Computes a normalized document embedding vector for the given text.
        Handles long documents through chunking if necessary.
        """
        model = self.get_model()
        if model is None or not text or not text.strip():
            return None

        clean_text = text.strip()
        try:
            # Check tokenizer chunking for long texts exceeding max_seq_length
            if hasattr(model, 'tokenizer') and hasattr(model, 'max_seq_length'):
                token_ids = model.tokenizer.encode(clean_text, add_special_tokens=False)
                if not token_ids:
                    return None
                chunk_size = max(1, int(model.max_seq_length) - 2)
                if len(token_ids) > chunk_size:
                    token_chunks = [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]
                    text_chunks = [
                        model.tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        for chunk in token_chunks
                    ]
                    vectors = model.encode(
                        text_chunks,
                        batch_size=16,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                    weights = np.asarray([len(chunk) for chunk in token_chunks], dtype=np.float32)
                    vector = np.average(np.asarray(vectors, dtype=np.float32), axis=0, weights=weights)
                    norm = float(np.linalg.norm(vector))
                    return (vector / norm) if norm > 0 else None

            # Standard encoding with normalization
            vector = model.encode(
                clean_text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vector
        except Exception as exc:
            logger.warning("Error generating embedding: %s", exc)
            return None

    def compute_similarity(self, text1, text2):
        """
        Computes cosine similarity between two texts using the singleton BERT model.
        Returns: float in range [-1.0, 1.0], or None if inference fails.
        """
        vec1 = self.compute_embedding(text1)
        vec2 = self.compute_embedding(text2)
        if vec1 is None or vec2 is None:
            return None

        sim = float(np.dot(vec1, vec2))
        return float(np.clip(sim, -1.0, 1.0))


def get_bert_model():
    """
    Public accessor function to get the singleton BERT SentenceTransformer model.
    """
    return BertModelSingleton().get_model()


def calculate_match_score(resume_text, job_description_text):
    """
    Computes the ATS Semantic Match Score between resume text and job description
    using the Singleton local BERT SentenceTransformer model.

    Returns:
        dict: {
            'match_score': float (0.0 to 100.0),
            'cosine_similarity': float (e.g. 0.8523),
            'available': bool,
            'match_level': str ('High', 'Moderate', 'Needs Alignment'),
            'model_name': str,
            'message': str
        }
    """
    if not resume_text or not resume_text.strip():
        return {
            'match_score': 0.0,
            'cosine_similarity': 0.0,
            'available': False,
            'match_level': 'N/A',
            'model_name': 'all-MiniLM-L6-v2 (BERT SentenceTransformer)',
            'message': 'Resume text is empty.'
        }

    if not job_description_text or not job_description_text.strip():
        return {
            'match_score': 0.0,
            'cosine_similarity': 0.0,
            'available': False,
            'match_level': 'N/A',
            'model_name': 'all-MiniLM-L6-v2 (BERT SentenceTransformer)',
            'message': 'Job description is empty.'
        }

    singleton = BertModelSingleton()
    sim = singleton.compute_similarity(resume_text, job_description_text)

    if sim is None:
        return {
            'match_score': 0.0,
            'cosine_similarity': 0.0,
            'available': False,
            'match_level': 'N/A',
            'model_name': 'all-MiniLM-L6-v2 (BERT SentenceTransformer)',
            'message': 'Semantic matching model unavailable.'
        }

    # Cosine similarity for sentence transformers typically falls between 0.0 and 1.0.
    # Map cosine similarity to 0-100 percentage.
    normalized_sim = max(0.0, min(1.0, sim))
    match_score = round(normalized_sim * 100, 1)

    if match_score >= 75.0:
        match_level = 'High Match'
    elif match_score >= 50.0:
        match_level = 'Moderate Match'
    else:
        match_level = 'Needs Alignment'

    return {
        'match_score': match_score,
        'cosine_similarity': round(sim, 4),
        'available': True,
        'match_level': match_level,
        'model_name': 'all-MiniLM-L6-v2 (BERT SentenceTransformer)',
        'message': 'Successfully computed semantic match.'
    }


# =====================================================================
# 2. Scikit-Learn Career Role Classifier
# =====================================================================

_SKLEARN_MODEL = None
_SKLEARN_LOCK = threading.Lock()

def get_classifier_model():
    """
    Loads and caches the trained scikit-learn classifier pipeline (TF-IDF + Logistic Regression).
    Thread-safe and cached in memory.
    """
    global _SKLEARN_MODEL
    if _SKLEARN_MODEL is not None:
        return _SKLEARN_MODEL

    with _SKLEARN_LOCK:
        if _SKLEARN_MODEL is not None:
            return _SKLEARN_MODEL

        if os.path.isfile(SKLEARN_MODEL_PATH):
            try:
                _SKLEARN_MODEL = joblib.load(SKLEARN_MODEL_PATH)
                return _SKLEARN_MODEL
            except Exception as e:
                logger.error("Failed to load scikit-learn classifier: %s", e)
                return None
    return None


def predict_role(resume_text):
    """
    Runs ML classification inference on extracted resume text to predict career category.
    Returns:
        dict: {
            'predicted_role': str,
            'confidence': float (0.0 to 100.0),
            'class_probabilities': dict of {role: percentage},
            'model_name': str
        } or None if model unavailable or empty text.
    """
    if not resume_text or not resume_text.strip():
        return None

    model = get_classifier_model()
    if model is None:
        return None

    try:
        prediction = model.predict([resume_text])[0]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([resume_text])[0]
            confidence = float(round(max(probabilities) * 100, 1))
            class_probs = {
                str(cls): float(round(prob * 100, 1))
                for cls, prob in zip(model.classes_, probabilities)
            }
            # Sort descending by probability
            sorted_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True))
        else:
            confidence = 100.0
            sorted_probs = {str(prediction): 100.0}

        return {
            'predicted_role': str(prediction),
            'confidence': confidence,
            'class_probabilities': sorted_probs,
            'model_name': 'Scikit-Learn TF-IDF + Logistic Regression'
        }
    except Exception as exc:
        logger.error("Error predicting career role: %s", exc)
        return None


# Backwards compatibility alias
classify_resume_text = predict_role
