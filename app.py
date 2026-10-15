from flask import Flask, jsonify, request, render_template
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Resolve files relative to this script so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'next_word_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')
CONFIG_PATH = os.path.join(BASE_DIR, 'model_config.pkl')

# Load model, tokenizer and config once at startup
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)

max_sequence_len = config['max_sequence_len']
vocab_size = config['vocab_size']

TOP_K = 3
MAX_INPUT_CHARS = 2000
SKIP_TOKENS = {'<OOV>', tokenizer.oov_token}


def is_valid_word(word):
    """Reject padding/OOV tokens and punctuation-only tokens such as a lone "'"."""
    return bool(word) and word not in SKIP_TOKENS and any(ch.isalnum() for ch in word)


def predict_next_words(text, top_k=TOP_K):
    """Return the top_k most likely next words for the given text."""
    try:
        seq = tokenizer.texts_to_sequences([text.lower()])[0]
        if not seq:
            return []

        # Same padding/truncation as training: keep the last (max_sequence_len - 1) tokens
        seq = pad_sequences([seq], maxlen=max_sequence_len - 1,
                            padding='pre', truncating='pre')
        pred = model.predict(seq, verbose=0)[0]

        # Walk down the ranked list until we have top_k real words.
        # (Filtering after slicing [:top_k] could return only 1-2 suggestions.)
        results = []
        for idx in np.argsort(pred)[::-1]:
            word = tokenizer.index_word.get(int(idx))
            if not is_valid_word(word):
                continue
            results.append({'word': word,
                            'confidence': round(float(pred[idx]) * 100, 1)})
            if len(results) == top_k:
                break
        return results
    except Exception as e:
        app.logger.exception("Error in prediction: %s", e)
        return []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected a JSON object like {"text": "..."}',
                        'predictions': []}), 400

    text = data.get('text', '')
    if not isinstance(text, str):
        return jsonify({'error': '"text" must be a string', 'predictions': []}), 400

    text = text.strip()[-MAX_INPUT_CHARS:]
    if not text:
        return jsonify({'predictions': []})

    return jsonify({'predictions': predict_next_words(text)})


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, port=port)
