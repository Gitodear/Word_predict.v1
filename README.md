# Word Predictor AI - Web Application

A web-based next-word predictor powered by an LSTM neural network trained on classical literature from Project Gutenberg.

📄 Full project report: [`docs/Project_Report.pdf`](docs/Project_Report.pdf)

## Features

✨ **Real-time Predictions** - Top 3 next-word suggestions as you type
🎨 **Modern UI** - Dark, responsive design
⌨️ **Virtual Keyboard** - QWERTY layout with Shift / Caps Lock, numbers and symbols
📱 **Mobile Friendly** - Works on desktop and phone screens
🚀 **Fast Inference** - Model is loaded once at startup and kept in memory

## Requirements

- Python 3.10 – 3.13
- Flask 3.x
- TensorFlow 2.16+ with Keras 3.11+ (the model file was saved with Keras 3.11; TensorFlow 2.15 and older cannot load it)
- NumPy 1.26+

## Installation

1. **Create a virtual environment (recommended) and install dependencies**:

   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Make sure the model files are next to `app.py`**:
   - `next_word_model.keras` - trained LSTM model
   - `tokenizer.pkl` - fitted Keras tokenizer
   - `model_config.pkl` - model configuration (`max_sequence_len`, `vocab_size`)

## Running the Application

```bash
python app.py
```

Then open **http://localhost:5000** and start typing.

Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | Port to serve on |
| `FLASK_DEBUG` | `0` | Set to `1` for auto-reload and debug pages during development |

## Usage

- **Type normally** or use the virtual keyboard
- **Click a prediction** to append it to your text
- **Shift**: tap once for one capital letter, double-tap for Caps Lock, tap again to turn Caps Lock off
- **123 / ●+= / ABC** switch between letter, number and symbol keyboards
- **Clear** resets the text

## Project Structure

```
Word_Predictor/
├── app.py                  # Flask backend + prediction logic
├── requirements.txt        # Python dependencies
├── word_predictor1.ipynb   # Training notebook (data prep, model, training, export)
├── next_word_model.keras   # Trained LSTM model
├── tokenizer.pkl           # Fitted tokenizer
├── model_config.pkl        # max_sequence_len / vocab_size
├── templates/
│   └── index.html          # Frontend UI
└── docs/
    └── Project_Report.pdf  # Project report
```

## Model Details

- **Architecture**: Embedding (128) → LSTM (128 units) → Dropout (0.2) → Dense (softmax)
- **Training data**: 4,000 shuffled paragraphs from `agentlans/project-gutenberg-top100` (Hugging Face)
- **Vocabulary**: tokenizer limited to the 10,000 most frequent words (`num_words=10000`); rarer words map to `<OOV>`
- **Context window**: last 9 words (`max_sequence_len = 10`)
- **Parameters**: ≈ 5.2 M
- **Loss / optimizer**: sparse categorical cross-entropy, Adam

## How It Works

1. **Input processing**: text is lower-cased and tokenized with the saved tokenizer
2. **Padding**: the sequence is pre-padded / truncated to the last 9 tokens
3. **Prediction**: the LSTM outputs a probability for every word in the vocabulary
4. **Top-K selection**: words are ranked by probability; `<OOV>`, padding and punctuation-only tokens are skipped until 3 real words are found
5. **Display**: suggestions are shown as clickable cards with their probability

## API Endpoint

### POST `/api/predict`

**Request**:

```json
{ "text": "she looked at" }
```

**Response** (example):

```json
{
  "predictions": [
    { "word": "a", "confidence": 44.8 },
    { "word": "the", "confidence": 16.7 },
    { "word": "her", "confidence": 15.1 }
  ]
}
```

`confidence` is the model's softmax probability in percent, so the values never add up to more than 100.
A request without a JSON object, or with a non-string `text`, gets HTTP `400`.

## Troubleshooting

**Model not found** - keep `next_word_model.keras`, `tokenizer.pkl` and `model_config.pkl` in the same folder as `app.py`.

**`ModuleNotFoundError: keras.src.legacy` or errors loading the `.keras` file** - your TensorFlow/Keras is too old. Run `pip install -U -r requirements.txt`.

**Module not found errors** - activate your virtual environment and run `pip install -r requirements.txt` again.

**First prediction is slow** - normal; TensorFlow warms up on the first call.

**GitHub warns that `next_word_model.keras` is larger than 50 MB** - the push still works (GitHub's hard limit is 100 MB). About two-thirds of the file is saved Adam optimizer state, which inference doesn't need.

## Future Enhancements

- [ ] Temperature-based sampling ("creative mode") in the UI
- [ ] Validation split and `val_loss` early stopping during training
- [ ] Size the output layer to the 10,000-word tokenizer vocabulary (smaller, faster model)
- [ ] Support for multiple languages
- [ ] Word history tracking

---

Made using LSTM and Flask
