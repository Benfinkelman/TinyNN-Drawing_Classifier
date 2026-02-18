# Drawing NN BB

A minimal drawing app that uses a tiny 1-neuron neural network for **binary classification** (A vs B). Draw on the canvas, label examples as A or B, train the model, then get predictions. Includes rejection rules for invalid drawings and visualizations of model weights and per-pixel contributions.

## Features

- **Draw** on a 300×300 canvas (brush size 10)
- **Label** drawings as **A** or **B** and save them to a npz dataset
- **Train** a simple logistic regression–style model (single neuron, sigmoid, gradient descent)
- **Predict** whether the current drawing is A or B (with confidence)
- **Visualize** learned weights (dark = favors A, bright = favors B)
- **Explain** per-pixel contributions for the current drawing
- **Rejection rules** prevent saving or predicting on invalid inputs (too little/too much ink, too small, line-like shapes)

Data and model are saved to `dataset.npz` and `model.npz` and can be loaded on startup.

## Requirements

- Python 3.10+
- NumPy
- Pillow (PIL)
- tkinter (usually included with Python)

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd "Drawing NN BB"
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the app:

```bash
python main.py
```

- **Draw** something, then click **A (save)** or **B (save)** to add that drawing to the dataset.
- You need at least **2 samples of A** and **2 of B** before **Train** is enabled.
- After training, use **Predict** to classify the current drawing (A vs B). Low-confidence predictions are reported as "Not sure."
- **Visualize** shows the learned 50×50 weight map (dark → A, bright → B).
- **Explain** shows per-pixel contribution (w * x) for the current drawing.
- **CLEAR** clears the canvas.

On startup you can choose **Load Saved Data** to load pretrained `dataset.npz` and `model.npz`, or **Start Fresh**.

## Project structure

| File               | Description |
|--------------------|-------------|
| `main.py`          | Tkinter UI, canvas, train/predict flow, save/load dataset and model |
| `nn.py`            | `TinyNN`: 1-neuron binary classifier (sigmoid, gradient descent) |
| `rejection_rules.py` | Ink stats and `reject_reason()` for validating drawings |
| `dataset.npz`      | Saved training data (created when you save samples) |
| `model.npz`        | Saved weights and bias (created when you train) |

## License

Use and modify as you like.
