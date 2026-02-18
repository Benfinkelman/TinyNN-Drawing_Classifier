# TinyNN Drawing Clasifier

A minimal drawing app that uses a tiny 1-neuron neural network for **binary classification** (A vs B). Draw on the canvas, label examples as A or B, train the model, then get predictions. Includes rejection rules for invalid drawings and visualizations of model weights and per-pixel contributions.

## Features

- **Draw** on a 300×300 canvas (brush size 10)
- **Label** drawings as **A** or **B** and save them to an npz dataset
- **Train** a simple logistic regression–style model (single neuron, sigmoid, gradient descent)
- **Predict** whether the current drawing is A or B (with confidence)
- **Visualize** learned weights (dark = favors A, bright = favors B)
- **Explain** per-pixel contributions for the current drawing
- **Rejection rules** prevent saving or predicting on invalid inputs (too little/too much ink, too small, line-like shapes)

Data and model are saved to `dataset.npz` and `model.npz` and can be loaded on startup.


## Neural Network Explained

This project uses a single-neuron neural network, equivalent to logistic regression, for binary classification between classes A and B. While simple, this model is fully trainable, interpretable, and implemented entirely from scratch.

### Input Representation

Each drawing is converted from pixel values on PIL into a **50×50 grayscale image**, then flattened into a vector:

x ∈ ℝ²⁵⁰⁰

Each element of `x` represents how much ink is present at a specific pixel location (values between 0 and 1).

---

### Model Parameters

The network consists of:

- **Weights**  
  w ∈ ℝ²⁵⁰⁰  
  One weight per pixel, representing how important that pixel is for distinguishing A vs B.

- **Bias**  
  b ∈ ℝ  
  A scalar offset that shifts the decision boundary.

---

### Forward Pass (Prediction)

Given an input drawing `x`, the model computes:

1. **Weighted sum**  
   z = w · x + b

2. **Sigmoid activation**  
   p = σ(z) = 1 / (1 + e⁻ᶻ)

The output `p` is interpreted as:

- p = P(class = B)
- If p ≥ 0.5 → predict **B**
- If p < 0.5 → predict **A**

This value is also used as a **confidence score**.

---

### Training (Learning)

Training adjusts `w` and `b` so that predictions match labeled examples.

For each labeled drawing `(x, y)`:
- y = 0 for A  
- y = 1 for B  

The model minimizes **binary cross-entropy loss** (implicitly):

L(y, p) = −[ y log(p) + (1 − y) log(1 − p) ]

For the combination of **sigmoid activation + binary cross-entropy**, the gradient simplifies to:

∂L / ∂z = p − y

This leads to the update rules:

- **Weight update**  
  w ← w − α (p − y) x

- **Bias update**  
  b ← b − α (p − y)

where α is the learning rate.

Training consists of repeating this process over all saved samples for multiple epochs.

---

### Interpretation of Weights

After training:

- **Positive weights**  
  Ink at that pixel pushes the prediction toward **B**

- **Negative weights**  
  Ink at that pixel pushes the prediction toward **A**

- **Near-zero weights**  
  Pixel has little effect on the decision

The **Visualize** feature reshapes `w` back into a 50×50 image so this learned structure can be inspected directly.

---

### Per-Pixel Explanation

For a given drawing, the model computes per-pixel contributions:

cᵢ = wᵢ · xᵢ

Reshaping this into 50×50 shows **which parts of the drawing influenced the prediction**:
- Bright → pushes toward B
- Dark → pushes toward A
- Gray → neutral

This makes the model’s decision process transparent.

---

### Limitations (By Design)

Because the model is **linear**, it does not understand shape, topology, or stroke connectivity. It reasons only about **where ink appears**, not how pixels relate to one another. Rejection rules and confidence thresholds are used to prevent invalid or ambiguous inputs from being force-classified.


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
