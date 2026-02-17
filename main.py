import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import os

from nn import TinyNN

CANVAS_SIZE = 300 
BRUSH_SIZE = 10
MODEL_IMG_SIZE = 50
DATA_FILE = "dataset.npz"
MODEL_FILE = "model.npz"


class App:
    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Minimal Drawing App")

        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack(padx=10, pady=10)

        self.status = tk.Label(self.root, text="Draw. Click Train to label (A/B). Click Predict to guess.")
        self.status.pack(padx=10, pady=(0, 10), anchor="w")

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(padx=10, pady=(0, 10), fill="x")

        # --- ML model and dataset ---
        self.model = TinyNN(input_dim=50 * 50, lr=0.25)

        self.X = []  # list of input vectors
        self.y = []  # list of labels (0 = A, 1 = B)
        self.count_A = 0
        self.count_B = 0
        self.trained = False
        
        
        # store last mouse position while drawing
        self.last_x = None
        self.last_y = None

        # bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_down) #LC
        self.canvas.bind("<B1-Motion>", self.on_drag) #MB move
        self.canvas.bind("<ButtonRelease-1>", self.on_up) #RLS LC

        # Backing image (this is what we’ll feed to the model later)
        self.img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)  # "L" = grayscale, 255 = white
        self.draw = ImageDraw.Draw(self.img)
        
        # CLEAR button
        btn = tk.Button(self.root, text="CLEAR", command=self.clear)
        btn.pack(pady=(0, 10))

        #Train and Predict Buttons
        tk.Button(btn_frame, text="Train", command=self.train_model).pack(side="left", expand=True, fill="x", padx=5)
        tk.Button(btn_frame, text="Predict", command=self.predict_clicked).pack(side="left", expand=True, fill="x", padx=5)

        #A and B Buttons
        tk.Button(btn_frame, text="A (save)", command=self.save_as_A).pack(side="left", expand=True, fill="x", padx=5)
        tk.Button(btn_frame, text="B (save)", command=self.save_as_B).pack(side="left", expand=True, fill="x", padx=5)

        #Visualization Buttons
        tk.Button(btn_frame, text="Visualize", command=self.visualize_weights).pack(side="left", expand=True, fill="x", padx=5)
        tk.Button(btn_frame, text="Explain", command=self.visualize_contributions).pack(side="left", expand=True, fill="x", padx=5)

        #DEBUG _DISABLED_
        #tk.Button(self.root, text="Save Debug PNG", command=self.save_debug).pack()
        #tk.Button(self.root, text="Print Vector Info", command=self.debug_vector).pack()
        #tk.Button(self.root, text = "Print Vector Image", command=self.debug_vector_image).pack()
        
        # Auto Load Data
        #self.load_dataset()
        #self.load_model()
        self.startup_prompt()
        self.update_status("Draw something to start")

    def on_down(self, event):
        self.last_x, self.last_y = event.x, event.y

    def on_up(self, event):
        """Mouse released: stop drawing."""
        self.last_x, self.last_y = None, None
    
    def on_drag(self, event):
        """Mouse moving while pressed: draw line segments."""
        if self.last_x is None:
            return

        # This draws on the tk window to see
        self.canvas.create_line(
            self.last_x, self.last_y, event.x, event.y,
            width=BRUSH_SIZE,
            fill="black",
            capstyle=tk.ROUND,
            smooth=True
        )
        # Draw on the PIL image (the real pixel data)
        self.draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill=0,               # 0 = black ink
            width=BRUSH_SIZE
        )
        
        self.last_x, self.last_y = event.x, event.y

    #Clear function, clears both canvases
    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.img)

    # Convert current drawing to a 50x50 flattened vector of floats in [0,1]. (2500 element array)
    def get_x_vector(self) -> np.ndarray:
        small = self.img.resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE), Image.Resampling.BILINEAR)
        arr = np.array(small, dtype=np.float32)
        ink = 1.0 - (arr / 255.0)
        return ink.reshape(-1)
    
    def train_model(self):
        # basic “enough data” checks
        if self.count_A < 2 or self.count_B < 2:
            self.update_status("Need at least 2 samples of A and 2 of B before training")
            return

        Xmat = np.stack(self.X, axis=0)
        yvec = np.array(self.y, dtype=np.float32)

        self.model.fit(Xmat, yvec, epochs=75)
        self.trained = True
        self.save_model()
        self.update_status("Training complete")

    def predict_clicked(self):
        if not self.trained:
            self.update_status("Train the model first")
            return

        x = self.get_x_vector()
        if float(x.sum()) < 1e-3:
            self.update_status("No ink on canvas (draw first)")
            return

        p = self.model.predict_proba(x)
        if p >= 0.5:
            label = "B"
            conf = p
        else:
            label = "A"
            conf = 1.0 - p

        self.update_status(f"Prediction: {label} (confidence {conf:.2f})")
    
    def save_as_A(self):
        x = self.get_x_vector()
        if float(x.sum()) < 1e-3:
            self.update_status("No ink on canvas (draw first)")
            return

        self.X.append(x)
        self.y.append(0.0)
        self.count_A += 1
        self.trained = False  # data changed, model is now “out of date”
        self.save_dataset()
        self.clear()
        self.update_status("Saved as A")

    def save_as_B(self):
        x = self.get_x_vector()
        if float(x.sum()) < 1e-3:
            self.update_status("No ink on canvas (draw first)")
            return

        self.X.append(x)
        self.y.append(1.0)
        self.count_B += 1
        self.trained = False
        self.save_dataset()
        self.clear()
        self.update_status("Saved as B")

    def update_status(self, extra=""):
        total = len(self.y)
        msg = f"Samples — A: {self.count_A} | B: {self.count_B} | Total: {total}"
        if self.trained:
            msg += " | Model: trained"
        else:
            msg += " | Model: not trained"
        if extra:
            msg += f" | {extra}"
        self.status.config(text=msg)

    # Visualization code converts wheights into pixel values then displays back as tk format
    def visualize_weights(self):
        if not getattr(self, "trained", False):
            self.update_status("Train the model first to visualize weights")
            return

        w = self.model.w  # shape (2500,)
        W = w.reshape(50, 50).astype(np.float32)

        # Normalize weights to 0..255 for display (centered around 0)
        # We’ll map negative -> dark, positive -> bright, zero -> mid-gray.
        max_abs = float(np.max(np.abs(W))) + 1e-8
        norm = (W / max_abs)  # now roughly in [-1, 1]

        # Convert to 0..255 where 128 is 0
        img_arr = (128 + 127 * norm).clip(0, 255).astype(np.uint8)

        # Make it bigger so it’s visible
        img = Image.fromarray(img_arr, mode="L").resize((300, 300), Image.Resampling.NEAREST)

        # Show in a new window
        win = tk.Toplevel(self.root)
        win.title("Model Weights (50x50) — dark=A, bright=B")

        tk.Label(win, text="Dark pixels push toward A; bright pixels push toward B.").pack(pady=(10, 5))

        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo)
        lbl.image = photo  # keep reference so it doesn't get garbage-collected
        lbl.pack(padx=10, pady=10)
        
    
    def visualize_contributions(self):
        if not getattr(self, "trained", False):
            self.update_status("Train first, then Explain")
            return

        # Get current drawing x (2500,)
        x = self.get_x_vector().astype(np.float32)

        # Contributions c = w * x (2500,)
        c = (self.model.w.astype(np.float32) * x)
        C = c.reshape(50, 50)

        # Normalize around 0 so negatives/dark and positives/bright are comparable
        max_abs = float(np.max(np.abs(C))) + 1e-8
        norm = C / max_abs  # ~[-1, 1]

        # Map: -1 -> dark, 0 -> gray, +1 -> bright
        img_arr = (128 + 127 * norm).clip(0, 255).astype(np.uint8)

        # Make it visible
        img = Image.fromarray(img_arr, mode="L").resize((300, 300), Image.Resampling.NEAREST)

        # Show in a new window
        win = tk.Toplevel(self.root)
        win.title("Explanation (w * x) — dark=A, bright=B")

        tk.Label(
            win,
            text="This shows per-pixel contribution: dark pushes A, bright pushes B."
        ).pack(pady=(10, 5))

        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo)
        lbl.image = photo
        lbl.pack(padx=10, pady=10)

        # Optional: also show the numeric score components
        z = float(np.dot(self.model.w, x) + self.model.b)
        p = float(self.model.predict_proba(x))
        tk.Label(win, text=f"z = {z:.3f}   p(B) = {p:.3f}").pack(pady=(0, 10))
    
    def save_dataset(self):
        if len(self.y) == 0:
            return
        Xmat = np.stack(self.X, axis=0).astype(np.float32)
        yvec = np.array(self.y, dtype=np.float32)
        np.savez(DATA_FILE, X=Xmat, y=yvec)  
        
    def load_dataset(self):
        if not os.path.exists(DATA_FILE):
            self.update_status("No saved dataset found")
            return

        data = np.load(DATA_FILE)
        Xmat = data["X"]
        yvec = data["y"]

        self.X = [Xmat[i].copy() for i in range(Xmat.shape[0])]
        self.y = [float(v) for v in yvec]

        self.count_A = int(np.sum(yvec == 0))
        self.count_B = int(np.sum(yvec == 1))
        self.trained = False
        self.update_status("Loaded dataset from disk")
    
    def save_model(self):
        np.savez(MODEL_FILE, w=self.model.w.astype(np.float32), b=np.float32(self.model.b))
    
    def load_model(self):
        if not os.path.exists(MODEL_FILE):
            self.update_status("No saved model found")
            return

        data = np.load(MODEL_FILE)
        self.model.w = data["w"].astype(np.float32)
        self.model.b = float(data["b"])
        self.trained = True
        self.update_status("Loaded trained model")

    def startup_prompt(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Startup Options")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(
            dlg,
            text="How would you like to start?",
            padx=20,
            pady=15
        ).pack()

        def load_previous():
            self.load_dataset()
            self.load_model()
            dlg.destroy()

        def start_fresh():
            self.update_status("Started fresh (no data loaded)")
            dlg.destroy()

        btns = tk.Frame(dlg, padx=15, pady=15)
        btns.pack()

        tk.Button(btns, text="Load Saved Data", width=20, command=load_previous)\
            .pack(side="left", padx=8)

        tk.Button(btns, text="Start Fresh", width=20, command=start_fresh)\
            .pack(side="left", padx=8)

        dlg.protocol("WM_DELETE_WINDOW", start_fresh)
        self.root.wait_window(dlg)


    #DEBUG 
    '''
    def save_debug(self):
        self.img.save("debug.png")
        print("Saved debug.png")

    def debug_vector(self):
        x = self.get_x_vector()
        print("x shape:", x.shape)
        print("min/max:", float(x.min()), float(x.max()))
        print("sum ink:", float(x.sum()))
        x = self.get_x_vector()

        print("=== VECTOR DEBUG ===")
        print("Shape:", x.shape)
        print("Min / Max:", float(x.min()), float(x.max()))
        print("Sum (total ink):", float(x.sum()))

        print("\nFirst 100 values:")
        print(x[:100])

        print("\nLast 100 values:")
        print(x[-100:])
    def debug_vector_image(self):
        x = self.get_x_vector()
        img50 = x.reshape(50, 50)

        np.set_printoptions(precision=2, suppress=True)
        print(img50) 
        '''

    def run(self):
        self.root.mainloop()


    
    
if __name__ == "__main__":
    App().run()