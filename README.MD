<div align="center">

# Neural Network Plays Chrome Dinosaur 🦖

**Learning to run, jump, and duck from human gameplay.**

A lightweight convolutional neural network that turns game frames into actions.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-3B7A57?style=flat-square)
![Parameters](https://img.shields.io/badge/Parameters-79%2C011-222222?style=flat-square)

[How it works](#how-it-works) · [Data collection](#data-collection) · [The CNN](#the-cnn) · [Get started](#get-started) · [Results](#results)

</div>

---

## The idea

Teach a neural network to play the dinosaur game by showing it how a human plays.

This project uses **behavioral cloning**: collect screenshots and player actions, train a CNN to predict those actions, and connect the trained model back to the game. Everything runs in a standalone Pygame recreation of Chrome’s dinosaur game.

The repository includes the game, a demonstration recorder, model experiments, a training notebook, and saved weights for model-controlled gameplay.

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/Offei-op/Neural_Network_Plays_Chrome_Dinosaur/main/Assets/ReadmeAssets/Architecture.jpg" alt="Game control loop: capture the screen, predict an action with the CNN, map the output to a game action, and update the game state" width="850">
</p>

The model reads the current screen and produces three action logits. The highest-scoring class controls the dinosaur, and the loop repeats with the next game frame.

| Prediction | Action | Human demonstration |
| :---: | :--- | :--- |
| **0** | Continue running | No arrow key pressed |
| **1** | Jump | ↑ Up arrow |
| **2** | Duck | ↓ Down arrow |

Training uses labeled demonstrations and cross-entropy loss. The model receives one frame at a time, with no explicit velocity input or frame history.

## Data collection

<p align="center">
  <img src="https://raw.githubusercontent.com/Offei-op/Neural_Network_Plays_Chrome_Dinosaur/main/Assets/ReadmeAssets/DataCollection.jpg" alt="Human gameplay data collection: screenshots and encoded keyboard inputs are saved together as training data" width="750">
</p>

`collect_data.py` records a screenshot and an action label on every gameplay frame. Images are numbered sequentially; labels are stored as a continuous string of `0`, `1`, and `2` in `inputs.txt`.

Before training, each image is converted to grayscale, resized to **216 × 216**, and scaled to **[0, 1]**. The dataset loader assigns the first 80% of samples to training and the final 20% to validation.

## The CNN

<p align="center">
  <img src="https://raw.githubusercontent.com/Offei-op/Neural_Network_Plays_Chrome_Dinosaur/main/Assets/ReadmeAssets/CNN%20Architecture.jpg" alt="CNN architecture illustration showing convolutional feature maps, pooling, and fully connected layers leading to three actions" width="1100">
</p>

Two convolutional blocks extract visual features, followed by a small classifier that predicts the action.

| Component | Configuration |
| :--- | :--- |
| Input | `1 × 216 × 216` grayscale image |
| First block | 5 × 5 convolution, 16 channels → BatchNorm → ReLU → 2 × 2 max pooling |
| Second block | 5 × 5 convolution, 32 channels → BatchNorm → ReLU → adaptive max pooling to 8 × 8 |
| Classifier | Flatten 2,048 features → Linear(32) → ReLU → Linear(3) |
| Trainable parameters | **79,011** |

*The table follows `model.py`: the second pool produces an 8 × 8 grid; the illustration’s “7 × 7” annotation differs from the implementation.*

## Get started

### Install

```bash
git clone https://github.com/Offei-op/Neural_Network_Plays_Chrome_Dinosaur.git
cd Neural_Network_Plays_Chrome_Dinosaur
python -m venv .venv
```

Activate the environment for your platform:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install numpy
```

Run commands from the repository root. Pygame requires a graphical desktop; the model automatically selects CUDA when available, otherwise CPU.

### Play

```bash
# Control the dinosaur yourself
python main.py

# Let the saved CNN control the dinosaur
python model_gameplay.py
```

Press **any key** at the menu to start or restart. In manual mode, use **↑** to jump and **↓** to duck.

### Collect and train

The action-label file is included, but its matching training images are not. To record a fresh dataset, back up the existing labels and any local images, create an empty `CNN_Training_Data/IMAGES/` folder, and start with an empty `CNN_Training_Data/inputs.txt`.

```bash
python collect_data.py
```

Keep image numbering and labels aligned. The collector removes the last three samples after a collision; finish a run before closing from the menu.

Install the notebook tools and open the training notebook:

```bash
python -m pip install notebook ipywidgets
python -m notebook Train_CNN_Model.ipynb
```

Run the cells in order. Training uses **Adam**, a **0.001 learning rate**, **batch size 64**, and **cross-entropy loss**, for up to **15 epochs** with an early-stop condition. Improvements in validation macro F1 overwrite `model.pth`, so back up that checkpoint before retraining if you want to keep it.

## Results

The latest saved run in [Train_CNN_Model.ipynb](Train_CNN_Model.ipynb) reaches its best validation macro F1 at **epoch 3**:

| Validation accuracy | Validation macro F1 | Training loss at epoch 3 |
| :---: | :---: | :---: |
| **90.09%** | **0.8871** | **0.2060** |

These are recorded notebook results for action classification on demonstration frames. They do not measure autonomous survival time or game score. Evaluation across independent gameplay sessions would provide a stronger test of generalization.

## Repository guide

| File | Purpose |
| :--- | :--- |
| `main.py` | Human-controlled game |
| `collect_data.py` | Record screenshots and action labels |
| `data.py` | Dataset loading and image preprocessing |
| `model.py` | CNN architecture |
| `Train_CNN_Model.ipynb` | Training, validation, and checkpoint selection |
| `Modelling Experimentation.ipynb` | Earlier MLP and CNN experiments |
| `model_gameplay.py` | Connect model predictions to game actions |
| `model.pth` | Saved model weights |
| `Assets/ReadmeAssets/` | Project and model diagrams |

<details>
<summary><strong>Implementation notes</strong></summary>

- **Inference:** `model_gameplay.py` currently omits `model.eval()` and does not scale its float image tensors to `[0, 1]`. Before evaluating gameplay, enable evaluation mode after loading the model and align preprocessing with the grayscale → resize → `ToTensor()` pipeline in `data.py`.
- **Paths:** `data.py` and `collect_data.py` use Windows-style backslashes in some dataset paths. For macOS/Linux, use forward slashes or `pathlib.Path`.
- **Data integrity:** verify that each numbered image has one matching label before training, especially after interrupting collection.
- **Earlier experiments:** the experimentation notebook additionally imports OpenCV (`opencv-python`). Use `Train_CNN_Model.ipynb` for the dedicated training workflow.

</details>

## Credits

**CNN and learning pipeline:** [Ofei Bekoe](https://github.com/Offei-op).

The original Pygame game implementation was credited to **Max Rohowsky** in the repository’s earlier README. See his [Chrome Dinosaur tutorial](https://youtu.be/wnBGG7JLrkg?feature=shared) for the underlying game implementation.
