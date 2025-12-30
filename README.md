# tiny-diffusion

A character-level language diffusion model for text generation. The model is a modified version of the [nanochat gpt](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
) implementation and is trained on Tiny Shakespeare! It is only 10.7 million parameters, so you can try it out locally!


## Installation

```bash
# Clone the repository
git clone <repository-url>
cd tiny-diffusion

# Install dependencies (Python 3.10+)
uv sync

# Download the dataset
wget https://github.com/nathan-barry/tiny-diffusion/releases/download/v2.0.0/data.txt
```


## Quick Start

The file `training.py` puts the weights in `weights/diffusion_model.pt`. The sample and animation files load the model from this file.

### Train Your Own Model
```bash
# Train from scratch on Shakespeare
uv run training.py

# Training will save checkpoints to weights/diffusion_model.pt
```

### Generate Text
To generate a continuous stream of output (currently 10 context lengths), run:

```bash
# Generate samples using the pre-trained model
uv run sample.py
```

### Visualize the Diffusion Process
To see the diffusion process as a nice animation, run:

```bash
# Watch the denoising process step-by-step
uv run animations/diffusion-process.py

# See Game of Life-inspired sampling (fun little experiment)
uv run animations/game-of-life.py
```


## Default Config

- **Parameters**: 10.7 million
- **Layers**: 6
- **Attention Heads**: 6
- **Embedding Dim**: 384
- **Sequence Length**: 256 characters
- **Diffusion Steps**: 128


## File Structure

```
tiny-diffusion/
├── model.py                    # Core diffusion transformer
├── training.py                 # Training script
├── sample.py                   # Text generation
├── data/
│   └── tiny_shakespeare.txt    # Training data
├── weights/
│   └── diffusion_model.pt      # Pre-trained weights
└── animations/
    ├── diffusion-process.py    # Denoising visualization
    └── game-of-life.py         # Game of Life sampling
```


## License
MIT
