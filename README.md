# LLaMA 3

A single-file implementation of [LLaMA 3](https://arxiv.org/abs/2407.21783), with support for jitting, KV caching and prompting.

The original implementation can be found at https://github.com/meta-llama/llama3.

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

### Using Pip

First of all, install [Python 3.8 or later](https://www.python.org). Open a terminal and run:

```bash
pip install git+https://github.com/lucadellalib/llama3@main#egg=llama3[all]
```

### From Source

First of all, install [Python 3.8 or later](https://www.python.org).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a terminal and run:

```bash
# Install the package locally in editable mode
pip install -e .[all]
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

### Importing the Model in Your Own Script

```python
import torch
from llama3 import LlamaDecoder

B, H, K = 3, 512, 30
model = LlamaDecoder(K)
print(model)

# Process 50 timesteps
input = torch.randn(B, 50, H)
output, state = model(input)
print(output.shape)

# Process 2 additional timesteps
input = torch.randn(B, 2, H)
output, state = model(input, state=state)
print(output.shape)

# JIT the model
model_jit = model.jit()
output_jit, state_jit = model_jit(input)
print(output.shape)
```

### Inference Example With Pretrained Checkpoint

First of all, download the model weights and tokenizer (pretrained variant, e.g. Llama3.2-1B). Check the official
website for instructions on how to [download the models](https://github.com/meta-llama/llama3#download).

Navigate to `<path-to-repository>`, open a terminal and run:

```bash
python main.py --checkpoint_path <path-to-checkpoint>
```

It is recommended to run this script on a machine with at least 1 GPU.

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------