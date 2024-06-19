# Attention is All You Need

This repository contains the implementation using Pytorch of the seminal paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. This implementation includes the Transformer model, tokenizer, and training scripts.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Tokenizer](#tokenizer)
  - [Training the Transformer](#training-the-transformer)
- [Examples](#examples)
- [References](#references)

## Overview

The Transformer model, introduced in the paper "Attention is All You Need," is a novel architecture designed to handle sequential data with self-attention mechanisms. This architecture has achieved state-of-the-art performance in various natural language processing tasks.

## Requirements

- Python 3.11 or higher
- PyTorch 2.3 or higher
- NumPy
- transformers 4.41 or higher

## Installation

To set up the environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/deeplearningcafe/transformer-pytorch.git
   cd transformer-pytorch
   ```
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Or if using conda:
   ```bash
   conda create -n transformer_torch
   conda activate transformer_torch
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Train tokenizer
We use the BPE tokenizer, the same as the original paper. To train the tokenizer from scratch we use the  `tokenizers` library from huggingface. The `vocabulary_size` is set to 37000.
   ```bash
   python train_tokenizer.py
   ```

### Train transformer
To train the Transformer model, use the provided script:
   ```bash
   python training.py
   ```
Here, `config.yaml` is a configuration file specifying the model parameters, training settings, and dataset paths. Parameter count becomes 56,754,176, while try to use the same configuration as in the paper.
   ```yalm
   transformer:
   hidden_dim: 512
   num_heads: 8
   intermediate_dim: 2048
   eps: 1e-06
   num_layers: 6
   dropout: 0.1
   label_smoothing: 0.1

   train:
   warmup_steps: 4000
   max_length: 128
   device: cpu
   train_batch: 128
   eval_batch: 128
   steps: 10
   val_steps: 1
   log_steps: 1
   save_steps: 1
   use_bitsandbytes: False
   save_path: weights/transformer_
   dataset_name: "bentrevett/multi30k"
   ```

## Examples
We include the `inference.py` file for inference that calls the `generate` method from the transformer class, with option to use greedy sampling or top_k sampling.
To evaluate the model on the test set we include `test_model.py` file.

## References
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is All You Need. arXiv preprint [arXiv:1706.03762.](https://arxiv.org/abs/1706.03762)
- Hugging Face Transformers: https://huggingface.co/transformers/

## Author
[aipracticecafe](https://github.com/deeplearningcafe)

## License

This project is licensed under the MIT license. Details are in the [LICENSE](LICENSE.txt) file.
