# GPT-NeoX-20B with Hugging Face Accelerate

An exeriment, using Hugging Face's [Accelerate](https://huggingface.co/docs/accelerate/big_modeling) library to run inference with Eleuther's [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) 20B model on a system without enough GPU or CPU RAM to hold 20 billion parameters.

It works, but is very slow: 30 minutes to predict 100 tokens.

## About GPT-NeoX-20B

GPT-NeoX-20B is a 20 billion parameter autoregressive language model developed by [EleutherAI](https://www.eleuther.ai/) with the support of [CoreWeave](https://www.coreweave.com/) and trained on [the Pile](https://arxiv.org/abs/2101.00027). Technical details about GPT-NeoX-20B can be found in [the associated paper](https://arxiv.org/abs/2204.06745). 

### The Weights

This code works with the [slim weights (39GB)](https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/). 

On Linux, to download from the command line to a folder named `20B_checkpoints`, use the following command:

```bash
wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/ -P 20B_checkpoints
```

Alternatively, weights can be downloaded using a BitTorrent client from [with this torrent link](https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights.torrent).


## Setup

### Installation

First, install PyTorch with the latest CUDA version for your platform from [pytorch.org](https://pytorch.org/get-started/locally/).

Then install other required Python libraries with pip:

```
pip install -r requirements.txt
```


## Disclaimer: Limitations and Biases

(This section was copied from [here](https://huggingface.co/EleutherAI/gpt-j-6B).)

The core functionality of GPT-NeoX is taking a string of text and predicting the next token. While language models are widely used for tasks other than this, there are a lot of unknowns with this work. When prompting GPT-NeoX it is important to remember that the statistically most likely next token is often not the token that produces the most "accurate" text. Never depend upon GPT-NeoX to produce factually accurate output.

GPT-NeoX was trained on the Pile, a dataset known to contain profanity, lewd, and otherwise abrasive language. Depending upon use case GPT-NeoX may produce socially unacceptable text. See Sections 5 and 6 of [the Pile paper](https://arxiv.org/abs/2101.00027) for a more detailed analysis of the biases in the Pile.

As with all language models, it is hard to predict in advance how GPT-NeoX will respond to particular prompts and offensive content may occur without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results.
