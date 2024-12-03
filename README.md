# llm_universal

We show that autoregressive decoding of a transformer-based language model can
realize universal computation, without external intervention or modification of the model’s weights. Establishing this result requires understanding how a language model can process arbitrarily long inputs using a bounded context. For this purpose, we consider a generalization of autoregressive decoding where, given a long input, emitted tokens are appended to the end of the sequence as the context window advances. We first show that the resulting system corresponds to a classical model of computation, a Lag system, that has long been known to be computationally universal. By leveraging a new proof, we show that a universal Turing machine can be simulated by a Lag system with 2027 production rules. We then investigate whether an existing large language model can simulate the behaviour of such a universal Lag system. We give an affirmative answer by showing that a single system-prompt can be developed for gemini-1.5-pro-001 that drives the model, under deterministic (greedy) decoding, to correctly apply each of the 2027 production rules. We conclude that, by the Church-Turing thesis, prompted gemini-1.5-pro-001 with extended autoregressive (greedy)
decoding is a general purpose computer.

[[arxiv]](https://arxiv.org/pdf/2410.03170)

## Setup

At the project root folder, simply do:

    pip install -e .

Also in your bashrc, setup the following env vars

    export GCP_PROJECT_ID=your_gcp_project_id
    export GCP_LOCATION=your_gcp_project_location

so that you can use the vertex gemini api.

You can verify if your gemini api is setup correctly by doing

    python utm/common/llm.py

## Usage

Navigate to `utm/` and run:

    python main_wam.py

It then run the verification of the rules we induced against gemini-1.5-pro-001.


## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```latex
@article{schuurmans2024autoregressive,
  title={Autoregressive Large Language Models are Computationally Universal},
  author={Schuurmans, Dale and Dai, Hanjun and Zanini, Francesco},
  journal={arXiv preprint arXiv:2410.03170},
  year={2024}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
