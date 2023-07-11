# lilGym Baselines

This repository contains example code for training the baselines of the paper [*lil*Gym: Natural Language Visual Reasoning with Reinforcement Learning](https://aclanthology.org/2023.acl-long.512/). Trained models on Zenodo: [link](https://zenodo.org/record/8128780).

[paper](https://aclanthology.org/2023.acl-long.512/) | TL;DR [tweet](https://twitter.com/yoavartzi/status/1605400521816346624) | [env code & data](https://github.com/lil-lab/lilgym) | [website](lil.nlp.cornell.edu/lilgym)

## Installation

Note: this code has been tested with PyTorch 1.12.1 and CUDA 11.2.

1. Install [lilgym](https://github.com/lil-lab/lilgym) and the dependencies by following the [installation instructions](https://github.com/lil-lab/lilgym#installation).

It also includes the installation of PyTorch.

2. Clone the current repo.

3. Install Python dependencies:

```
cd /path/to/lilgym-baselines
pip install -r requirements.txt
```

## Training

### Example of training commands

Training a C3+BERT model with PPO+SF on the TowerScratch environment:

```
python main.py --env-name TowerScratch-v0 --env-opt tower --learn-opt scratch --algo ppo --stop-forcing  --seed 1 --model c3bert --text-feat bertfix --num-processes 1 --num-steps 2048 --lr 3e-4 --entropy-coef 0.1 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 4000000 --use-gae --optim-type adam --scheduler linear --warmup-percent 0 --log-interval 1 --eval-interval 10 --log-dir ${path} --save-dir ${path} --save-interval 20 --wandb --wandb-run-name name-of-the-run
```

Training a ViLT model with PPO on the TowerFlipIt environment:

```
python main.py --env-name TowerFlipIt-v0 --env-opt tower --learn-opt flipit --algo ppo --stop-forcing  --seed 1 --model vilt --num-processes 1 --num-steps 2048 --lr 3e-5 --entropy-coef 0.1 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 4000000 --use-gae --optim-type adamw --scheduler cosine --warmup-percent 0.01 --log-interval 1 --eval-interval 10 --log-dir ${path} --save-dir ${path} --save-interval 20 --wandb --wandb-run-name name-of-the-run
```

## Acknoledgements

The RL code is based on [Kostrikov, 2018](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). We thank the authors for open-sourcing their code.
