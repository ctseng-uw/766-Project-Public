#!/bin/bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt  -f https://nelsonliu.me/files/pytorch/whl/torch_stable.html
mkdir checkpoints dataset dataset_test tmp_imgs ttf