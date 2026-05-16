#!/usr/bin/env bash
set -e

# python Grad_CAM_v3.py --num-per-class 300 --only-model xception --save-only-misclassified
# python Grad_CAM_v3.py --num-per-class 100 --only-model f3net_fad
python Grad_CAM_v3.py --num-per-class 4000 --only-model m2tr --save-only-misclassified --datasets WildDeepfake --pred-fake-threshold 0.6 --labels fake 
# python Grad_CAM_v3.py --num-per-class 300 --only-model ours --save-only-misclassified