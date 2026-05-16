# IDCNet: Image Decomposition and Cross-View Distillation for Generalizable Deepfake Detection
![alt text](./assets/image.png)

Code reference for the paper "IDCNet: Image Decomposition and Cross-View Distillation for Generalizable Deepfake Detection".

## Environment and code instructions
### install
1. The Python environment is managed using `uv`. Refer to [uv official](https://docs.astral.sh/uv/#highlights) for installation instructions for the `uv` tool.
2. Run the command `uv sync` to install dependencies.
3. `exp_scripts/main.py` is the entry file, and we provide a simple training and testing function.
4. Our code cannot be run directly, it only provides a reference implementation.
### checkpoints
The relevant weights can be downloaded [here](https://drive.google.com/drive/folders/1v2Xnmr7Z0phuOihY9UobDDFOD9r85imD?usp=drive_link).
## reference
```bibtex
@ARTICLE{11098842,
  author={Wang, Zhiyuan and Chen, Yanxiang and Yao, Yuanzhi and Han, Meng and Xing, Wenpeng and Li, Meng},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={IDCNet: Image Decomposition and Cross-View Distillation for Generalizable Deepfake Detection}, 
  year={2025},
  volume={20},
  number={},
  pages={8373-8386},
  keywords={Deepfakes;Forgery;Feature extraction;Faces;Image decomposition;Training;Generators;Facial animation;Detectors;Knowledge transfer;Face forgery detection;deepfake detection;multi-view learning;representation disentanglement;mutual information},
  doi={10.1109/TIFS.2025.3593353}}
```
