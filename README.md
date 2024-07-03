# DiCTI - Diffusion-based Clothing Designer via Text-guided Input

## Requirements (Python >= 3.7):

* pytorch, torchvision
* diffusers
* transformers
* detectron2-densepose (for densepose predictions)
* pillow (PIL)

Install requirements with:

```
pip install torch torchvision torchaudio
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
pip install -r requirements.txt
```

_Note:_ pytorch (torch) may require separate installation based on GPU / CUDA compatibility. Refer to [Pytorch](https://pytorch.org/get-started/locally/)

_Note:_ some requirements get installed as dependencies to ensure version compatibility (numpy, pillow, opencv, ...).

## Precompute DensePose

1. Prepare model files (choose one of the two options)
   * Download pretrained weights [densepose_rcnn_R_101_FPN_s1x.pkl](https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl) to `weigths` folder.
   * Pick a model from [DensePose Model ZOO](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_IUV.md#ModelZoo). Download the `.pkl` file to `weights` and `.yaml` to `configs`.
2. Set input and output path in `scripts/run_densepose.sh`.
3. Run `scripts/run_densepose.sh`.

## Run DiCTI

```
python scripts/pipeline.py
```

## TODO:

1. Add DensePose computation to inference script
2. Improve image blending/face restoration
3. Improve masking
4. Web demo (or colab) maybe?

