# Scalable Label-efficient Footpath Network Generation Using Remote Sensing Data and Self-supervised Learning

PyTorch implementation and models for footpath segmentation and GIS layer generation. For details, please see <a href="https://arxiv.org/abs/2309.09446">our paper</a>.

Our pipeline can be divided into two main parts, footpath segmentation and GIS layer generation.

The first part contains two training processes involving the construction of two models, self-supervised pre-training (DINO-MC) and footpath segmentation fine-tuning (U-Net).

The code of the self-supervised pre-training can be found at <a href='https://github.com/WennyXY/DINO-MC'>DINO-MC</a>.



We provide code of the second part which converts the model output mask to a GIS layer. 
The implementation references the <a href='https://github.com/VIDA-NYU/tile2net'> Tile2Net</a>.

