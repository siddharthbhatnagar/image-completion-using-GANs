# Context-Aware Image Inpainting using Deep Convolutional Generative Adversarial Networks

Context Aware Image Inpainting aims to fill a region in an image based on the context of surrounding pixels. Given a mask, a network is trained to predict the content in the mask. The generated content needs to be well aligned with surrounding pixels and should look realistic based on the context. We plan to use Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic images along with scores of how realistic they are. The DCGANs output can then be used to reconstruct the whole image by solving the optimization problem of a proposed loss function.

## Implementation Details
  
- Datasets:  [Oxford 102 Flower Category Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- Framework: TensorFlow
- DCGAN Architecture: DCGAN model architecture inspired from [Radford et al](https://arxiv.org/pdf/1511.06434.pdf). The input is a noise vector, drawn from some well-known prior distribution (for example: a uniform distribution [−1, 1]), followed by a fully connected layer. This is followed by a series of fractionally-strided convolutions, where the numbers of channels are halved, and image dimension doubles from the previous layer. The output layer is of dimension equal to the size of the image space.
- Performance Measure: Qualitative

<p align="center">
![Screenshot](https://cloud.githubusercontent.com/assets/21965720/24930883/1c40158a-1ed9-11e7-87c3-afded1f927cc.png)
</p>
