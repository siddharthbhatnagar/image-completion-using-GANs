# Context-Aware Image Inpainting using Deep Convolutional Generative Adversarial Networks

Context Aware Image Inpainting aims to fill a region in an image based on the context of surrounding pixels. Given a mask, a network is trained to predict the content in the mask. The generated content needs to be well aligned with surrounding pixels and should look realistic based on the context. We plan to use Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic images along with scores of how realistic they are. The DCGANs output can then be used to reconstruct the whole image by solving the optimization problem of a proposed loss function.

## Implementation Details
  
- Dataset:  [Oxford 102 Flower Category Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | 
            [Download Link](https://www.dropbox.com/s/2n7qd2qq39hsmnw/jpg2.zip?dl=0)
- Framework: TensorFlow
- DCGAN Architecture: DCGAN model architecture inspired from [Radford et al](https://arxiv.org/pdf/1511.06434.pdf). The input is a noise vector, drawn from some well-known prior distribution (for example: a uniform distribution [−1, 1]), followed by a fully connected layer. This is followed by a series of fractionally-strided convolutions, where the numbers of channels are halved, and image dimension doubles from the previous layer. The output layer is of dimension equal to the size of the image space.
- Performance Measure: Qualitative

![Screenshot](https://cloud.githubusercontent.com/assets/21965720/25057753/9470978c-2140-11e7-9a51-ec6ee19754bc.png)


## Some Fun GIFs of our results - 

These GIFs show our training process, from the first epoch to the end of training, for a given set of noise points

![alt text](https://raw.githubusercontent.com/siddharthbhatnagar/image-completion-using-GANs/master/face_completion.gif)

![alt text](https://raw.githubusercontent.com/siddharthbhatnagar/image-completion-using-GANs/master/flower_completion.gif)

## Authors:

[Aashima Arora](https://github.com/aashima-arora239), [Nishant Puri](https://github.com/nishant-puri), [Siddharth Bhatnagar](https://github.com/siddharthbhatnagar)


This is the course project for COMS 4995 - Deep Learning for Computer Vision

## References

* Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014
* Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 2015
* Salimans, Tim, et al. "Improved techniques for training gans." Advances in Neural Information Processing Systems. 2016
* Yeh, Raymond, et al. "Semantic Image Inpainting with Perceptual and Contextual Losses." arXiv preprint arXiv:1607.07539 2016
* Chen, Xi, et al. "Infogan: Interpretable representation learning by information maximizing generative adversarial nets." Advances in Neural Information Processing Systems. 2016
* Denton, Emily L., Soumith Chintala, and Rob Fergus. "Deep Generative Image Models using a￼ Laplacian Pyramid of Adversarial Networks." Advances in neural information processing systems. 2015
* https://github.com/jacobgil/keras-dcgan
* http://bamos.github.io/2016/08/09/deep-completion/
* https://github.com/soumith/dcgan.torch
* https://github.com/carpedm20/DCGAN-tensorflow
* https://github.com/rajathkumarmp/DCGAN
