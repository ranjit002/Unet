# Unet
Implementation of a deep convolutional network (Unet) for binary segmentation

The pictures below come from a model trained completely on my laptop!
These pictures are all part of the testing dataset, so the model hasn't seen them before.

The model was trained using a loss function that deals with the class imbalance.

 The specific loss used (DiceLoss in the architecture.py file) leads to the model only classifying a pixels as part of a seep if it's confident, so it tends to underpredict pixels as part of a seep
 However it predicts the overall seep's shape quite well.

![alt text](https://github.com/ranjit002/Unet/blob/main/imgs/comparison3.png?raw=true)

The evenly coloured patches of pixels are missing pixels in the original image.
![alt text](https://github.com/ranjit002/Unet/blob/main/imgs/comparison2.png?raw=true)

![alt text](https://github.com/ranjit002/Unet/blob/main/imgs/comparison1.png?raw=true)
