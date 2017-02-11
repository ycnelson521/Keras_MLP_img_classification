This implemented a multi-layer perceptron(MLP) to classiffy images.
The MLP is made of 4 fully connected layers.
Layer configuration is 200-128-128-128.
The training images of the same class should be basicly similar in view.
This MLP may be able to adapt to lightning variation and some outliers (non-scene objects).

A pretrained weight is prepared.
However, a new classification task with different images require addtional training or re-train.

To prepare the training data, put images of the same class into a directory under /train.
To prepare the testing data, likewise, put images of the same class into a directory under /test.
Modifying the path to training and testing directory enables running different training/testing.

The program will automatically traverse images in different class (label) directories.
All image will be loaded into memory.
Some computers may encounter “memory error" due to excessive memory usage.
(This should be modified for more efficient memory usage)
