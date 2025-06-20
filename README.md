# DeepLearningModel

Deep learning model using pytorch

- Techniques:
  - Dropout usually used for Linear layers
  - Dropout randomly zeroes individual activations, which can destroy important spatial features in images => don't use Dropout in Convolutional layers
  - Batch Normalization used for Convolutional layers
  - Conv -> BatchNorm -> ReLU -> MaxPool2d
  - Conv -> BatchNorm because of creation of many channels with different distributions
  - Linear does not use BatchNorm because the output distribution is not as complex
- Weights init:
  - Convolutional layers: Kaiming Normal
  - Linear layers: Xavier Uniform
  - BatchNorm layers: Constant 1, Bias 0

- In GAN:
  - Generator use ReLU while Discriminator use LeakyReLU
