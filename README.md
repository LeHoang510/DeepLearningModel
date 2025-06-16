# DeepLearningModel

Deep learning model using pytorch

- Techniques:
  - Dropout used for Linear layers
  - Batch Normalization used for Convolutional layers
  - Conv -> BatchNorm -> ReLU -> MaxPool2d
  - Conv -> BatchNorm because of creation of many channels with different distributions
  - Linear does not use BatchNorm because the output distribution is not as complex
- Weights init:
  - Convolutional layers: Kaiming Normal
  - Linear layers: Xavier Normal
  - BatchNorm layers: Constant 1, Bias 0
