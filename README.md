# Deep Learning Notebooks

A curated collection of hands-on deep learning notebooks implemented in PyTorch.  
Each notebook focuses on a specific concept, from fundamental building blocks like autograd and broadcasting to full-fledged tasks like transfer learning, custom layers, and sequence modeling with RNNs.

This repository is ideal for students, researchers, and practitioners looking to deepen their understanding of deep learning by building key components from scratch and experimenting with real-world datasets like MNIST and CIFAR-10.

## 📓 Notebooks Overview

### `1_basic_classifiers.ipynb` — Digit Classification with MNIST  
Learn how to classify handwritten digits using the MNIST dataset. This notebook covers data preprocessing and applies basic computational learning techniques for digit recognition, with practical applications in document automation, banking, and postal systems.

<p align="center"><img width="844" height="464" alt="image" src="https://github.com/user-attachments/assets/ff6cb203-f188-449f-a7b7-5037e9f6534f" /></p>


### `2_broadcast_tensors.ipynb` — Broadcasting Mechanics  
Recreate the behavior of `expand_as` and `broadcast_tensors` in PyTorch. This notebook demystifies tensor broadcasting by mimicking these essential utilities.

<p align="center"><img width="603" height="194" alt="image" src="https://github.com/user-attachments/assets/c7372c6e-19f4-4883-b9ef-7300137c8ba7" /></p>

### `3_autograd.ipynb` — Building Autograd and Multinomial  
Implement a simplified version of PyTorch’s autograd system and the multinomial sampling function. Understand how gradients are computed under the hood.

<p align="center"><img width="620" height="370" alt="image" src="https://github.com/user-attachments/assets/7cf56a17-e5d8-48eb-855a-75d1ee4e68fe" /></p>

### `4_deciles_vs_percentiles.ipynb` — Tabular Learning and Data Loading  
Define a flexible `DataLoader` class for tabular datasets and use it to train multiple neural network models. Learn how deciles and percentiles affect class labeling in regression-to-classification tasks.

<p align="center"><img width="901" height="314" alt="image" src="https://github.com/user-attachments/assets/5dcfde32-aa62-41e4-b377-52c89fc89714" /></p>

### `5_splitlinear_dropnorm.ipynb` — Custom Layers: SplitLinear and DropNorm  
Implement two original PyTorch layers:  
- `SplitLinear`: Splits the input, applies shared transformations, and recombines the output.  
- `DropNorm`: A combined Dropout + Normalization module.  
Evaluate their performance against standard PyTorch layers.

<p align="center"><img width="367" height="51" alt="image" src="https://github.com/user-attachments/assets/dbd2a722-6624-416a-a21a-2c127d972d1f" /></p>

### `6_conv2d_transfer_learning.ipynb` — Conv2D from Scratch and Transfer Learning  
This comprehensive notebook is divided into three parts:  
1. Implement a custom 2D convolutional layer (with padding and stride).  
2. Derive the mathematical formulation for backpropagation through convolutions.  
3. Apply transfer learning on CIFAR-10 using a pre-trained ResNet, and compare it with a custom CNN.

<p align="center"><img width="911" height="568" alt="image" src="https://github.com/user-attachments/assets/895a6ea6-2672-48f3-8811-a4d63d2fa79d" /></p>

### `7_rnn.ipynb` — Sentiment Analysis and Custom RNN Cell  
1. Train an RNN for sentence-level sentiment classification (SST-2), using truncated BPTT to address gradient issues.  
2. Design a novel RNN cell with a reset gate and train it to overfit a small dataset, validating gradient flow and architecture design.

<p align="center"><img width="889" height="431" alt="image" src="https://github.com/user-attachments/assets/a0fd6e1d-a858-4008-b7db-5414c033af17" /></p>

### `8_autoencoder.ipynb` — Deep Autoencoders and Inversion Accuracy  
Build a deep autoencoder architecture with symmetric encoder-decoder layers.  
Train it on MNIST and evaluate how well each decoder layer approximates the inverse of its encoder counterpart. Compare dense vs. convolutional autoencoders.

<p align="center"><img width="894" height="360" alt="image" src="https://github.com/user-attachments/assets/28f4488a-9af7-4cde-866d-145236a10995" /></p>

## 🔧 Technologies Used

- **Python 3.9+**
- **PyTorch**
- **NumPy / Matplotlib / Pandas**
- **Jupyter Notebooks**

## 📁 Structure

```
deep-learning-notebooks/
├── 1\_basic\_classifiers.ipynb
├── 2\_broadcast\_tensors.ipynb
├── 3\_autograd.ipynb
├── 4\_deciles\_vs\_percentiles.ipynb
├── 5\_splitlinear\_dropnorm.ipynb
├── 6\_conv2d\_transfer\_learning.ipynb
├── 7\_rnn.ipynb
└── 8\_autoencoder.ipynb
```
