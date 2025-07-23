Available Components
====================

This document lists all components that are implemented in the dendritic modeling framework.

Network Architectures
------------------------------

* **EINet** - Main excitation-inhibition network architecture  
* **MLP** - Standard multi-layer perceptron
* **MatchedParamMLP** - Parameter-matched baseline for EINet
* **CNNDownsample** - Convolutional encoder with downsampling
* **CNNUpsample** - Convolutional decoder with upsampling  
* **AlexNet** - Pre-trained AlexNet feature extractor
* **Identity** - Pass-through network

Encoder/Decoder Networks
------------------------------

* **Identity** - No transformation applied
* **CNNDownsample** - Convolutional downsampling for images
* **CNNUpsample** - Convolutional upsampling for generation
* **AlexNet** - Pre-trained feature extraction
* **MLPAutoencoder** - MLP-based autoencoder
* **CNNAutoencoder** - CNN-based autoencoder  
* **MLPVariationalAutoencoder** - Variational MLP autoencoder
* **CNNVariationalAutoencoder** - Variational CNN autoencoder

Input Transform Networks
------------------------------

* **IndexInputTransform** - Random index-based input splitting
* **FeatureExtractorMLP** - Pre-trained MLP for feature extraction

Synaptic Mechanisms
------------------------------

* **TopK Selection**: Standard, stochastic, variance-based, DeepST
* **Weight Initialization**: Xavier, Kaiming, orthogonal, normal, uniform
* **Dynamic Sparse Training (DeepST)**: Connectivity evolution with rewiring

Activation Functions
------------------------------

* ``none`` - Identity function
* ``relu`` - Standard ReLU
* ``sigmoid`` - Standard sigmoid  
* ``tanh`` - Standard tanh
* ``param_tanh`` - Learnable tanh with slope and bias
* ``param_tanh_only_m`` - Learnable tanh with slope only
* ``param_linear_sigmoid`` - Trainable linear-sigmoid interpolation
* ``linear_sigmoid`` - Fixed linear-sigmoid interpolation
* ``param_linear_tanh`` - Trainable linear-tanh interpolation
* ``linear_tanh`` - Fixed linear-tanh interpolation
* ``param_linear_tanh_transition`` - Smooth transition linear to tanh
* ``linear_tanh_transition`` - Fixed transition function
* ``ParametricLinearTanh`` - Learnable linear-tanh

Loss Functions
------------------------------

* Cross-entropy (classification)
* MSE (regression)
* Voltage stabilization loss
* Custom loss functions

Training Strategies
------------------------------

* **standard** - Standard backpropagation
* **freeze_layers** - Layer-wise sequential training
* **freeze_branches** - Branch-wise sequential training  
* **two_step** - Local pre-training + global fine-tuning
* **two_step_kl** - Two-step with KL divergence regularization
* **voltage_stabilization** - Biologically-motivated stabilization

Analysis Components
------------------------------

* **Performance Analysis**: Accuracy, log-likelihood, MSE
* **Information Analysis**: Mutual information with multiple estimators
* **Ablation Analysis**: Layer, compartment, and synapse ablation
* **Noise Perturbation**: Gaussian and uniform noise robustness
* **Weight Analysis**: Synaptic weight distributions
* **Correlation Analysis**: Tuning curves and correlations

Scripts
------------------------------

Main training scripts are in ``src/dendritic_modeling/scripts/``:
* ``train_experiments.py`` - Main experiment runner
* ``train_encoder_network.py`` - Encoder pre-training 