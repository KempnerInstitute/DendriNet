# Dendritic Neural Network for MNIST Modulo 10

code: Dendrinet_HS_simple.ipynb

## Overview

This repository contains a Jupyter notebook that implements a dendritic neural network designed to process two MNIST digits and classify their sum modulo 10. The network consists of inhibitory and excitatory inputs, integrated by a sequential set of dendrites, with optional attention mechanisms to enhance information processing.

## Goal

The main goal of this project is to understand how a network composed of inhibitory/excitatory inputs integrated by a sequential set of dendrites can generate information in the soma. We aim to quantify the learning process, understand how information is distributed among inhibitory and excitatory weights, and explore the role of inhibition, morphology, and activation functions in this process.

## Network Structure

The network is designed to handle two MNIST digits placed side by side as input and classify their sum modulo 10. The structure includes:
- **Inhibitory and Excitatory Inputs**: These inputs are processed by dendrites in multiple layers.
- **Dendritic Layers**: Sequentially integrate inputs and propagate signals to the soma.
- **Soma Layer**: Integrates signals from dendrites to produce the final output.
- **Attention Mechanism (Optional)**: Enhances the focus on relevant input features.

## Classes and Functions

### Classes

1. **`TopKLinear`**:
   - Implements a linear layer with top-k sparsity in weights.
   - **Input**: Input tensor `x`.
   - **Output**: Weighted output tensor after applying top-k sparsity.

2. **`DendriteBranchLayer`**:
   - Represents a single dendritic branch.
   - **Input**: Input tensor `x`, inhibitory outputs, excitatory inputs, and previous dendritic branch outputs.
   - **Output**: Activation of the dendritic branch.

3. **`NetWithoutAttention`**:
   - Implements the dendritic neural network without attention.
   - **Input**: Input tensor `x`.
   - **Output**: Final output tensor representing the sum modulo 10.

4. **`NetWithAttention`**:
   - Implements the dendritic neural network with an attention mechanism.
   - **Input**: Input tensor `x`.
   - **Output**: Final output tensor representing the sum modulo 10, with attention applied to relevant features.

### Functions

1. **`train_network`**:
   - Trains the neural network for a specified number of epochs.
   - **Input**: Network instance, training data generator, test data generator, number of epochs, learning rate, and scheduler gamma.
   - **Output**: Trained network, loss history, and accuracy history.

2. **`evaluate_network`**:
   - Evaluates the network on test data.
   - **Input**: Network instance, test data generator.
   - **Output**: Accuracy, all predictions, and all labels.

3. **`compute_information_flow`**:
   - Computes the information flow through the network layers.
   - **Input**: Network instance, input data, layer name.
   - **Output**: Activation values for each layer.

4. **`compute_mutual_information`**:
   - Computes mutual information between input and activations.
   - **Input**: Input data, activation data.
   - **Output**: Mutual information values.

5. **`analyze_variance_signals`**:
   - Analyzes variance in input and output signals.
   - **Input**: Input data, output data.
   - **Output**: Input variance and output variance.

6. **`analyze_weights`**:
   - Analyzes the distribution of network weights.
   - **Input**: Network instance.
   - **Output**: Weight statistics.

7. **`visualize_attention_weights`**:
   - Visualizes attention weights to understand feature focus.
   - **Input**: Network instance, layer name.
   - **Output**: Attention weight matrix.

8. **`plot_training_metrics`**:
   - Plots training loss and accuracy over epochs.
   - **Input**: Loss history, accuracy history.
   - **Output**: Plots of loss and accuracy.

9. **`plot_dendrite_activations_by_label`**:
   - Visualizes dendrite activations for different labels.
   - **Input**: Network instance, training data generator, number of dendrites, number of labels.
   - **Output**: Activation plots for each dendrite and label.

10. **`plot_weight_representations`**:
    - Plots the representation of weights for different labels.
    - **Input**: Network instance, number of labels.
    - **Output**: Weight representation plots.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/dendritic-neural-network.git
    cd dendritic-neural-network
    ```

2. **Install required packages**:
    ```bash
    pip install torch torchvision matplotlib numpy scikit-learn graphviz
    ```

3. **Run the Jupyter notebook**:
    Open the Jupyter notebook in your preferred environment (e.g., JupyterLab, VSCode) and execute the cells to train and evaluate the network.

## Results

The results include:
- Performance metrics (accuracy, loss) on the MNIST modulo 10 task.
- Visualizations of dendrite activations and weight representations.
- Analysis of information flow and mutual information during learning.
- Visualization of attention weights and their significance.

## other codes: 

inh_exc_info_noisecorrelation.py: Simulating information represened by Soma for a simple network with two excitatory and one inhibitory input for different distribution of information in the inhibitory input about the input and excitatory inputs and their noise.

Updated_Conductance_multi_dendrite sum_MNIST_HS.ipynb: Original notebook for Dendrinet prepared by Ella and Bernardo.


## Contribution

Feel free to contribute by opening issues or submitting pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




