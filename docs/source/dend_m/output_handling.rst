.. _output_handling:

Output Handling and Loss Functions
==================================

This document explains how outputs are processed in the dendritic modeling framework and the various loss functions available for training models.

Overview
--------

The dendritic modeling framework supports both deterministic and probabilistic outputs for different tasks. The output handling mechanisms determine how the network's final layer is processed and the appropriate loss functions for training.

The framework supports the following tasks:

1. **Classification**: Predicting discrete classes
2. **Regression**: Predicting continuous values

Output Types
------------

The primary distinction in output handling is between deterministic and probabilistic outputs:

.. code-block:: yaml

    model:
      task: "classification"  # Options: "classification", "regression"
      probabilistic: true     # Whether to use probabilistic outputs

Deterministic vs. Probabilistic Outputs
---------------------------------------

1. **Deterministic outputs**:
   - Direct mapping from network output to prediction
   - Simpler training procedure
   - Less expressive, cannot represent uncertainty

2. **Probabilistic outputs**:
   - Models a distribution over possible outputs
   - Can represent uncertainty in predictions
   - More expressive but more complex to train
   - Better calibrated probabilities

The choice between deterministic and probabilistic outputs affects both the network architecture and the loss functions used for training.

Probabilistic Output Layers
---------------------------

When ``probabilistic: true`` is set, the network adds a distribution layer on top of the base network. The framework implements this through distribution output layers that transform the base network's outputs into distribution parameters.

For classification tasks, this typically involves a categorical distribution, while regression tasks use a Gaussian distribution. The parameters of these distributions (such as logits for categorical or mean and variance for Gaussian) are predicted by the network.

Classification with Probabilistic Outputs
-----------------------------------------

For classification tasks with probabilistic outputs, the framework uses a categorical distribution. The network outputs logits that are transformed into class probabilities through a softmax function.

The negative log likelihood (NLL) loss is then used to train the model, which is equivalent to the standard cross-entropy loss for classification.

Regression with Probabilistic Outputs
-------------------------------------

For regression tasks with probabilistic outputs, the framework typically uses a Gaussian distribution. The network outputs both the mean prediction and a measure of uncertainty (variance or standard deviation).

This allows the model to predict not just a point estimate but also the uncertainty associated with each prediction, which can be valuable for decision-making in many applications.

Loss Functions
--------------

The framework provides different loss functions depending on the task and whether probabilistic outputs are used:

.. code-block:: yaml

    train:
      loss:
        type: "nll"  # Options: "nll", "mse", "bce"

Negative Log Likelihood (NLL) Loss
----------------------------------

For probabilistic models, the primary loss function is negative log likelihood (NLL). This loss directly optimizes the likelihood of the observed data under the model's predicted distribution.

For classification tasks, this is equivalent to the standard cross-entropy loss, while for regression tasks with Gaussian outputs, it takes into account both the prediction error and the model's uncertainty estimates.

Mean Squared Error (MSE) Loss
-----------------------------

For deterministic regression, the framework uses mean squared error:

.. code-block:: yaml

    train:
      loss:
        type: "mse"

This is the standard loss function for regression tasks, measuring the average squared difference between the predictions and the targets.

Binary Cross Entropy (BCE) Loss
-------------------------------

For binary classification with deterministic outputs:

.. code-block:: yaml

    train:
      loss:
        type: "bce"

This is the standard loss function for binary classification tasks, measuring the cross-entropy between the predicted probabilities and the binary targets.

Combined Loss Functions
-----------------------

The framework also supports combining multiple loss terms, such as for regularization:

.. code-block:: yaml

    train:
      loss:
        type: "combined"
        components:
          - type: "nll"
            weight: 1.0
          - type: "l2_regularization"
            weight: 0.001

This allows for adding regularization terms or other custom loss components to the primary task loss.

Example Configurations
----------------------

Classification with probabilistic outputs:

.. code-block:: yaml

    model:
      task: "classification"
      probabilistic: true
      network:
        # Network configuration...
        
    train:
      loss:
        type: "nll"
      optimizer:
        type: "adam"
        lr: 0.001

Regression with deterministic outputs:

.. code-block:: yaml

    model:
      task: "regression"
      probabilistic: false
      network:
        # Network configuration...
        
    train:
      loss:
        type: "mse"
      optimizer:
        type: "adam"
        lr: 0.001

Evaluation Metrics
------------------

The framework provides common metrics for evaluating models depending on the task:

1. **Classification metrics**:
   - Accuracy
   - Precision, recall, F1-score
   - AUC-ROC

2. **Regression metrics**:
   - Mean squared error (MSE)
   - Mean absolute error (MAE)
   - R-squared

Configuration example:

.. code-block:: yaml

    evaluate:
      metrics:
        - "accuracy"
        - "f1_score"

General Best Practices for Output Handling
------------------------------------------

The following are general recommendations for working with outputs in dendritic networks:

1. **For classification tasks**:
   - Consider using probabilistic outputs when uncertainty estimation is important
   - Deterministic outputs may be sufficient for simple classification tasks

2. **For regression tasks**:
   - Probabilistic outputs can capture varying uncertainty across the input space
   - Deterministic outputs are simpler and may be adequate for many applications

3. **For evaluation**:
   - Use multiple metrics to get a comprehensive view of model performance
   - Choose metrics that align with your application's requirements

4. **For loss functions**:
   - Start with standard losses (NLL, MSE) before trying more complex alternatives
   - Consider adding regularization if overfitting is observed 