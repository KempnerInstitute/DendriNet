# Running Scenarios

In the dendritic modeling framework, there are numerous ways to run a simulation and train a model. These scenarios are controlled and defined by the user in the config file. Here we breifly describe the different scenarios that can be run.

An ML model can be trained for two different tasks, these include the following:
1. **Classification**: In this task, the model is trained to classify the input data into different classes. The output of the model is a class label.
2. **Regression**: In this task, the model is trained to predict a continuous value. The output of the model is a continuous value.

Each of these tasks can be trained as a:
1. **probabilistic** model: The probabilistic model outputs a probability distribution over the classes or continuous values, or
2. **deterministic** model: The deterministic model outputs a single class label or continuous value.

After deciding the task and type of model, the user can choose the network. Currently the following networks are supported:

1. **Feedforward**: A feed forward network neural network.
2. **EINet**: An excitatory-inhibitory network.

The following configurations are examples of the different scenarios that can be run:

Example 1: Train a deterministic feedforward model for classification task

```yaml

model:
  task: classification
  probabilistic: false
  network: 
    type: "feedforward"
    parameters:
      input_dim: 784
      output_dim: 10

train:
  train_valid_split: 0.8
  lr: 0.005
  epochs: 50
  batch_size: 64
  grad_clip_value: 0.5

wandb:
  entity: # <---- Add your entity
  project: dendritic_modeling_pr
  group: debug
  tags: [mnist_1]

settings:
  visualizations: False

```

Example 2: Train a probabilistic EINet model for classification task


```yaml
model:
  task: classification
  probabilistic: True
  network: 
    type: "einet"
    parameters:
      input_dim: 784
      excitatory_layer_sizes: [10]
      inhibitory_layer_sizes: [20]
      excitatory_branch_factors: [24]
      inhibitory_branch_factors: []
      reactivate: true
      somatic_synapses: true
      ee_synapses_per_branch_per_layer: [24]
      ei_synapses_per_branch_per_layer: [100]
      ie_synapses_per_branch_per_layer: [1]
      ii_synapses_per_branch_per_layer: []
      topk_init_method: "xavier_normal"

train:
  train_valid_split: 0.8
  lr: 0.005
  epochs: 50
  batch_size: 64
  grad_clip_value: 0.5

wandb:
  entity: # <---- Add your entity
  project: dendritic_modeling_pr
  group: debug
  tags: [mnist_1]

settings:
  visualizations: False

```

To run the scenarios, the user can use the following command:

```bash
python scripts/train_mnist_restrucured_model.py configs/mnist_rs.yaml --experiment_name example --output_dir results
```


