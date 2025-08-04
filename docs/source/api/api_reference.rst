.. _api_reference:

API Reference
=============

This section provides detailed documentation of the key classes and methods in the dendritic modeling framework.

ExcitationInhibitionNetwork
---------------------------

The ``ExcitationInhibitionNetwork`` is the core network class that implements the excitatory-inhibitory architecture.

.. code-block:: python

    class ExcitationInhibitionNetwork(nn.Module):
        """
        A network with multiple excitatory-inhibitory layers.

        This network implements a sequence of excitatory and inhibitory dendritic processing
        layers that model the complex interactions in biological neural networks.

        Parameters
        ----------
        input_dim : int
            Dimension of the input data
        excitatory_layer_sizes : list of int
            Sizes of the excitatory layers
        inhibitory_layer_sizes : list of int
            Sizes of the inhibitory layers
        excitatory_branch_factors : list of int
            Branch factors for excitatory dendrites in each layer
        inhibitory_branch_factors : list of int
            Branch factors for inhibitory dendrites in each layer
        ee_synapses_per_branch_per_layer : list of int
            Number of excitatory-to-excitatory synapses per branch in each layer
        ei_synapses_per_branch_per_layer : list of int
            Number of excitatory-to-inhibitory synapses per branch in each layer
        ie_synapses_per_branch_per_layer : list of int
            Number of inhibitory-to-excitatory synapses per branch in each layer
        ii_synapses_per_branch_per_layer : list of int
            Number of inhibitory-to-inhibitory synapses per branch in each layer
        reactivate : bool
            Whether to apply reactivation to prevent branch "death" during training
        somatic_synapses : bool
            Whether to allow direct connections to the somatic layer
        topk_init_method : str
            Initialization method for TopKLinear weights
        use_shunting : bool
            Whether to enable shunting inhibition
        reactivation_type : str
            Type of reactivation function to use (e.g., "param_tanh")
        reactivation_strategy : str
            Strategy for reactivation (e.g., "inverse")
        blocklinear_strategy : str
            Strategy for BlockLinear layers (e.g., "block_conductance_dynamic")
        enable_branch_outputs : bool
            Whether to enable access to branch outputs for visualization
        output_layer : bool
            Whether to include an output layer
        output_dim : int
            Dimension of the output
        input_mode : int
            Input processing mode (0, 1, or 2)
        input_net_dict : dict
            Configuration for input transformation network (used if input_mode=2)
        """

        def forward(self, x):
            """
            Forward pass through the network.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, input_dim)

            Returns
            -------
            torch.Tensor
                Output tensor of shape (batch_size, output_dim)
            """
            # Implementation details...

ExcitationInhibitionLayer
-------------------------

The ``ExcitationInhibitionLayer`` implements a single layer of excitatory and inhibitory processing.

.. code-block:: python

    class ExcitationInhibitionLayer(nn.Module):
        """
        A layer that processes inputs through excitatory and inhibitory pathways.

        This layer contains an excitatory dendrinet and an inhibitory dendrinet,
        which interact to produce the layer's output.

        Parameters
        ----------
        n_excitatory_cells : int
            Number of excitatory cells
        n_inhibitory_cells : int
            Number of inhibitory cells
        excitatory_branch_factors : list of int
            Branch factors for excitatory dendrites
        inhibitory_branch_factors : list of int
            Branch factors for inhibitory dendrites
        excitatory_input_dim : int
            Dimension of the excitatory input
        ee_synapses_per_branch : int
            Number of excitatory-to-excitatory synapses per branch
        ei_synapses_per_branch : int
            Number of excitatory-to-inhibitory synapses per branch
        inhibitory_input_dim : int, optional
            Dimension of the inhibitory input
        ie_synapses_per_branch : int, optional
            Number of inhibitory-to-excitatory synapses per branch
        ii_synapses_per_branch : int, optional
            Number of inhibitory-to-inhibitory synapses per branch
        reactivate : bool
            Whether to apply reactivation
        somatic_synapses : bool
            Whether to allow direct connections to the somatic layer
        topk_init_method : str
            Initialization method for TopKLinear weights
        use_shunting : bool
            Whether to enable shunting inhibition
        reactivation_type : str
            Type of reactivation function
        reactivation_strategy : str
            Strategy for reactivation
        blocklinear_strategy : str
            Strategy for BlockLinear layers
        enable_branch_outputs : bool
            Whether to enable access to branch outputs
        topk_strategy : str
            Strategy for topk selection of synaptic weights. Options:
            - "standard": Traditional deterministic top-K selection by weight magnitude
            - "stochastic": Probabilistic selection based on weight rank
            - "variance": Selection based on input activation variance and weight magnitude
        """

        def forward(self, excitatory_input, inhibitory_input=None):
            """
            Forward pass through the layer.

            Parameters
            ----------
            excitatory_input : torch.Tensor
                Excitatory input tensor
            inhibitory_input : torch.Tensor, optional
                Inhibitory input tensor

            Returns
            -------
            tuple of torch.Tensor
                Excitatory and inhibitory outputs
            """
            # Implementation details...

DendriNet
---------

The ``DendriNet`` class implements a dendritic network that processes inputs through a series of branch layers.

.. code-block:: python

    class DendriNet(nn.Module):
        """
        A network that models dendritic computation with branch-specific processing.

        Parameters
        ----------
        n_soma : int
            Number of soma (output units)
        branch_factors : list of int
            Branching factors for the network
        excitatory_input_dim : int
            Dimension of the excitatory input
        excitatory_synapses_per_branch : int
            Number of excitatory synapses per branch
        inhibitory_input_dim : int, optional
            Dimension of the inhibitory input
        inhibitory_synapses_per_branch : int, optional
            Number of inhibitory synapses per branch
        reactivate : bool
            Whether to apply reactivation
        somatic_synapses : bool
            Whether to allow direct connections to the somatic layer
        topk_init_method : str
            Initialization method for TopKLinear weights
        use_shunting : bool
            Whether to enable shunting inhibition
        reactivation_type : str
            Type of reactivation function
        reactivation_strategy : str
            Strategy for reactivation
        blocklinear_strategy : str
            Strategy for BlockLinear layers
        topk_strategy : str
            Strategy for topk selection of synaptic weights. Options:
            - "standard": Traditional deterministic top-K selection by weight magnitude
            - "stochastic": Probabilistic selection based on weight rank
            - "variance": Selection based on input activation variance and weight magnitude
        """

        def forward(self, x, inhibitory_input=None):
            """
            Forward pass through the dendritic network.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor
            inhibitory_input : torch.Tensor, optional
                Inhibitory input for modulation

            Returns
            -------
            torch.Tensor
                Output tensor
            """
            # Implementation details...

DendriticBranchLayer
--------------------

The ``DendriticBranchLayer`` implements a single layer of dendritic branches.

.. code-block:: python

    class DendriticBranchLayer(nn.Module):
        """
        A layer that models computation in dendritic branches.

        This layer is an analogue to the MLP which emulates dendritic computation.
        It processes excitatory and inhibitory inputs through specialized pathways,
        integrating information in a biologically-inspired manner.

        Parameters
        ----------
        output_dim : int
            Number of output units (soma)
        excitatory_input_dim : int
            Dimension of the excitatory input
        excitatory_synapses_per_branch : int
            Number of excitatory synapses per branch
        inhibitory_input_dim : int, optional
            Dimension of the inhibitory input
        inhibitory_synapses_per_branch : int, optional
            Number of inhibitory synapses per branch
        input_branch_factor : int
            Branching factor for the layer
        reactivate : bool
            Whether to apply reactivation
        somatic_synapses : bool
            Whether to allow direct connections to the somatic layer
        topk_init_method : str
            Initialization method for TopKLinear weights
        use_shunting : bool
            Whether to enable shunting inhibition
        reactivation_type : str
            Type of reactivation function
        reactivation_strategy : str
            Strategy for reactivation
        blocklinear_strategy : str
            Strategy for BlockLinear layers
        topk_strategy : str
            Strategy for topk selection of synaptic weights. Options:
            - "standard": Traditional deterministic top-K selection by weight magnitude
            - "stochastic": Probabilistic selection based on weight rank
            - "variance": Selection based on input activation variance and weight magnitude
        """

        def forward(self, x, inhibitory_input=None):
            """
            Forward pass through the branch layer.

            Parameters
            ----------
            x : torch.Tensor
                Excitatory input tensor
            inhibitory_input : torch.Tensor, optional
                Inhibitory input tensor

            Returns
            -------
            torch.Tensor
                Output tensor
            """
            # Implementation details...

TopKLinear
----------

The ``TopKLinear`` class implements a linear transformation with top-k selection of weights.

.. code-block:: python

    class TopKLinear(nn.Module):
        """
        A linear layer that retains only the top K strongest synaptic weights for in_features.

        This module implements a linear transformation with weights constrained to
        be positive. For each output neuron, only the top K weights are kept,
        and the rest are set to zero. This simulates a neuron receiving
        inputs only from its strongest synaptic connections.

        Parameters
        ----------
        in_features : int
            The number of input features.

        out_features : int
            The number of output features.

        K : int
            The number of strongest synapses to keep per dendritic branch.

        param_space : str, optional
            The parameter space for the weights. Options are 'log' and 'presigmoid'.
            If `'log'`, the weights are parameterized as exponentials of `pre_w`.
            If `'presigmoid'`, the weights are parameterized as sigmoids of `pre_w`.
            Defaults to 'log'.

        init_method : str
            Initialization method for weights.

        init_gain : float
            Gain factor for weight initialization.
        """

        def forward(self, x):
            """
            Forward pass through the layer.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor

            Returns
            -------
            torch.Tensor
                Output tensor
            """
            # Implementation details...

StochasticTopKLinear
------------------

The ``StochasticTopKLinear`` class implements a variant of TopKLinear with stochastic weight selection.

.. code-block:: python

    class StochasticTopKLinear(TopKLinear):
        """
        A variant of TopKLinear where we select top-K stochastically based on rank:

        This implementation:
        1. Ranks weights by magnitude (or magnitude + noise)
        2. Converts rank to a keep probability using a logistic function
        3. Samples from Bernoulli distribution to determine which weights to keep

        This stochastic approach allows smaller weights to occasionally contribute,
        potentially improving exploration during training and preventing overfitting to
        specific features.

        Parameters
        ----------
        in_features : int
            The number of input features.

        out_features : int
            The number of output features.

        K : int
            The number of strongest synapses to keep per dendritic branch.

        param_space : str, optional
            The parameter space for the weights. Options are 'log' and 'presigmoid'.
            If `'log'`, the weights are parameterized as exponentials of `pre_w`.
            If `'presigmoid'`, the weights are parameterized as sigmoids of `pre_w`.
            Defaults to 'log'.

        init_method : str
            Initialization method for weights.

        init_gain : float
            Gain factor for weight initialization.

        noise_level : float
            Standard deviation of Gaussian noise to add to weights during selection.

        temperature : float
            Controls the "sharpness" of the logistic function. Lower values make the
            selection more deterministic (approaching standard TopK), while higher
            values make it more uniform.

        ultrafast : bool
            When True, uses a highly optimized implementation that approximates
            the stochastic selection with a simpler algorithm. This can be 10-50x
            faster with similar behavior.
        """

        def weight_mask(self):
            """
            Generate a stochastic mask for weights based on their rank.

            Instead of a hard top-k, ranks by magnitude and keeps each weight with
            probability ~ sigmoid((k - rank)/(k*temperature)), so top ranks have
            high probability ~1 of being kept.

            Returns
            -------
            torch.Tensor
                Binary mask for weights
            """
            # Implementation details...

        def weight_mask_ultrafast(self):
            """
            Ultra-fast approximation of stochastic TopK that uses simple noise
            addition followed by standard TopK selection.

            This method:
            1. Adds scaled noise directly to the weights (noise proportional to temperature)
            2. Performs standard TopK selection on these noisy weights
            3. Avoids the more complex ranking and sampling procedure

            This provides a 10-50x speedup with similar stochastic behavior.

            Returns
            -------
            torch.Tensor
                Binary mask for weights
            """
            # Implementation details...

VarianceTopKLinear
----------------

The ``VarianceTopKLinear`` class implements a variant of TopKLinear that selects weights based on input activation variance.

.. code-block:: python

    class VarianceTopKLinear(TopKLinear):
        """
        A variance-based TopK approach that tracks input activation statistics over batches
        and selects weights based on the variance of activations at each input.

        This implementation:
        1. Uses exponential moving averages to track input statistics
        2. Computes variance estimates for each input dimension
        3. Selects connections based on input variance * weight magnitude

        This approach prioritizes connections to inputs that have high variance
        (carry more information) and have strong weights (model already considers important).

        Parameters
        ----------
        in_features : int
            The number of input features.

        out_features : int
            The number of output features.

        K : int
            The number of strongest synapses to keep per dendritic branch.

        param_space : str, optional
            The parameter space for the weights. Options are 'log' and 'presigmoid'.
            If `'log'`, the weights are parameterized as exponentials of `pre_w`.
            If `'presigmoid'`, the weights are parameterized as sigmoids of `pre_w`.
            Defaults to 'log'.

        init_method : str
            Initialization method for weights.

        init_gain : float
            Gain factor for weight initialization.

        noise_level : float
            Standard deviation of Gaussian noise to add to weights during selection.

        momentum : float
            Momentum parameter for the exponential moving average updates of activation
            statistics. Higher values (closer to 1.0) give more weight to past
            observations.
        """

        def update_stats(self, x):
            """
            Update activation statistics with current batch.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor
            """
            # Implementation details...

Analysis and Evaluation Tools
---------------------------

The framework includes comprehensive analysis tools for evaluating model performance and understanding dendritic computation.

AnalysisManager
~~~~~~~~~~~~~

The ``AnalysisManager`` coordinates various analysis tasks:

.. code-block:: python

    class AnalysisManager:
        """
        Centralized manager for running various types of analysis on models.
        
        This class coordinates performance analysis, ablation studies, information
        analysis, and other evaluation tasks.
        
        Parameters
        ----------
        output_dir : str
            Directory for saving analysis results
        save_format : str
            Format for saving results ('json', 'csv', 'pkl')
        """
        
        def run_full_analysis(self, model, dataset, config):
            """
            Run comprehensive analysis suite on a trained model.
            
            Parameters
            ----------
            model : torch.nn.Module
                Trained model to analyze
            dataset : Dataset
                Dataset for evaluation
            config : dict
                Analysis configuration
                
            Returns
            -------
            dict
                Analysis results
            """
            # Implementation details...

PerformanceAnalyzer
~~~~~~~~~~~~~~~~

Analyzes model performance metrics:

.. code-block:: python

    class PerformanceAnalyzer:
        """
        Analyzer for model performance metrics including accuracy, loss, and timing.
        
        Provides comprehensive performance evaluation with statistical analysis
        and visualization capabilities.
        """
        
        def analyze_accuracy(self, model, dataset):
            """Analyze model accuracy across different conditions."""
            pass
            
        def analyze_confusion_matrix(self, model, dataset):
            """Generate and analyze confusion matrix."""
            pass
            
        def benchmark_inference_time(self, model, input_shapes):
            """Benchmark model inference time."""
            pass

AblationAnalyzer
~~~~~~~~~~~~~~

Performs ablation studies on dendritic components:

.. code-block:: python

    class AblationAnalyzer:
        """
        Analyzer for conducting ablation studies on dendritic networks.
        
        Tests the impact of disabling different components (branches, layers,
        connections) on model performance.
        """
        
        def ablate_branches(self, model, dataset, branch_indices):
            """Ablate specific dendritic branches."""
            pass
            
        def ablate_layers(self, model, dataset, layer_indices):
            """Ablate entire layers."""
            pass
            
        def systematic_ablation(self, model, dataset):
            """Run systematic ablation across all components."""
            pass

BranchInformationAnalyzer
~~~~~~~~~~~~~~~~~~~~~~

Analyzes information processing in dendritic branches:

.. code-block:: python

    class BranchInformationAnalyzer:
        """
        Analyzer for studying information flow and processing in dendritic branches.
        
        Examines how information is distributed, processed, and integrated
        across different branches and layers.
        """
        
        def analyze_branch_activations(self, model, dataset):
            """Analyze activation patterns across branches."""
            pass
            
        def compute_branch_mutual_information(self, model, dataset):
            """Compute mutual information between branches and targets."""
            pass
            
        def analyze_branch_specialization(self, model, dataset):
            """Analyze how branches specialize for different inputs."""
            pass

Training Strategies
-----------------

The framework supports multiple training strategies optimized for dendritic networks.

TrainerMLE
~~~~~~~~

Maximum Likelihood Estimation trainer:

.. code-block:: python

    class TrainerMLE:
        """
        Trainer using Maximum Likelihood Estimation.
        
        Standard end-to-end training approach suitable for most dendritic
        network architectures.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for parameter updates
        criterion : torch.nn.Module, optional
            Loss function (defaults to CrossEntropyLoss)
        """
        
        def train(self, model, train_data, valid_data, epochs, **kwargs):
            """
            Train model using MLE approach.
            
            Parameters
            ----------
            model : torch.nn.Module
                Model to train
            train_data : Dataset
                Training dataset
            valid_data : Dataset
                Validation dataset
            epochs : int
                Number of training epochs
                
            Returns
            -------
            dict
                Training results and metrics
            """
            # Implementation details...

TwoStepTrainer
~~~~~~~~~~~~

Two-step training for dendritic networks:

.. code-block:: python

    class TwoStepTrainer:
        """
        Two-step training strategy for dendritic networks.
        
        First pretrains dendritic components, then fine-tunes the entire network.
        Useful for complex architectures or when standard training fails.
        """
        
        def pretrain_dendrites(self, model, data, epochs):
            """Pretrain dendritic components."""
            pass
            
        def finetune_network(self, model, data, epochs):
            """Fine-tune entire network."""
            pass

Dataset Utilities
---------------

Comprehensive dataset loading and preprocessing utilities.

Synthetic Datasets
~~~~~~~~~~~~~~~~

.. code-block:: python

    def load_line_orientation_dataset(n_samples=1000, **kwargs):
        """
        Load line orientation classification dataset.
        
        Tests the ability to discriminate between different line orientations,
        a classic test of spatial pattern recognition.
        """
        pass
    
    def load_xor_dataset(n_samples=1000, noise_level=0.1, **kwargs):
        """
        Load XOR dataset for testing nonlinear separation capabilities.
        """
        pass
    
    def load_center_surround_dataset(n_samples=1000, **kwargs):
        """
        Load center-surround pattern recognition dataset.
        
        Tests spatial pattern recognition similar to retinal processing.
        """
        pass

Real Datasets
~~~~~~~~~~~

.. code-block:: python

    def load_mnist_as_datasets(data_path="", train_valid_split=0.8, **kwargs):
        """
        Load MNIST dataset with train/validation split.
        
        Returns
        -------
        tuple
            (train_dataset, valid_dataset, test_dataset)
        """
        pass
    
    def load_cifar10_as_datasets(data_path="", train_valid_split=0.8, **kwargs):
        """Load CIFAR-10 dataset with train/validation split."""
        pass

Visualization Tools
-----------------

Comprehensive plotting and visualization utilities.

Network Visualization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def plot_network_architecture(model, save_path=None):
        """
        Visualize dendritic network architecture.
        
        Creates a graph representation showing the structure of dendritic
        branches, connections, and data flow.
        """
        pass
    
    def plot_branch_connectivity(model, layer_idx=0, save_path=None):
        """Plot connectivity patterns in dendritic branches."""
        pass

Training Visualization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def plot_training_curves(training_results, save_path=None):
        """Plot training and validation curves."""
        pass
    
    def plot_learning_rate_schedule(optimizer_history, save_path=None):
        """Plot learning rate schedule over training."""
        pass

Analysis Visualization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def plot_ablation_results(ablation_results, save_path=None):
        """Visualize ablation study results."""
        pass
    
    def plot_branch_activations(activations, save_path=None):
        """Plot branch activation patterns."""
        pass
    
    def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix with customizable styling."""
        pass

Configuration System
------------------

The framework uses a sophisticated configuration system based on OmegaConf for experiment management.

Configuration Classes
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @dataclass
    class EINetParameters:
        """Configuration for ExcitationInhibition network parameters."""
        input_dim: int
        excitatory_layer_sizes: List[int]
        inhibitory_layer_sizes: List[int]
        output_dim: int = 10
        # ... additional parameters
    
    @dataclass
    class TrainingConfig:
        """Training configuration parameters."""
        learning_strategy: str = "mle"
        lr: float = 0.001
        epochs: int = 20
        batch_size: int = 64
        # ... additional parameters
    
    @dataclass
    class ExperimentConfig:
        """Top-level experiment configuration."""
        model: ModelConfig
        train: TrainingConfig
        task: TaskConfig
        outputs: OutputConfig

Usage Examples
~~~~~~~~~~~~

.. code-block:: python

    # Load configuration from YAML
    config = load_config("experiment_config.yaml")
    
    # Create model from config
    model = create_model_from_config(config.model)
    
    # Run training
    trainer = create_trainer_from_config(config.train)
    results = trainer.train(model, datasets, **config.train)

For complete API documentation with all parameters and methods, see the auto-generated documentation in the following sections.
            # Implementation details...

        def weight_mask(self):
            """
            Select TopK connections based on a combination of:
            - Input activation variance
            - Weight magnitude

            The intuition is that we want to keep connections to inputs that:
            1. Have high variance (carry more information)
            2. Have strong weights (model cares about them)

            Returns
            -------
            torch.Tensor
                Binary mask for weights
            """
            # Implementation details...

        def train(self, mode=True):
            """
            Override train method to control stats tracking.

            Parameters
            ----------
            mode : bool
                Whether to set the module in training mode

            Returns
            -------
            self
            """
            # Implementation details...

        def eval(self):
            """
            Override eval method to freeze stats in eval mode.

            Returns
            -------
            self
            """
            # Implementation details...

BlockLinear
----------

The ``BlockLinear`` class implements a linear transformation that aggregates information from converging branches.

.. code-block:: python

    class BlockLinear(nn.Module):
        """
        A custom linear layer that aggregates information from converging branches
        in a previous branch layer.

        Parameters
        ----------
        in_features : int
            The number of input features.

        out_features : int
            The number of output features.

        Attributes
        ----------
        weight : torch.nn.Parameter
            The weights for the linear transformation.

        in_features : int
            The number of input features.

        out_features : int
            The number of output features.

        block_size : int
            The number of branches converging onto a single branch.
        """

        def forward(self, x):
            """
            Forward pass through the layer.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor

            Returns
            -------
            torch.Tensor
                Output tensor
            """
            # Implementation details...

ParametricTanh
-------------

The ``ParametricTanh`` class implements a parametric tanh activation function with learnable parameters.

.. code-block:: python

    class ParametricTanh(nn.Module):
        """
        A parametric tanh reactivation function with learnable parameters m and b.
        Both parameters are trainable.

        Parameters
        ----------
        output_dim : int
            Dimension of the output
        init_m : float
            Initial value for the slope parameter
        init_b : float
            Initial value for the offset parameter
        """

        def forward(self, V):
            """
            Forward pass through the activation function.

            Parameters
            ----------
            V : torch.Tensor
                Input tensor (membrane potential)

            Returns
            -------
            torch.Tensor
                Output tensor (activation)
            """
            # Implementation details...

Training Strategies
------------------

The framework includes several training strategies for dendritic networks:

TrainerMLE
~~~~~~~~~

.. code-block:: python

    class TrainerMLE(BaseTrainer):
        """
        Trainer using Maximum Likelihood Estimation.

        This trainer implements standard backpropagation training as well as various
        freeze-branch strategies.

        Methods
        -------
        train(model, train_data, valid_data, ...)
            Standard training loop
        train_freeze_branches(model, train_data, valid_data, ...)
            Train with freeze branches strategy
        train_freeze_branches_reversed(model, train_data, valid_data, ...)
            Train with freeze branches reversed strategy
        train_freeze_branches_with_pretraining(model, train_data, valid_data, ...)
            Train with freeze branches with pretraining strategy
        train_freeze_branches_reversed_with_pretraining(model, train_data, valid_data, ...)
            Train with freeze branches reversed with pretraining strategy
        """

TwoStepTrainer
~~~~~~~~~~~~

.. code-block:: python

    class TwoStepTrainer(BaseTrainer):
        """
        Trainer implementing two-step training.

        This trainer implements a two-phase approach with pretraining and main training.

        Methods
        -------
        train(model, train_data, valid_data, pretrain_epochs, maintrain_epochs, ...)
            Two-step training loop
        """

VoltageStabilizationTrainer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class VoltageStabilizationTrainer(BaseTrainer):
        """
        Trainer with voltage stabilization warmup.

        This trainer implements a warmup phase to stabilize model activations.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to use
        warmup_epochs : int
            Number of epochs for warmup phase
        stability_target_voltage : float
            Target activation value for stabilization

        Methods
        -------
        train(model, train_data, valid_data, ...)
            Training loop with warmup phase
        """

Helper Functions
--------------

The framework includes various utility functions for working with dendritic networks:

.. code-block:: python

    def evaluate_accuracy(classifier, train_ds=None, valid_ds=None, test_ds=None, ...):
        """
        Evaluate the accuracy of a classifier on datasets.

        Parameters
        ----------
        classifier : torch.nn.Module
            The classifier to evaluate
        train_ds : torch.utils.data.Dataset, optional
            Training dataset
        valid_ds : torch.utils.data.Dataset, optional
            Validation dataset
        test_ds : torch.utils.data.Dataset, optional
            Test dataset

        Returns
        -------
        tuple
            (train_accuracy, valid_accuracy, test_accuracy)
        """

    def analyze_branch_ablation(model, dataset, layer_idx, branch_ids, ...):
        """
        Analyze the effect of ablating specific branches.

        Parameters
        ----------
        model : torch.nn.Module
            The model to analyze
        dataset : torch.utils.data.Dataset
            Dataset for evaluation
        layer_idx : int
            Layer index to ablate
        branch_ids : list of int
            Branch IDs to ablate

        Returns
        -------
        tuple
            (baseline_accuracy, {branch_id: (accuracy_after_ablation, accuracy_drop)})
        """
