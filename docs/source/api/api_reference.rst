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
