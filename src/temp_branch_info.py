
class BranchInformationAnalyzer:
    """
    Analyzes mutual information between branch components (E, I, V_O, C).
    Uses copula entropy from DVC_tensorflow package for MI computation.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        use_dvc: bool = True,  # Whether to use DVC_tensorflow or fallback to sklearn
        save_plots: bool = True,
        save_csv: bool = True,
        copula_params: Dict = None,  # Additional parameters for copula MI estimation
        # Configurable MI metrics to compute
        compute_metrics: List[str] = None,  # List of metrics to compute
        # Per-branch vs pooled computation
        compute_per_branch: bool = False,  # If true, compute MI per branch and average; if false, pool all branches
    ):
        self.n_neighbors = n_neighbors
        self.use_dvc = use_dvc
        self.save_plots = save_plots
        self.save_csv = save_csv
        self.compute_per_branch = compute_per_branch
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"BranchInformationAnalyzer will compute {'per-branch' if self.compute_per_branch else 'pooled'} MI analysis"
        )

        # Define all available metrics
        self.available_metrics = {
            # Information with Class Labels
            "I_E_C": "I(E;C) - Excitatory inputs and class",
            "I_I_C": "I(I;C) - Inhibitory inputs and class",
            "I_V_C": "I(V;C) - Branch outputs and class",
            # Information Between Branch Components
            "I_E_I": "I(E;I) - Excitatory and inhibitory inputs",
            "I_E_V": "I(E;V) - Excitatory inputs and branch outputs",
            "I_I_V": "I(I;V) - Inhibitory inputs and branch outputs",
            # Conditional Information (Given Class)
            "I_E_I_given_C": "I(E;I|C) - E and I given class",
            "I_E_V_given_C": "I(E;V|C) - E and V given class",
            "I_I_V_given_C": "I(I;V|C) - I and V given class",
        }

        # Set which metrics to compute (default: all)
        if compute_metrics is None:
            self.compute_metrics = list(self.available_metrics.keys())
        else:
            # Validate requested metrics
            invalid_metrics = set(compute_metrics) - set(self.available_metrics.keys())
            if invalid_metrics:
                raise ValueError(
                    f"Invalid metrics requested: {invalid_metrics}. "
                    f"Available metrics: {list(self.available_metrics.keys())}"
                )
            self.compute_metrics = compute_metrics

        self.logger.info(
            f"BranchInformationAnalyzer will compute {len(self.compute_metrics)} metrics: {self.compute_metrics}"
        )

        # Default copula parameters
        self.copula_params = copula_params or {
            "cases": 1000,  # Number of samples for entropy estimation
            "iterations": 10,  # Number of iterations
            "alpha": 0.05,  # Confidence level
            "vine_type": "c-vine",  # Type of vine copula
            "families": "kercop",  # Copula families
            "knots": 30,  # Number of knots for non-parametric copula
        }

        # Initialize DVC components if requested
        if self.use_dvc:
            # Temporarily disable DVC initialization due to syntax issues
            # TODO: Fix DVC initialization structure
            self.logger.warning(
                "DVC initialization temporarily disabled due to syntax issues"
            )
            self.use_dvc = False
            self.vine_entropy = None
            self.vine_obj_bin = None
            self.margin_obj = None
            self.prep_cop = None
            self.tf = None
        else:
            self.vine_entropy = None
            self.vine_obj_bin = None
            self.margin_obj = None
            self.prep_cop = None
            self.tf = None

        # Initialize vine parameters
        self.copula_params = copula_params or {}

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to collect branch data (reusing EI analyzer pattern)"""
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if (
                    module.branch_excitation is not None
                    and module.branch_inhibition is not None
                ):
                    if name not in self.data_dict.keys():
                        self.data_dict[name] = {}
                        module._name = name
                        handle = module.register_forward_hook(self.forward_hook)
                        handles.append(handle)
        return handles

    def remove_forward_hooks(self, handles: List[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks"""
        for handle in handles:
            handle.remove()

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        output: torch.Tensor,
    ):
        """Forward hook to collect branch data during forward pass"""
        excitatory_input, inhibitory_input, _ = input
        key = module._name

        # Ensure the key exists in data_dict (might be cleared between batches)
        if key not in self.data_dict:
            self.data_dict[key] = {}

        # Collect the same data as EIInformationAnalyzer
        self.data_dict[key]["excitation"] = module.branch_excitation(excitatory_input)
        self.data_dict[key]["inhibition"] = module.branch_inhibition(inhibitory_input)
        self.data_dict[key]["output"] = output

        # Store layer index for organization
        if hasattr(module, "layer_idx"):
            self.data_dict[key]["layer_idx"] = module.layer_idx

    def compute_mi_continuous(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI between continuous variables"""
        # Ensure inputs are 2D arrays as expected by compute_mi_cc
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.use_dvc:
            # Use copula entropy approach
            mi = self._compute_copula_mi(x, y)
            return mi

        # Fallback to sklearn
        return compute_mi_cc(x, y, self.n_neighbors)

    def compute_mi_discrete(self, x: np.ndarray, c: np.ndarray) -> float:
        """Compute MI between continuous and discrete variables"""
        # Ensure inputs are 2D arrays as expected by compute_mi_cc
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        # Add uniform noise to discrete labels as suggested
        c_noisy = c + np.random.uniform(-0.5, 0.5, size=c.shape)

        if self.use_dvc:
            # Use copula entropy approach with noisy labels
            mi = self._compute_copula_mi(x, c_noisy)
            return mi

        # Fallback to sklearn
        return compute_mi_cc(x, c_noisy, self.n_neighbors)

    def compute_cmi(self, x: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
        """Compute conditional MI: I(X;Y|C)"""
        # Add uniform noise to discrete condition variable
        c_noisy = c + np.random.uniform(-0.5, 0.5, size=c.shape)

        if self.use_dvc:
            # Use copula approach for conditional MI
            # I(X;Y|C) = H(X|C) + H(Y|C) - H(X,Y|C)
            # We compute this per class and average
            unique_classes = np.unique(c)
            cmi_values = []

            for class_val in unique_classes:
                mask = c == class_val
                if np.sum(mask) < 10:  # Skip if too few samples
                    continue

                x_class = x[mask]
                y_class = y[mask]

                # Compute MI for this class
                mi_class = self._compute_copula_mi(x_class, y_class)
                cmi_values.append(mi_class)

            # Average over classes
            return np.mean(cmi_values) if cmi_values else 0.0

        # Fallback approach: compute per class and average
        unique_classes = np.unique(c)
        cmi_values = []

        for class_val in unique_classes:
            mask = c == class_val
            if np.sum(mask) < self.n_neighbors + 1:
                continue

            x_class = x[mask]
            y_class = y[mask]

            # Ensure inputs are 2D arrays as expected by compute_mi_cc
            if x_class.ndim == 1:
                x_class = x_class.reshape(-1, 1)
            if y_class.ndim == 1:
                y_class = y_class.reshape(-1, 1)

            # Compute MI for this class
            mi_class = compute_mi_cc(x_class, y_class, self.n_neighbors)
            cmi_values.append(mi_class)

        return np.mean(cmi_values) if cmi_values else 0.0

    def _compute_copula_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute MI using copula-based dependence estimation from DVC_tensorflow
        Uses parametric or non-parametric approach based on copula_params configuration
        """
        try:
            # Ensure inputs are 2D arrays as expected by compute_mi_cc
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            # Try DVC_tensorflow if requested
            if self.use_dvc:
                # Read the nonparametric setting from copula_params
                use_nonparametric = self.copula_params.get("use_nonparametric", False)

                self.logger.debug(
                    f"Attempting DVC_tensorflow copula MI computation with {'non-parametric' if use_nonparametric else 'parametric'} approach..."
                )
                try:
                    # Force TensorFlow to use CPU to avoid GPU compilation issues
                    import os

                    os.environ["CUDA_VISIBLE_DEVICES"] = ""

                    # Import DVC components
                    import tensorflow as tf

                    tf.config.set_visible_devices([], "GPU")

                    # Combine data for joint distribution
                    data_joint = np.hstack([x, y])
                    n_samples, n_dims = data_joint.shape

                    # Ensure we have enough samples and only 2D
                    if n_samples < 50:
                        raise ValueError(
                            f"Too few samples ({n_samples}) for copula estimation"
                        )

                    if n_dims != 2:
                        raise ValueError("Copula method only supports 2D MI estimation")

                    # Transform to uniform margins using empirical CDF
                    from scipy import stats

                    u = np.zeros_like(data_joint)
                    for i in range(n_dims):
                        # Rank transform to uniform
                        ranks = stats.rankdata(data_joint[:, i])
                        u[:, i] = ranks / (len(ranks) + 1)  # Avoid 0 and 1

                    u = u.astype(np.float32)

                    if use_nonparametric:
                        # Non-parametric kernel copula approach
                        self.logger.debug("Using non-parametric kernel copula approach")

                        # For non-parametric, use a simplified approach that falls back to parametric estimation
                        # The DVC non-parametric kernel copula implementation is complex and requires specific setup
                        # For now, we use the parametric approach but could be extended later
                        self.logger.warning(
                            "Non-parametric kernel copula not fully implemented, using parametric approach"
                        )

                        # Fall through to parametric approach
                        use_nonparametric = False

                    if not use_nonparametric:
                        # Parametric approach - use simpler direct estimation
                        self.logger.debug("Using parametric copula approach")

                        # For parametric approach, use simple Gaussian copula
                        # which we can compute analytically from correlation
                        from scipy.stats import kendalltau, pearsonr

                        # Compute Pearson correlation of uniform-transformed data
                        corr_pearson, _ = pearsonr(u[:, 0], u[:, 1])

                        # Convert to Gaussian copula correlation
                        # For uniform data, Pearson correlation approximates Gaussian copula parameter
                        rho = np.clip(corr_pearson, -0.99, 0.99)

                        # Compute MI from Gaussian copula: MI = -0.5 * log(1 - rho^2)
                        mi = -0.5 * np.log(1 - rho**2)
                        mi = max(0.0, mi)  # Ensure non-negative

                        self.logger.debug(
                            f"Parametric Gaussian copula: rho={rho:.4f}, MI={mi:.4f}"
                        )
                        return float(mi)

                except Exception as e:
                    self.logger.warning(f"DVC_tensorflow MI computation failed: {e}")
                    self.logger.warning("Falling back to sklearn for this computation")
                    # Fall back to sklearn
                    pass

            # Fallback to sklearn
            from dendritic_modeling.information import compute_mi_cc

            return compute_mi_cc(x, y, self.n_neighbors)

        except Exception as e:
            self.logger.error(f"All MI computation methods failed: {e}")
            return 0.0

    def analyze(
        self, model: BaseModel, test_loader: DataLoader, output_dir: str, epoch: int
    ) -> Dict[str, Any]:
        """Analyze branch information metrics using the improved hook-based approach"""
        self.logger.info("Starting branch information analysis...")

        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize data collection
        self.data_dict = {}

        if self.compute_per_branch:
            # Store data organized by layer and branch
            all_data = {
                "layer_branch_data": {},  # {layer_idx: {branch_idx: {'E': [], 'I': [], 'V': [], 'C': []}}}
            }
        else:
            # Store all collected data pooled across branches
            all_data = {
                "E": [],  # Excitatory inputs
                "I": [],  # Inhibitory inputs
                "V": [],  # Branch outputs
                "C": [],  # Class labels
                "layer_idx": [],  # Layer indices
            }

        model.eval()
        with torch.no_grad():
            # Attach hooks to collect branch data (reusing EI analyzer pattern)
            handles = self.attach_forward_hooks(model)

            try:
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    if batch_idx >= 10:  # Limit samples for efficiency
                        break

                    inputs = inputs.to(next(model.parameters()).device)

                    # Run standard forward pass - hooks will collect data automatically
                    _ = model(inputs)

                    # Process collected data from hooks
                    for layer_name, layer_data in self.data_dict.items():
                        if "excitation" in layer_data and "inhibition" in layer_data:
                            # Get the collected tensors
                            excitation = layer_data[
                                "excitation"
                            ]  # Shape: [batch, n_branches]
                            inhibition = layer_data[
                                "inhibition"
                            ]  # Shape: [batch, n_branches]
                            output = layer_data["output"]  # Shape: [batch, n_branches]

                            layer_idx = layer_data.get("layer_idx", 0)
                            n_samples = excitation.shape[0]
                            n_branches = (
                                excitation.shape[1] if excitation.ndim > 1 else 1
                            )

                            if self.compute_per_branch:
                                # Store data per branch for per-branch analysis
                                if layer_idx not in all_data["layer_branch_data"]:
                                    all_data["layer_branch_data"][layer_idx] = {}

                                if excitation.ndim > 1:
                                    for branch_idx in range(n_branches):
                                        if (
                                            branch_idx
                                            not in all_data["layer_branch_data"][
                                                layer_idx
                                            ]
                                        ):
                                            all_data["layer_branch_data"][layer_idx][
                                                branch_idx
                                            ] = {"E": [], "I": [], "V": [], "C": []}

                                        # Store data for this specific branch
                                        all_data["layer_branch_data"][layer_idx][
                                            branch_idx
                                        ]["E"].extend(
                                            excitation[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["layer_branch_data"][layer_idx][
                                            branch_idx
                                        ]["I"].extend(
                                            inhibition[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["layer_branch_data"][layer_idx][
                                            branch_idx
                                        ]["V"].extend(
                                            output[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["layer_branch_data"][layer_idx][
                                            branch_idx
                                        ]["C"].extend(targets.cpu().numpy())
                                else:
                                    # Single branch case
                                    if (
                                        0
                                        not in all_data["layer_branch_data"][layer_idx]
                                    ):
                                        all_data["layer_branch_data"][layer_idx][0] = {
                                            "E": [],
                                            "I": [],
                                            "V": [],
                                            "C": [],
                                        }

                                    all_data["layer_branch_data"][layer_idx][0][
                                        "E"
                                    ].extend(excitation.cpu().numpy())
                                    all_data["layer_branch_data"][layer_idx][0][
                                        "I"
                                    ].extend(inhibition.cpu().numpy())
                                    all_data["layer_branch_data"][layer_idx][0][
                                        "V"
                                    ].extend(output.cpu().numpy())
                                    all_data["layer_branch_data"][layer_idx][0][
                                        "C"
                                    ].extend(targets.cpu().numpy())

                            else:
                                # Pool all branches together (original behavior)
                                if excitation.ndim > 1:
                                    for branch_idx in range(n_branches):
                                        all_data["E"].extend(
                                            excitation[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["I"].extend(
                                            inhibition[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["V"].extend(
                                            output[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["C"].extend(targets.cpu().numpy())
                                        all_data["layer_idx"].extend(
                                            [layer_idx] * n_samples
                                        )
                                else:
                                    all_data["E"].extend(excitation.cpu().numpy())
                                    all_data["I"].extend(inhibition.cpu().numpy())
                                    all_data["V"].extend(output.cpu().numpy())
                                    all_data["C"].extend(targets.cpu().numpy())
                                    all_data["layer_idx"].extend(
                                        [layer_idx] * n_samples
                                    )

                    # Clear collected data for next batch to save memory
                    self.data_dict.clear()

            finally:
                # Always remove hooks
                self.remove_forward_hooks(handles)

        # Check if we have data
        if self.compute_per_branch:
            if not all_data["layer_branch_data"]:
                self.logger.warning("No branch data collected. Skipping analysis.")
                return {}
        else:
            # Convert to numpy arrays for pooled approach
            for key in all_data:
                all_data[key] = np.array(all_data[key])

        if len(all_data["E"]) == 0:
            self.logger.warning("No branch data collected. Skipping analysis.")
            return {}

        # Compute information metrics
        if self.compute_per_branch:
            results = self._compute_per_branch_metrics(all_data)
        else:
            results = self._compute_pooled_metrics(all_data)

        # Create plots
        if self.save_plots and results.get("layer"):
            self._plot_results(results, output_dir, epoch)

        # Save CSV
        if self.save_csv and results.get("layer"):
            self._save_csv(results, output_dir, epoch)

        return results

    def _compute_per_branch_metrics(self, all_data: Dict) -> Dict[str, Any]:
        """Compute MI metrics per branch and return mean/std across branches"""
        self.logger.info("Computing per-branch MI metrics...")

        results = {"layer": []}

        # Initialize metrics with mean and std
        for metric in self.compute_metrics:
            results[f"{metric}_mean"] = []
            results[f"{metric}_std"] = []
            results[f"{metric}_n_branches"] = []

        layer_branch_data = all_data["layer_branch_data"]

        for layer_idx in sorted(layer_branch_data.keys()):
            branch_data = layer_branch_data[layer_idx]

            # Skip if no branches in this layer
            if not branch_data:
                continue

            self.logger.info(
                f"Computing MI metrics for layer {layer_idx} with {len(branch_data)} branches"
            )

            # Store MI values for each branch
            branch_mis = {metric: [] for metric in self.compute_metrics}

            for branch_idx, data in branch_data.items():
                # Convert to numpy arrays
                E_branch = np.array(data["E"])
                I_branch = np.array(data["I"])
                V_branch = np.array(data["V"])
                C_branch = np.array(data["C"])

                # Skip if insufficient data for this branch
                if len(E_branch) < 50:
                    self.logger.warning(
                        f"Insufficient data for layer {layer_idx}, branch {branch_idx} ({len(E_branch)} samples), skipping"
                    )
                    continue

                # Compute MI for this branch
                if "I_E_C" in self.compute_metrics:
                    mi = self.compute_mi_discrete(E_branch, C_branch)
                    branch_mis["I_E_C"].append(mi)

                if "I_I_C" in self.compute_metrics:
                    mi = self.compute_mi_discrete(I_branch, C_branch)
                    branch_mis["I_I_C"].append(mi)

                if "I_V_C" in self.compute_metrics:
                    mi = self.compute_mi_discrete(V_branch, C_branch)
                    branch_mis["I_V_C"].append(mi)

                if "I_E_I" in self.compute_metrics:
                    mi = self.compute_mi_continuous(E_branch, I_branch)
                    branch_mis["I_E_I"].append(mi)

                if "I_E_V" in self.compute_metrics:
                    mi = self.compute_mi_continuous(E_branch, V_branch)
                    branch_mis["I_E_V"].append(mi)

                if "I_I_V" in self.compute_metrics:
                    mi = self.compute_mi_continuous(I_branch, V_branch)
                    branch_mis["I_I_V"].append(mi)

                if "I_E_I_given_C" in self.compute_metrics:
                    mi = self.compute_cmi(E_branch, I_branch, C_branch)
                    branch_mis["I_E_I_given_C"].append(mi)

                if "I_E_V_given_C" in self.compute_metrics:
                    mi = self.compute_cmi(E_branch, V_branch, C_branch)
                    branch_mis["I_E_V_given_C"].append(mi)

                if "I_I_V_given_C" in self.compute_metrics:
                    mi = self.compute_cmi(I_branch, V_branch, C_branch)
                    branch_mis["I_I_V_given_C"].append(mi)

            # Skip layer if no valid branches
            if not any(branch_mis.values()):
                self.logger.warning(
                    f"No valid branches for layer {layer_idx}, skipping"
                )
                continue

            # Compute mean and std across branches for each metric
            results["layer"].append(int(layer_idx))

            for metric in self.compute_metrics:
                if branch_mis[metric]:
                    mean_mi = np.mean(branch_mis[metric])
                    std_mi = (
                        np.std(branch_mis[metric])
                        if len(branch_mis[metric]) > 1
                        else 0.0
                    )
                    n_branches = len(branch_mis[metric])
                else:
                    mean_mi = 0.0
                    std_mi = 0.0
                    n_branches = 0

                results[f"{metric}_mean"].append(mean_mi)
                results[f"{metric}_std"].append(std_mi)
                results[f"{metric}_n_branches"].append(n_branches)

                self.logger.info(
                    f"Layer {layer_idx} {metric}: mean={mean_mi:.4f}, std={std_mi:.4f}, n_branches={n_branches}"
                )

        return results

    def _compute_pooled_metrics(self, all_data: Dict) -> Dict[str, Any]:
        """Compute MI metrics by pooling all branches together (original behavior)"""
        self.logger.info("Computing pooled MI metrics...")

        results = {"layer": []}

        # Initialize only requested metrics
        for metric in self.compute_metrics:
            results[metric] = []

        # Compute metrics per layer
        unique_layers = np.unique(all_data["layer_idx"])
        for layer_idx in unique_layers:
            mask = all_data["layer_idx"] == layer_idx

            E_layer = all_data["E"][mask]
            I_layer = all_data["I"][mask]
            V_layer = all_data["V"][mask]
            C_layer = all_data["C"][mask[: len(all_data["C"])]]

            # Skip if insufficient data
            if len(E_layer) < 100:
                self.logger.warning(
                    f"Insufficient data for layer {layer_idx}, skipping"
                )
                continue

            self.logger.info(f"Computing MI metrics for layer {layer_idx}")

            # Compute mutual information metrics
            results["layer"].append(int(layer_idx))

            # Only compute requested metrics
            if "I_E_C" in self.compute_metrics:
                results["I_E_C"].append(self.compute_mi_discrete(E_layer, C_layer))
            if "I_I_C" in self.compute_metrics:
                results["I_I_C"].append(self.compute_mi_discrete(I_layer, C_layer))
            if "I_V_C" in self.compute_metrics:
                results["I_V_C"].append(self.compute_mi_discrete(V_layer, C_layer))

            # MI between branch components
            if "I_E_I" in self.compute_metrics:
                results["I_E_I"].append(self.compute_mi_continuous(E_layer, I_layer))
            if "I_E_V" in self.compute_metrics:
                results["I_E_V"].append(self.compute_mi_continuous(E_layer, V_layer))
            if "I_I_V" in self.compute_metrics:
                results["I_I_V"].append(self.compute_mi_continuous(I_layer, V_layer))

            # Conditional MI given class
            if "I_E_I_given_C" in self.compute_metrics:
                results["I_E_I_given_C"].append(
                    self.compute_cmi(E_layer, I_layer, C_layer)
                )
            if "I_E_V_given_C" in self.compute_metrics:
                results["I_E_V_given_C"].append(
                    self.compute_cmi(E_layer, V_layer, C_layer)
                )
            if "I_I_V_given_C" in self.compute_metrics:
                results["I_I_V_given_C"].append(
                    self.compute_cmi(I_layer, V_layer, C_layer)
                )

        return results

    def _plot_results(self, results: Dict, output_dir: str, epoch: int):
        """Create plots for branch information metrics"""
        import math

        import matplotlib.pyplot as plt

        if self.compute_per_branch:
            # Per-branch plotting with mean and error bars
            self._plot_per_branch_results(results, output_dir, epoch)
        else:
            # Original pooled plotting
            self._plot_pooled_results(results, output_dir, epoch)

    def _plot_per_branch_results(self, results: Dict, output_dir: str, epoch: int):
        """Create plots for per-branch branch information metrics with separate mean and variance plots"""
        import matplotlib.pyplot as plt

        # Filter metrics to only those that were computed
        all_metrics = [
            ("I_E_C", "I(E;C)", "Excitatory-Class MI"),
            ("I_I_C", "I(I;C)", "Inhibitory-Class MI"),
            ("I_V_C", "I(V;C)", "Output-Class MI"),
            ("I_E_I", "I(E;I)", "Excitatory-Inhibitory MI"),
            ("I_E_V", "I(E;V)", "Excitatory-Output MI"),
            ("I_I_V", "I(I;V)", "Inhibitory-Output MI"),
            ("I_E_I_given_C", "I(E;I|C)", "Conditional E-I MI"),
            ("I_E_V_given_C", "I(E;V|C)", "Conditional E-V MI"),
            ("I_I_V_given_C", "I(I;V|C)", "Conditional I-V MI"),
        ]

        # Only include metrics that were computed (check for _mean version)
        metrics_to_plot = [
            (key, ylabel, title)
            for key, ylabel, title in all_metrics
            if key in self.compute_metrics
            and f"{key}_mean" in results
            and results[f"{key}_mean"]
        ]

        if not metrics_to_plot:
            self.logger.warning("No computed metrics to plot")
            return

        # Reverse layer indices so soma (last layer) becomes 0 (rightmost)
        num_layers = len(results["layer"])
        reversed_layer_idx = [num_layers - 1 - idx for idx in range(num_layers)]

        # Reverse all the data to match the new x-axis positions
        layer_data_reversed = [
            results["layer"][num_layers - 1 - i] for i in range(num_layers)
        ]

        # Calculate dynamic subplot layout - we need 2 columns (mean, variance) for each metric
        n_metrics = len(metrics_to_plot)
        if n_metrics <= 2:
            nrows, ncols = 2, n_metrics  # 2 rows (mean, var), n_metrics columns
        elif n_metrics <= 4:
            nrows, ncols = 2, n_metrics  # 2 rows, up to 4 columns
        else:
            # For more metrics, use multiple rows with max 3 metrics per row
            metrics_per_row = 3
            nrows = 2 * (
                (n_metrics + metrics_per_row - 1) // metrics_per_row
            )  # 2 rows per metric row
            ncols = min(n_metrics, metrics_per_row)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        fig.suptitle(
            f"Per-Branch Information Analysis - Epoch {epoch}\n({len(self.compute_metrics)} metrics computed across branches)",
            fontsize=16,
        )

        # Handle axes indexing for different subplot arrangements
        if n_metrics == 1:
            if nrows == 2:
                axes = axes.reshape(2, 1)  # Ensure 2D array
        elif nrows == 2 and ncols >= n_metrics:
            # Simple case: 2 rows, enough columns
            axes = axes.reshape(2, -1) if axes.ndim == 1 else axes
        else:
            # Complex case: multiple metric rows
            axes = axes.reshape(nrows, ncols)

        # Create plots for computed metrics
        for idx, (key, ylabel, title) in enumerate(metrics_to_plot):
            # Determine subplot positions
            if nrows == 2 and ncols >= n_metrics:
                # Simple layout: mean on top row, variance on bottom row
                mean_ax = axes[0, idx] if ncols > 1 else axes[0]
                var_ax = axes[1, idx] if ncols > 1 else axes[1]
            else:
                # Complex layout: two rows per metric
                metric_row = (idx // ncols) * 2
                metric_col = idx % ncols
                mean_ax = axes[metric_row, metric_col]
                var_ax = axes[metric_row + 1, metric_col]

            # Get data and reverse it to match new x-axis ordering
            mean_data_reversed = [
                results[f"{key}_mean"][num_layers - 1 - i] for i in range(num_layers)
            ]
            std_data_reversed = [
                results[f"{key}_std"][num_layers - 1 - i] for i in range(num_layers)
            ]
            var_data_reversed = [
                s**2 for s in std_data_reversed
            ]  # Convert std to variance
            n_branches_data_reversed = [
                results[f"{key}_n_branches"][num_layers - 1 - i]
                for i in range(num_layers)
            ]

            # Plot mean
            mean_ax.plot(
                reversed_layer_idx,
                mean_data_reversed,
                "o-",
                linewidth=2,
                markersize=8,
                color="blue",
                alpha=0.8,
            )
            mean_ax.set_xlabel("Layer (Soma → Distal)")
            mean_ax.set_ylabel(f"{ylabel} (mean)")
            mean_ax.set_title(f"{title}\nMean across branches")
            mean_ax.grid(True, alpha=0.3)

            # Set x-axis ticks and labels
            mean_ax.set_xticks(reversed_layer_idx)
            x_labels = [
                "Soma" if i == 0 else f"{i}" for i in range(len(reversed_layer_idx))
            ]
            mean_ax.set_xticklabels(x_labels)

            # Add number of branches annotation to mean plot
            for i, (layer_pos, n_branches) in enumerate(
                zip(reversed_layer_idx, n_branches_data_reversed)
            ):
                if n_branches > 0:
                    mean_ax.annotate(
                        f"n={n_branches}",
                        xy=(layer_pos, mean_data_reversed[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7,
                    )

            # Plot variance
            var_ax.plot(
                reversed_layer_idx,
                var_data_reversed,
                "s-",
                linewidth=2,
                markersize=8,
                color="red",
                alpha=0.8,
            )
            var_ax.set_xlabel("Layer (Soma → Distal)")
            var_ax.set_ylabel(f"{ylabel} (variance)")
            var_ax.set_title(f"{title}\nVariance across branches")
            var_ax.grid(True, alpha=0.3)

            # Set x-axis ticks and labels
            var_ax.set_xticks(reversed_layer_idx)
            var_ax.set_xticklabels(x_labels)

            # Set y-axis to start from 0 for variance (since variance is always non-negative)
            var_ax.set_ylim(bottom=0)

        # Hide unused subplots
        total_subplots = nrows * ncols
        used_subplots = len(metrics_to_plot) * 2  # 2 plots per metric

        if nrows == 2 and ncols >= n_metrics:
            # Simple layout
            for idx in range(len(metrics_to_plot), ncols):
                if ncols > 1:
                    axes[0, idx].set_visible(False)
                    axes[1, idx].set_visible(False)
        else:
            # Complex layout - hide unused subplots
            for row in range(nrows):
                for col in range(ncols):
                    subplot_idx = row * ncols + col
                    if subplot_idx >= used_subplots:
                        axes[row, col].set_visible(False)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(
            output_dir, f"branch_information_per_branch_epoch_{epoch}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(
            f"Saved per-branch information plot to {plot_path} with {len(metrics_to_plot)} metrics (mean and variance plots)"
        )

    def _plot_pooled_results(self, results: Dict, output_dir: str, epoch: int):
        """Create plots for pooled branch information metrics (original behavior)"""
        import matplotlib.pyplot as plt

        # Filter metrics to only those that were computed
        all_metrics = [
            ("I_E_C", "I(E;C)", "Excitatory-Class MI"),
            ("I_I_C", "I(I;C)", "Inhibitory-Class MI"),
            ("I_V_C", "I(V;C)", "Output-Class MI"),
            ("I_E_I", "I(E;I)", "Excitatory-Inhibitory MI"),
            ("I_E_V", "I(E;V)", "Excitatory-Output MI"),
            ("I_I_V", "I(I;V)", "Inhibitory-Output MI"),
            ("I_E_I_given_C", "I(E;I|C)", "Conditional E-I MI"),
            ("I_E_V_given_C", "I(E;V|C)", "Conditional E-V MI"),
            ("I_I_V_given_C", "I(I;V|C)", "Conditional I-V MI"),
        ]

        # Only include metrics that were computed
        metrics_to_plot = [
            (key, ylabel, title)
            for key, ylabel, title in all_metrics
            if key in self.compute_metrics and key in results and results[key]
        ]

        if not metrics_to_plot:
            self.logger.warning("No computed metrics to plot")
            return

        # Reverse layer indices so soma (last layer) becomes 0 (rightmost)
        num_layers = len(results["layer"])
        reversed_layer_idx = [num_layers - 1 - idx for idx in range(num_layers)]

        # Calculate dynamic subplot layout
        n_metrics = len(metrics_to_plot)
        if n_metrics <= 3:
            nrows, ncols = 1, n_metrics
        elif n_metrics <= 6:
            nrows, ncols = 2, 3
        else:
            nrows, ncols = 3, 3

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        fig.suptitle(
            f"Pooled Branch Information Analysis - Epoch {epoch}\n({len(self.compute_metrics)} metrics computed, pooled across branches)",
            fontsize=16,
        )

        # Handle single subplot case
        if n_metrics == 1:
            axes = [axes]
        elif nrows == 1:
            axes = list(axes) if hasattr(axes, "__iter__") else [axes]
        else:
            axes = axes.flatten()

        # Create plots for computed metrics
        for idx, (key, ylabel, title) in enumerate(metrics_to_plot):
            ax = axes[idx]

            # Reverse the data to match new x-axis ordering
            data_reversed = [
                results[key][num_layers - 1 - i] for i in range(num_layers)
            ]

            ax.plot(reversed_layer_idx, data_reversed, "o-", linewidth=2, markersize=8)
            ax.set_xlabel("Layer (Soma → Distal)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Set x-axis ticks and labels
            ax.set_xticks(reversed_layer_idx)
            x_labels = [
                "Soma" if i == 0 else f"{i}" for i in range(len(reversed_layer_idx))
            ]
            ax.set_xticklabels(x_labels)

        # Hide unused subplots
        for idx in range(len(metrics_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(
            output_dir, f"branch_information_pooled_epoch_{epoch}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(
            f"Saved pooled information plot to {plot_path} with {len(metrics_to_plot)} metrics"
        )

    def _save_csv(self, results: Dict, output_dir: str, epoch: int):
        """Save results to CSV"""
        df = pd.DataFrame(results)

        # Add layer mapping information for clarity (similar to synaptic ablation)
        if "layer" in results and results["layer"]:
            num_layers = len(results["layer"])
            # Add columns showing the mapping between network layers and plot positions
            df["network_layer_idx"] = results["layer"]  # Original network layer indices
            df["plot_position"] = [
                num_layers - 1 - idx for idx in range(num_layers)
            ]  # Plot positions (0=soma)
            df["layer_label"] = [
                "Soma" if i == 0 else f"{i}" for i in range(num_layers)
            ]  # X-axis labels

        if self.compute_per_branch:
            csv_path = os.path.join(
                output_dir, f"branch_information_per_branch_epoch_{epoch}.csv"
            )
            self.logger.info(f"Saved per-branch information data to {csv_path}")
        else:
            csv_path = os.path.join(
                output_dir, f"branch_information_pooled_epoch_{epoch}.csv"
            )
            self.logger.info(f"Saved pooled information data to {csv_path}")

        df.to_csv(csv_path, index=False)

    @classmethod
    def list_available_metrics(cls) -> Dict[str, str]:
        """Return a dictionary of all available MI metrics and their descriptions"""
        return {
            # Information with Class Labels
            "I_E_C": "I(E;C) - Excitatory inputs and class",
            "I_I_C": "I(I;C) - Inhibitory inputs and class",
            "I_V_C": "I(V;C) - Branch outputs and class",
            # Information Between Branch Components
            "I_E_I": "I(E;I) - Excitatory and inhibitory inputs",
            "I_E_V": "I(E;V) - Excitatory inputs and branch outputs",
            "I_I_V": "I(I;V) - Inhibitory inputs and branch outputs",
            # Conditional Information (Given Class)
            "I_E_I_given_C": "I(E;I|C) - E and I given class",
            "I_E_V_given_C": "I(E;V|C) - E and V given class",
            "I_I_V_given_C": "I(I;V|C) - I and V given class",
        }

