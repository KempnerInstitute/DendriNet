import torch
import numpy as np
import logging
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.train_experiments import initialize_model
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.analysis.core.information import InformationAnalyzer

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load config
config = load_config("configs/config_exp.yaml")

# Log the information analysis config
logger.info(f"Information analysis config: {config.analysis.information_analysis}")

# Load datasets
train_dataset, valid_dataset, test_dataset = get_unified_datasets(config.task, config.train, outputs_cfg=config.outputs)

# Get input shape from data
input_sample, _ = train_dataset[0]
config.model.encoder_network.parameters.input_dim = input_sample.numel()
config.model.encoder_network.parameters.input_shape = tuple(input_sample.shape)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, encoder_network = initialize_model(config.model)
model = model.to(device)

# Create analyzer
analyzer = InformationAnalyzer(config.analysis.information_analysis.params)
logger.info(f"Analyzer compute_basic_mi: {analyzer.compute_basic_mi}")
logger.info(f"Analyzer method: {analyzer.method}")

# Attach hooks to collect data
model.eval()
handles = analyzer.attach_forward_hooks(model)

# Get a small batch
batch_size = 100
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

logger.info(f"Input shape: {inputs.shape}")
logger.info(f"Labels shape: {labels.shape}")

# Forward pass
with torch.no_grad():
    outputs = model(inputs)
    logger.info(f"Output shape: {outputs.shape}")

# Remove hooks
analyzer.remove_forward_hooks(handles)

# Check the collected data
logger.info(f"\nNumber of branches: {len(analyzer.data_dict)}")

for branch_name, branch_data in analyzer.data_dict.items():
    logger.info(f"\nBranch: {branch_name}")
    
    if "excitation" in branch_data:
        E = branch_data["excitation"].detach().cpu().numpy()
        logger.info(f"  Excitation shape: {E.shape}")
        logger.info(f"  Excitation range: [{E.min():.4f}, {E.max():.4f}]")
        logger.info(f"  Excitation std: {E.std():.4f}")
        logger.info(f"  Excitation unique values: {len(np.unique(E))}")
        
    if "inhibition" in branch_data:
        I = branch_data["inhibition"].detach().cpu().numpy()
        logger.info(f"  Inhibition shape: {I.shape}")
        logger.info(f"  Inhibition range: [{I.min():.4f}, {I.max():.4f}]")
        logger.info(f"  Inhibition std: {I.std():.4f}")
        logger.info(f"  Inhibition unique values: {len(np.unique(I))}")
        
    if "output" in branch_data:
        V = branch_data["output"].detach().cpu().numpy()
        logger.info(f"  Output shape: {V.shape}")
        logger.info(f"  Output range: [{V.min():.4f}, {V.max():.4f}]")
        logger.info(f"  Output std: {V.std():.4f}")
        logger.info(f"  Output unique values: {len(np.unique(V))}")

# Check labels
C = labels.detach().cpu().numpy()
logger.info(f"\nLabels shape: {C.shape}")
logger.info(f"Labels unique values: {np.unique(C)}")

# Try computing MI manually for one branch
if analyzer.data_dict:
    branch_name = list(analyzer.data_dict.keys())[0]
    branch_data = analyzer.data_dict[branch_name]
    
    E = branch_data["excitation"].detach().cpu().numpy()
    I = branch_data["inhibition"].detach().cpu().numpy()
    V = branch_data["output"].detach().cpu().numpy()
    
    # Reshape for the analyzer
    E = E.reshape(E.shape[0], -1)
    I = I.reshape(I.shape[0], -1)
    V = V.reshape(V.shape[0], -1)
    
    logger.info(f"\nComputing MI for branch {branch_name}...")
    logger.info(f"E shape: {E.shape}, I shape: {I.shape}, V shape: {V.shape}, C shape: {C.shape}")
    
    # Compute metrics
    try:
        results = analyzer.compute_metrics(E, I, V, C)
        
        logger.info("\nResults:")
        logger.info(f"Results keys: {list(results.keys())}")
        
        if "basic_mi" in results:
            logger.info("Basic MI results:")
            for key, value in results["basic_mi"].items():
                logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info("No basic_mi in results!")
            
    except Exception as e:
        logger.error(f"Error computing metrics: {e}", exc_info=True)
            
# Check if the issue is with constant values
logger.info("\nChecking for constant values:")
if analyzer.data_dict:
    branch_data = list(analyzer.data_dict.values())[0]
    for name, tensor_name in [("Excitation", "excitation"), ("Inhibition", "inhibition"), ("Output", "output")]:
        if tensor_name in branch_data:
            data = branch_data[tensor_name].detach().cpu().numpy()
            is_constant = np.allclose(data, data[0])
            logger.info(f"  {name} is constant: {is_constant}")
            if is_constant:
                logger.info(f"    Constant value: {data.flat[0]}")
            else:
                # Check variance per dimension
                var_per_dim = np.var(data, axis=0)
                logger.info(f"    Min variance: {var_per_dim.min():.6f}")
                logger.info(f"    Max variance: {var_per_dim.max():.6f}")
                logger.info(f"    Dimensions with zero variance: {np.sum(var_per_dim == 0)}/{var_per_dim.size}") 