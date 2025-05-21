# Set up the logging environment
from dendritic_modeling.logging_config import LoggerManager

logger_manager = LoggerManager()
logger = logger_manager.get_logger()

from dendritic_modeling import pid_utils
from dendritic_modeling.information import (
    build_sr_matrix,
    compute_EI_MI,
    compute_mi_cc,
    compute_mi_cd,
    compute_pairwise_sr,
    compute_synaptic_information,
)

# Import utility functions
from dendritic_modeling.utils import (
    SafeHookModule,
    register_safe_hook,
    safe_backward_hook,
    safe_register_hooks,
)
