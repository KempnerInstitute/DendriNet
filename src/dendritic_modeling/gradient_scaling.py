


class GradientScalingStrategies:
    @staticmethod
    def none(layer_idx, factor, branches_to_output):
        return 1.0
    
    @staticmethod
    def uniform_scale(layer_idx, factor, branches_to_output):
        return factor
    
    @staticmethod
    def distal_upweight(layer_idx, factor, branches_to_output):
        # average the vector in branches_to_output if it exists
        if branches_to_output is not None and hasattr(branches_to_output, 'grad_scale'):
            return branches_to_output.grad_scale.mean().item() * factor
        else:
            return factor
    
    @staticmethod
    def distal_upweight_by_idx(layer_idx, factor, branches_to_output):
        return (1 + layer_idx) * factor

