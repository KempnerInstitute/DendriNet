

from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNNUpsample,
    CNNDownsample
)

from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    MatchedParamMLP
)

from dendritic_modeling.networks.architectures.classical.autoencoder import (
    MLPAutoencoder,
    CNNAutoencoder,
    MLPVariationalAutoencoder,
    CNNVariationalAutoencoder
)

from dendritic_modeling.networks.architectures.classical.identity import Identity

from dendritic_modeling.networks.architectures.excitation_inhibition import ExcitationInhibitionNetwork

def get_architecture(type: str, parameters: dict):
    type = type.lower()
    if type == "alexnet":
        return AlexNet(**parameters)
    elif type in ["cnn", "cnn_down", "cnndownsample", "cnn_downsample"]:
        return CNNDownsample(**parameters)
    elif type in ["cnn_up", "cnnupsample", "cnn_upsample"]:
        return CNNUpsample(**parameters)
    elif type == "mlp":
        return MLP(**parameters)
    elif type == "matchedparammlp":
        return MatchedParamMLP(**parameters)
    elif type == "mlpautoencoder":
        return MLPAutoencoder(**parameters)
    elif type == "cnnautoencoder":
        return CNNAutoencoder(**parameters)
    elif type == "mlpvariationalautoencoder":
        return MLPVariationalAutoencoder(**parameters)
    elif type == "cnnvariationalautoencoder":
        return CNNVariationalAutoencoder(**parameters)
    elif type in ["identity", "none", None]:
        return Identity(**parameters)
    elif type == "einet":
        return ExcitationInhibitionNetwork(**parameters)
    else:
        raise ValueError(f"Invalid architecture type: {type}")

