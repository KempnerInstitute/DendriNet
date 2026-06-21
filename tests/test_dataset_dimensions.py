from dendritic_modeling.datasets.dimensions import (
    DATASET_CLASS_COUNTS,
    dataset_class_count,
    dataset_input_shape,
    flat_dataset_input_dim,
)


def test_dataset_input_shape_preserves_builtin_flat_shapes():
    assert dataset_input_shape("mnist") == (784,)
    assert dataset_input_shape("fashion_mnist") == (784,)
    assert dataset_input_shape("cifar10") == (3072,)
    assert dataset_input_shape("imagenet", flatten=True) == (150528,)


def test_dataset_input_shape_preserves_imagenet_image_shape():
    assert dataset_input_shape("imagenet", flatten=False) == (3, 224, 224)


def test_dataset_input_shape_preserves_unknown_fallback_behavior():
    assert dataset_input_shape("unknown") is None
    assert dataset_input_shape("unknown", default_input_dim=123) == (123,)
    assert flat_dataset_input_dim("unknown", default_input_dim=784) == 784


def test_flat_dataset_input_dim_preserves_existing_values():
    assert flat_dataset_input_dim("mnist", default_input_dim=1) == 784
    assert flat_dataset_input_dim("cifar10", default_input_dim=1) == 3072
    assert flat_dataset_input_dim("imagenet", default_input_dim=1) == 150528


def test_dataset_class_counts_preserve_gpu_estimator_defaults():
    assert DATASET_CLASS_COUNTS == {
        "mnist": 10,
        "cifar10": 10,
        "imagenet": 1000,
    }


def test_dataset_class_count_returns_builtin_or_explicit_fallback():
    assert dataset_class_count("cifar10", default_class_count=99) == 10
    assert dataset_class_count("custom", default_class_count=17) == 17
    assert dataset_class_count("custom") is None
