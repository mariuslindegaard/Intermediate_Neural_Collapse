from Experiment import Experiment
import os

# If on the MIT Openmind cluster: Place this file under /om2/user/lindegrd/NN_layerwise/src/ for the correct imports and paths

# Paths of finished experiments in the supporting zip-file, containing the experiments with plots shown in the paper.
PATHS = [
    # MLPs
    "logs/matrix/customnets_param_search/2023-01-23T04:47/mlp_large/cifar10/lr_0.01/wd_0.002",
    "logs/matrix/customnets_param_search/2023-01-23T04:47/mlp_large/FashionMNIST/lr_0.01/wd_0.002",
    "logs/matrix/customnets_param_search/2023-01-23T04:47/mlp_large/MNIST/lr_0.02/wd_0.002",
    "logs/matrix/customnets_param_search/2023-01-23T04:47/mlp_large/svhn/lr_0.005/wd_0.002",

    # ConvNet
    "logs/matrix/customnets_param_search/2023-01-23T04:47/convnet_deep/cifar10/lr_0.01/wd_0.002",
    "logs/matrix/customnets_param_search/2023-01-23T04:47/convnet_deep/FashionMNIST/lr_0.01/wd_0.002",
    "logs/matrix/customnets/2023-01-22T10:05/convnet_deep/MNIST",
    "logs/matrix/customnets_param_search/2023-01-23T04:47/convnet_deep/svhn/lr_0.01/wd_0.002",

    # ResNet
    "logs/matrix/papyan_mseloss/resnet50_cifar10_reruns/2023-01-25T03:29/lr_0.03/run_0",
    "logs/matrix/papyan_mseloss/2023-01-20T09:09/FashionMNIST/resnet18",
    "logs/matrix/papyan_mseloss/2023-01-20T09:09/MNIST/resnet18",
    "logs/matrix/papyan_mseloss/2023-01-20T09:09/SVHN/resnet34",
]


def main():
    config_path = "../config/debug.yaml"
    # config_path = os.path.join("..", PATHS[0], "config.yaml")
    exp = Experiment(config_path)
    # Load the latest checkpoint and train until epoch number specified in config.
    # Will not actually train if final epoch is already reached.
    exp.train()

    model = exp.wrapped_model  # contains all fwd hooks etc., check its methods in Models.py
    # base_model = model.base_model
    tmp_inputs, tmp_targets = next(iter(exp.dataset.train_loader))
    out, embeddings = model(tmp_inputs)

    print(f"Targets shape: {out.shape}")
    print(f"Embeddings shape:")
    print(*[f"\t{layer_name}: {activations.shape}" for layer_name, activations in embeddings.items()], sep="\n")


if __name__ == "__main__":
    main()
