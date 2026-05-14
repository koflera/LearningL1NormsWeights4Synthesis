# %%
import mrpro
import torch
from adaptive_l1.data.utils import read_split_file
from adaptive_l1.data.augmentation import Compose, RandomFlip, RandomRotate90
from adaptive_l1.data.data_classes import LowFieldMRDataset
from adaptive_l1.models.modl import MoDL, MoDLBlock
from adaptive_l1.models.spatially_adaptive_conv_synthesis import (
    SpatiallyAdaptiveConvSynthesisNet2D, ConvSynthesisParameterMapNetwork2D
)
from adaptive_l1.models.spatially_adaptive_tv import SpatiallyAdaptiveTVNet2D, TVParameterMapNetwork2D
from adaptive_l1.models.unet import UNet

from adaptive_l1.data.utils import load_config

from adaptive_l1.training.trainer import train_model

from pathlib import Path

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_training", type=str, required=True)
parser.add_argument("--config_data", type=str, required=True)

args = parser.parse_args()

cfg_training = load_config(args.config_training)
cfg_data = load_config(args.config_data)

data_dir = cfg_data["data_dir"]
split_dir = cfg_data["split_dir"]

n_training = cfg_training["training"]["n_training"]
n_validation = cfg_training["training"]["n_validation"]
batch_size = cfg_training["training"]["batch_size"]

training_files = read_split_file(
    data_dir=data_dir,
    split_file=split_dir + "fastmri_training.txt",
)[:n_training]

validation_files = read_split_file(
    data_dir=data_dir,
    split_file=split_dir + "fastmri_validation.txt",
)[:n_validation]


training_image_data = mrpro.phantoms.FastMRIImageDataset(
    path=training_files,
    coil_combine=True,
)
validation_image_data = mrpro.phantoms.FastMRIImageDataset(
    path=validation_files,
    coil_combine=True,
)

base_seed = 42
rng = mrpro.utils.RandomGenerator(seed=base_seed)
train_transform = Compose([
    RandomFlip(dim=-1, p=0.5),
    RandomFlip(dim=-2, p=0.5),
    RandomRotate90(p=0.5),
],rng)

noise_std_low, noise_std_high = cfg_training["training"]["noise_std"]["low"], cfg_training["training"]["noise_std"]["high"]
noise_variance_dict = {"low": noise_std_low, "high":noise_std_high}

training_data = LowFieldMRDataset(
    image_dataset=training_image_data,
    noise_variance=noise_variance_dict,
    n_k1=cfg_training["training"]["n_k1"],
    transform=train_transform,
    base_seed=base_seed
)
validation_data = LowFieldMRDataset(
    image_dataset=validation_image_data,
    noise_variance=noise_variance_dict,
    n_k1=cfg_training["training"]["n_k1"],
    base_seed=base_seed,
)

training_loader = torch.utils.data.DataLoader(
    training_data, batch_size=batch_size, shuffle=True
)
validation_loader = torch.utils.data.DataLoader(
    validation_data, batch_size=batch_size, shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cfg_training["model"]["name"] == "modl":
    n_layers = cfg_training["model"]["cnn_block"]["n_layers"]
    n_filters = cfg_training["model"]["cnn_block"]["n_filters"]
    cnn_block = MoDLBlock(n_layers=cfg_training["model"]["cnn_block"]["n_layers"],
                          n_ch_in=cfg_training["model"]["cnn_block"]["n_ch_in"],
                          n_ch_out=cfg_training["model"]["cnn_block"]["n_ch_out"],
                          n_filters=n_filters
                          )
    model = MoDL(cnn_block=cnn_block, n_iterations=cfg_training["model"]["n_iterations"]).to(device)
    
    params_list = [
            {
                "params": model.cnn_block.parameters(),
                "lr": cfg_training["training"]["learning_rate"],
                "weight_decay": cfg_training["training"]["weight_decay"],
            },
            {
                "params": model._regularization_parameter, 
                "lr": cfg_training["training"]["learning_rate_scalar"]
            },
        ]
    hyperparameters_identification = f"n_layers{n_layers}_n_filters{n_filters}"
    
elif cfg_training["model"]["name"] in ["cdl", "tv"]:
    n_enc_stages = cfg_training["model"]["cnn_block"]["n_enc_stages"]
    n_convs_per_stage = cfg_training["model"]["cnn_block"]["n_convs_per_stage"]
    n_filters = cfg_training["model"]["cnn_block"]["n_filters"]
    cnn_block = UNet(
        dim=2,
        n_ch_in=cfg_training["model"]["cnn_block"]["n_ch_in"],
        n_ch_out=cfg_training["model"]["cnn_block"]["n_ch_out"],
        n_enc_stages=n_enc_stages,
        n_convs_per_stage=n_convs_per_stage,
        n_filters=n_filters,
        kernel_size=3,
        pooling_kernel_size=2,
        bias=False,
    )
    if cfg_training["model"]["name"] == "tv":
        parameter_map_network = TVParameterMapNetwork2D(cnn_block = cnn_block)
        model = SpatiallyAdaptiveTVNet2D(
            parameter_map_network=parameter_map_network, 
            n_iterations=cfg_training["model"]["n_iterations"]
        ).to(device)
        
        params_list = [
            {
                "params": model.parameter_map_network.cnn_block.parameters(),
                "lr": cfg_training["training"]["learning_rate"],
                "weight_decay": cfg_training["training"]["weight_decay"],
            },
            {
                "params": [model.parameter_map_network._global_scaling], 
                "lr": cfg_training["training"]["learning_rate_global_scaling"]
            },
        ]
        
        hyperparameters_identification = f"n_enc_stages{n_enc_stages}_"\
            f"n_convs_per_stage{n_convs_per_stage}_n_filters{n_filters}"
        
    else:
        n_conv_dictionary_filters = cfg_training["model"]["conv_dictionary"]["n_conv_kernel_filters"]
        kernel_size = cfg_training["model"]["conv_dictionary"]["n_conv_kernel_size"]
        sparsity_param = cfg_training["model"]["conv_dictionary"]["sparsity_param"]
        lowpass_param = cfg_training["model"]["conv_dictionary"]["lowpass_param"]
        
        conv_dictionary_kernel_dir = cfg_data["conv_dictionary_dir"]
        kernel_fname = Path(f"K{n_conv_dictionary_filters}_k{kernel_size}x{kernel_size}_"\
                        f"sparsity_param{str(sparsity_param).replace('.','_')}_"\
                        f"lowpass_param{str(lowpass_param).replace('.','_')}.pt")
        conv_dictionary_kernel = torch.load(Path(conv_dictionary_kernel_dir) / kernel_fname)
        
        parameter_map_network = ConvSynthesisParameterMapNetwork2D(cnn_block = cnn_block)
        model = SpatiallyAdaptiveConvSynthesisNet2D(
            kernel = conv_dictionary_kernel,
            parameter_map_network=parameter_map_network,
            n_iterations=cfg_training["model"]["n_iterations"]
        ).to(device)
        
        params_list = [
            {
                "params": list(model.parameter_map_network.cnn_block.parameters()),
                "lr": cfg_training["training"]["learning_rate"],
                "weight_decay": cfg_training["training"]["weight_decay"],
            },
            {
                "params": [model._low_pass_filtering_parameter], 
                "lr": cfg_training["training"]["learning_rate_low_pass_param"]
            },
        ]
        
        hyperparameters_identification = f"n_enc_stages{n_enc_stages}_"\
            f"n_convs_per_stage{n_convs_per_stage}_n_filters{n_filters}_"\
            f"n_conv_dictionary_filters{n_conv_dictionary_filters}_conv_kernel_size{kernel_size}"
else:
    raise ValueError(f"Model name should be either 'modl', 'cdl', or 'tv', but got{cfg_training['model']['name']}")


optimizer = torch.optim.Adam(params=params_list)

run_dir_name = Path(f"{cfg_training["model"]["name"]}")
run_dir = Path("runs") / run_dir_name  / hyperparameters_identification
run_dir.mkdir(parents=True, exist_ok=True)

experiment_name = f"{cfg_training["model"]["name"]}_{hyperparameters_identification}"

# collect all hyperparameters defining the experiment
config = {
        name: value
        for name, value in locals().items()
        if isinstance(value, (int, float, tuple, str, bool))
    }

model = train_model(
    model = model, 
    training_loader=training_loader, 
    validation_loader=validation_loader, 
    optimizer=optimizer,
    loss_function = torch.nn.MSELoss(),
    device = device,
    n_epochs=cfg_training["training"]["n_epochs"],
    run_dir=run_dir,
    config=config
)