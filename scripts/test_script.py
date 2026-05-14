# %%
import mrpro
import torch
from adaptive_l1.data.utils import read_split_file
from adaptive_l1.data.data_classes import LowFieldMRDataset
from adaptive_l1.models.modl import MoDL, MoDLBlock
from adaptive_l1.models.spatially_adaptive_conv_synthesis import (
    SpatiallyAdaptiveConvSynthesisNet2D,
    ConvSynthesisParameterMapNetwork2D,
)
from adaptive_l1.models.spatially_adaptive_tv import (
    SpatiallyAdaptiveTVNet2D,
    TVParameterMapNetwork2D,
)
from adaptive_l1.models.unet import UNet

from adaptive_l1.data.utils import load_config

from adaptive_l1.testing.tester import test_model

from pathlib import Path
from datetime import datetime

from tqdm import tqdm

# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_test", type=str, required=True)
parser.add_argument("--config_data", type=str, required=True)

args = parser.parse_args()

cfg_test = load_config(args.config_test)
cfg_data = load_config(args.config_data)

noise_std = cfg_test["testing"]["noise_std"]
if not (isinstance(noise_std, float) or isinstance(noise_std, int)):
    raise ValueError(f"noise standard deviation should be float or integer, got {noise_std}")

n_k1 = cfg_test["testing"]["n_k1"]
if not (isinstance(n_k1, float) or isinstance(n_k1, int)):
    raise ValueError(f"A single number of n_k1 samples (`float` or `int`) should be "
                     f"given for model testing; got {type(n_k1)}")

data_dir = cfg_data["data_dir"]
split_dir = cfg_data["split_dir"]

batch_size = cfg_test["testing"]["batch_size"]

n_test = cfg_test["testing"]["n_test"]
test_files = read_split_file(
    data_dir=data_dir,
    split_file=split_dir + "fastmri_test.txt",
)[:n_test]

test_image_data = mrpro.phantoms.FastMRIImageDataset(
    path=test_files,
    coil_combine=True,
)

base_seed = 42

test_data = LowFieldMRDataset(
    image_dataset=test_image_data,
    noise_variance=noise_std ** 2,
    n_k1=n_k1,
    base_seed=base_seed,
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cfg_test["model"]["name"] == "modl":
    n_layers = cfg_test["model"]["cnn_block"]["n_layers"]
    n_filters = cfg_test["model"]["cnn_block"]["n_filters"]
    cnn_block = MoDLBlock(n_layers=cfg_test["model"]["cnn_block"]["n_layers"],
                          n_ch_in=cfg_test["model"]["cnn_block"]["n_ch_in"],
                          n_ch_out=cfg_test["model"]["cnn_block"]["n_ch_out"],
                          n_filters=n_filters
                          )
    model = MoDL(cnn_block=cnn_block, n_iterations=cfg_test["model"]["n_iterations"]).to(device)

    hyperparameters_identification = f"n_layers{n_layers}_n_filters{n_filters}"
    
elif cfg_test["model"]["name"] in ["cdl", "tv"]:
    n_enc_stages = cfg_test["model"]["cnn_block"]["n_enc_stages"]
    n_convs_per_stage = cfg_test["model"]["cnn_block"]["n_convs_per_stage"]
    n_filters = cfg_test["model"]["cnn_block"]["n_filters"]
    cnn_block = UNet(
        dim=2,
        n_ch_in=cfg_test["model"]["cnn_block"]["n_ch_in"],
        n_ch_out=cfg_test["model"]["cnn_block"]["n_ch_out"],
        n_enc_stages=n_enc_stages,
        n_convs_per_stage=n_convs_per_stage,
        n_filters=n_filters,
        kernel_size=3,
        pooling_kernel_size=2,
        bias=False,
    )
    if cfg_test["model"]["name"] == "tv":
        parameter_map_network = TVParameterMapNetwork2D(cnn_block = cnn_block)
        model = SpatiallyAdaptiveTVNet2D(
            parameter_map_network=parameter_map_network, 
            n_iterations=cfg_test["model"]["n_iterations"]
        ).to(device)
        
        hyperparameters_identification = f"n_enc_stages{n_enc_stages}_"\
            f"n_convs_per_stage{n_convs_per_stage}_n_filters{n_filters}"
        
    else:
        n_conv_dictionary_filters = cfg_test["model"]["conv_dictionary"]["n_conv_kernel_filters"]
        kernel_size = cfg_test["model"]["conv_dictionary"]["n_conv_kernel_size"]
        sparsity_param = cfg_test["model"]["conv_dictionary"]["sparsity_param"]
        lowpass_param = cfg_test["model"]["conv_dictionary"]["lowpass_param"]
        
        conv_dictionary_kernel_dir = cfg_data["conv_dictionary_dir"]
        kernel_fname = Path(f"K{n_conv_dictionary_filters}_k{kernel_size}x{kernel_size}_"\
                        f"sparsity_param{str(sparsity_param).replace(".","_")}_"\
                        f"lowpass_param{str(lowpass_param).replace(".","_")}.pt")
        conv_dictionary_kernel = torch.load(Path(conv_dictionary_kernel_dir) / kernel_fname)
        
        parameter_map_network = ConvSynthesisParameterMapNetwork2D(cnn_block = cnn_block)
        model = SpatiallyAdaptiveConvSynthesisNet2D(
            kernel = conv_dictionary_kernel,
            parameter_map_network=parameter_map_network,
            n_iterations=cfg_test["model"]["n_iterations"]
        ).to(device)
        
        hyperparameters_identification = f"n_enc_stages{n_enc_stages}_"\
            f"n_convs_per_stage{n_convs_per_stage}_n_filters{n_filters}_"\
            f"n_conv_dictionary_filters{n_conv_dictionary_filters}_conv_kernel_size{kernel_size}"
else:
    raise ValueError(f"Model name should be either 'modl', 'cdl', or 'tv', but got{cfg_test["model"]["name"]}")


run_dir_name = Path(f"{cfg_test["model"]["name"]}")
run_dir = Path("runs") / run_dir_name  / hyperparameters_identification
run_dir.mkdir(parents=True, exist_ok=True)
checkpoint = torch.load(run_dir / "model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

metrics_fname = f"test_metrics_{cfg_test["testing"]["noise_std"]}"
test_model(model, test_loader, device, run_dir, metrics_fname)