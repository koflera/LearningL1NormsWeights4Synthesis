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

data_dir = cfg_data["data_dir"]
split_dir = cfg_data["split_dir"]

batch_size = cfg_test["testing"]["batch_size"]

test_files = read_split_file(
    data_dir=data_dir,
    split_file=split_dir + "fastmri_test.txt",
)[:50]

test_image_data = mrpro.phantoms.FastMRIImageDataset(
    path=test_files,
    coil_combine=True,
)

base_seed = 42

test_data = LowFieldMRDataset(
    image_dataset=test_image_data,
    noise_variance=cfg_test["testing"]["noise_variance"],
    n_k1=cfg_test["testing"]["n_k1"],
    base_seed=base_seed,
)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cfg_test["model"]["name"] == "modl":
    cnn_block = MoDLBlock(n_layers=cfg_test["model"]["cnn_block"]["n_layers"],
                          n_ch_in=cfg_test["model"]["cnn_block"]["n_ch_in"],
                          n_ch_out=cfg_test["model"]["cnn_block"]["n_ch_out"],
                          n_filters=cfg_test["model"]["cnn_block"]["n_filters"]
                          )
    model = MoDL(cnn_block=cnn_block, n_iterations=cfg_test["model"]["n_iterations"]).to(device)
    
elif cfg_test["model"]["name"] in ["cdl", "tv"]:
    cnn_block = UNet(
        dim=2,
        n_ch_in=cfg_test["cnn_block"]["n_ch_in"],
        n_ch_out=cfg_test["cnn_block"]["n_ch_out"],
        n_enc_stages=cfg_test["cnn_block"]["n_enc_stages"],
        n_convs_per_stage=cfg_test["cnn_block"]["n_convs_per_stage"],
        n_filters=cfg_test["cnn_block"]["n_filters"],
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
        
    else:
        parameter_map_network = ConvSynthesisParameterMapNetwork2D(cnn_block = cnn_block)
        model = SpatiallyAdaptiveConvSynthesisNet2D(
            parameter_map_network=parameter_map_network,
            n_iterations=cfg_test["model"]["n_iterations"]
        ).to(device)
else:
    raise ValueError(f"Model name should be either 'modl', 'cdl', or 'tv', but got{cfg_test["model"]["name"]}")


run_dir_name = Path(f"{cfg_test["model"]["name"]}")
run_dir = Path("runs") / run_dir_name # Path.joinpath(run_dir_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

checkpoint = torch.load(run_dir / "model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

metrics_fname = f"test_metrics_{cfg_test["testing"]["noise_variance"]}"
test_model(model, test_loader, device, run_dir, metrics_fname)