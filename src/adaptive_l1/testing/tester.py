import mrpro
import torch

from tqdm import tqdm
from adaptive_l1.testing.statistics import brain_mask, psnr

import csv


def test_model(model, test_loader, device, run_dir, metrics_fname):

    metrics_file = run_dir / f"{metrics_fname}.csv"
    model.eval()
    with torch.no_grad():

        mse_values_sum = 0.0
        ssim_values_sum = 0.0
        psnr_values_sum = 0.0

        test_n_samples = 0

        for batch in tqdm(
            test_loader,
            desc="test loop",
            position=0,
            leave=False,
            disable=False,
        ):
            kdata = batch["kdata"].to(device)
            adjoint = batch["adjoint"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            mask_operator = mrpro.operators.CartesianMaskingOp(mask)

            recon = model(adjoint, kdata, mask_operator)

            batch_size = target.shape[0]
            target_mask = torch.cat(
                [
                    brain_mask(target[k].abs().squeeze().squeeze(), threshold=0.1)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    for k in range(batch_size)
                ],
                dim=0,
            )
            mse_metric = mrpro.operators.functionals.MSE(
                target=target, weight=target_mask
            )
            (mse_value,) = mse_metric(recon)

            ssim_metric = mrpro.operators.functionals.SSIM(
                target=target,
                weight=target_mask,
            )
            (ssim_value,) = ssim_metric(recon)

            psnr_value = psnr(target, recon, weight=target_mask, reduction="mean")

            mse_values_sum += mse_value.item() * batch_size
            ssim_values_sum += ssim_value.item() * batch_size
            psnr_values_sum += psnr_value.item() * batch_size

            test_n_samples += batch_size

        mse_average = mse_values_sum / test_n_samples
        ssim_average = ssim_values_sum / test_n_samples
        psnr_average = psnr_values_sum / test_n_samples

        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ssim:", ssim_average])
            writer.writerow(["psnr:", psnr_average])
            writer.writerow(["mse:", mse_average])
