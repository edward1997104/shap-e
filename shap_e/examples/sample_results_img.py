import torch
import os
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.util.image_util import load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    output_dir = 'shape-mesh-output'
    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    batch_size = 1
    guidance_scale = 3.0

    # To get the best result, you should remove the background and show only the object of interest to the model.
    image = load_image("example_data/corgi.png")

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    mesh = decode_latent_mesh(xm=xm, latent=latents[0])
    mesh = mesh.tri_mesh()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(
            f'{output_dir}/testing.obj', 'w'
    ) as f:
        mesh.write_obj(f)
