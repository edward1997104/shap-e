import torch
import os
import trimesh
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.util.image_util import load_image
import math
import cloudpathlib
import boto3
import tempfile
import pickle
import tyro
from dataclasses import dataclass
import multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)


@dataclass
class Args:
    workers : int = 8
    cuda_cnt : int = 8
    render_resolution : int = 384
    output_dir : str = 'shape-mesh-output-val'
    batch_size : int  = 1
    guidance_scale : float = 3.0

args = tyro.cli(Args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rotate_around_axis(mesh, axis = 'x', reverse = False):
    if reverse:
        angle = math.pi / 2
    else:
        angle = -math.pi / 2

    if axis == 'x':
        direction = [1, 0, 0]
    elif axis == 'y':
        direction = [0, 1, 0]
    else:
        direction = [0, 0, 1]

    center = mesh.centroid

    rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)

    mesh.apply_transform(rot_matrix)

    return mesh


img_bucket_mapping = {
            f'ABC_renders_{args.render_resolution}': 'abc-renders',
            f'BuildingNet_renders_{args.render_resolution}': 'buildingnet-renders',
            f'Fusion_renders_{args.render_resolution}': 'fusion-renders',
            f'ModelNet40_renders_{args.render_resolution}': 'modelnet40-renders',
            f'Objaverse_renders_{args.render_resolution}': '000-objaverse-renders',
            f'ShapeNet_V2_renders_{args.render_resolution}': 'shapenet-v2-renders',
            f'Thingi10K_renders_{args.render_resolution}': 'thingi10k-renders',
            f'Thingiverse_renders_{args.render_resolution}': 'thingiverse-renders',
            f'Github_renders_{args.render_resolution}': 'github-renders',
            f'Infinigen_renders_{args.render_resolution}': 'infinigen-renders-us',
            f'Smpl_renders_{args.render_resolution}': 'smpl-renders',
            f'Smal_renders_{args.render_resolution}': 'smal-renders',
            f'Coma_renders_{args.render_resolution}': 'coma-renders',
            f'DeformingThings4D_renders_{args.render_resolution}': 'deformingthings4d-renders',
            f'Abo_renders_{args.render_resolution}': 'abo-renders',
            f'Fg3d_renders_{args.render_resolution}': 'fg3d-renders',
            f'House3d_renders_{args.render_resolution}': 'house3d-renders',
            f'Toy4k_renders_{args.render_resolution}': 'toy4k-renders',
            f'Gso_renders_{args.render_resolution}': 'gso-renders',
            f'3DFuture_renders_{args.render_resolution}': '3dfuture-renders',
        }

def process_one(xm, model, diffusion, img):

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset, id = img
        obj_path = f'{args.output_dir}/{id}.obj'
        if os.path.exists(obj_path):
            print("File exists: ", obj_path)
            return

        dataset = dataset.replace("_wavelet_latents", f'_renders_{args.render_resolution}').replace("_fixed", "")
        bucket = img_bucket_mapping[dataset]
        print("start processing: ", img)

        cloudpath = cloudpathlib.CloudPath(f's3://{bucket}/{id}/img/018.png')
        save_filename = f"{id}.png"
        save_path = os.path.join(tmp_dir, save_filename)
        cloudpath.download_to(save_path)

        # To get the best result, you should remove the background and show only the object of interest to the model.
        image = load_image(save_path)

        latents = sample_latents(
            batch_size=args.batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=args.guidance_scale,
            model_kwargs=dict(images=[image] * args.batch_size),
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
        mesh = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)
        mesh = rotate_around_axis(mesh, axis='x', reverse=False)

        mesh.export()
        print("Saved: ", f'{args.output_dir}/{id}.obj')


def worker(queue, count, worker_i):

    cuda_id = worker_i % args.cuda_cnt
    # torch.cuda.set_device(f'cuda:{cuda_id}')
    device = torch.device(f'cuda:{cuda_id}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)

    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    while True:
        item = queue.get()
        if item is None:
            break
        try:
            process_one(xm, model, diffusion, item)
        except Exception as e:
            print(e)
        queue.task_done()
        with count.get_lock():
            count.value += 1

if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)

    file_path_pickle = 'file_paths.pkl'
    with open(file_path_pickle, 'rb') as f:
        img_lists = pickle.load(f)

    print("Number of images: ", len(img_lists))

    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    for worker_i in range(args.workers):
        process = multiprocessing.Process(
            target=worker, args=(queue, count, worker_i)
        )
        # process.daemon = True
        process.start()

    for img in img_lists:
        queue.put(img)

    queue.join()

    for _ in range(args.workers):
        queue.put(None)

    print("Processing finished!")



