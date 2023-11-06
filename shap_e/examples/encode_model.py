
import torch

from shap_e.models.download import load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import decode_latent_mesh
import glob
import os
import trimesh
import math
import tyro
from dataclasses import dataclass
import multiprocessing


@dataclass
class Args:
    input_dir : str
    output_dir : str
    workers : int
    blender_path : str = '/home/ubuntu/blender-3.3.1-linux-x64/blender'

args = tyro.cli(Args)

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    worker_idx : int
) -> None:

    while True:
        item = queue.get()
        if item is None:
            break
        try:
            process_one(item, worker_idx)
        except Exception as e:
            print(e)
        queue.task_done()
        with count.get_lock():
            count.value += 1
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
def process_one(model_path, cuda_id):
    torch.cuda.set_device(f'cuda:{cuda_id}')
    device = torch.device(f'cuda:{cuda_id}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
    os.environ['BLENDER_PATH'] = args.blender_path

    xm = load_model('transmitter', device=device)


    file_id = os.path.basename(model_path.split('/')[-2])
    print("Processing", file_id)
    # This may take a few minutes, since it requires rendering the model twice
    # in two different modes.
    batch = load_or_create_multimodal_batch(
        device,
        model_path=model_path,
        mv_light_mode="basic",
        mv_image_size=256,
        cache_dir=f"{args.output_dir}/cached/{file_id}/",
        verbose=True,  # this will show Blender output during renders
    )

    with torch.no_grad():
        latent = xm.encoder.encode_to_bottleneck(batch)
        mesh = decode_latent_mesh(xm=xm, latent=latent)
        mesh = mesh.tri_mesh()
        mesh = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)
        mesh = rotate_around_axis(mesh, axis='x', reverse=False)
        mesh.export(f'{args.output_dir}/reconstructed/{file_id}.obj')
        print(f"Saved {args.output_dir}/reconstructed/{file_id}.obj")





if __name__ == '__main__':
    model_paths = glob.glob(f'{args.input_dir}/*.obj')
    print(f'Found {len(model_paths)} models')

    os.makedirs(f'{args.output_dir}/cached/', exist_ok=True)
    os.makedirs(f'{args.output_dir}/reconstructed/', exist_ok=True)

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    for worker_i in range(args.workers):
        process = multiprocessing.Process(
            target=worker, args=(queue, count)
        )
        process.daemon = True
        process.start()

    for model_path in model_paths:
        queue.put(model_path)

    queue.join()

    for _ in range(args.workers):
        queue.put(None)

    print(f'Processed {count.value} models')




