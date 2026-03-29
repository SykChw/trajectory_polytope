import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import torch

from load_model import load_model
from sampling import boundary_sample
from alpha_shape import compute_alpha_shape
from volume import compute_volume


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model("trajectory_model.pt", device)

    ensure_dir("frames")
    ensure_dir("outputs")

    t_values = np.linspace(0, 1, 30)

    volumes = []
    frames = []

    for i, t in enumerate(t_values):
        print(f"t={t:.2f}")

        _, inside_pts = boundary_sample(model, t, device=device)

        shape = compute_alpha_shape(inside_pts)
        vol = compute_volume(shape)

        volumes.append(vol)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if shape is not None:
            try:
                verts = shape.vertices
                faces = shape.faces

                ax.plot_trisurf(
                    verts[:,0], verts[:,1], verts[:,2],
                    triangles=faces,
                    alpha=0.6
                )
            except:
                ax.scatter(inside_pts[:,0], inside_pts[:,1], inside_pts[:,2], s=1)
        else:
            ax.scatter(inside_pts[:,0], inside_pts[:,1], inside_pts[:,2], s=1)

        ax.set_title(f"t={t:.2f}, vol={vol:.3f}")
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])

        fname = f"frames/frame_{i:03d}.png"
        plt.savefig(fname)
        plt.close()

        frames.append(imageio.imread(fname))

    imageio.mimsave("outputs/evolving.gif", frames, fps=5)

    plt.figure()
    plt.plot(t_values, volumes)
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.title("Volume Evolution")
    plt.savefig("outputs/volume.png")

    print("Done.")


if __name__ == "__main__":
    main()