import trimesh

def compute_volume(shape):
    if shape is None:
        return 0.0

    try:
        mesh = trimesh.Trimesh(vertices=shape.vertices, faces=shape.faces)
        return mesh.volume
    except:
        return 0.0