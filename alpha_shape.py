import alphashape

def compute_alpha_shape(points, alpha=None):
    if len(points) < 10:
        return None

    if alpha is None:
        alpha = alphashape.optimizealpha(points)

    try:
        shape = alphashape.alphashape(points, alpha)
        return shape
    except:
        return None