import numpy as np
import open3d as o3d
import pymesh

from . import LabelParser

def convert_scannet_to_ht(src_path: str, dst_path: str):
    assert src_path.endswith('vh_clean_2.labels.ply')
    assert dst_path.endswith('.ply')

    gt_mesh = o3d.io.read_triangle_mesh(src_path)

    gt_mesh_pymesh = pymesh.load_mesh(src_path)

    my_color_arr = np.asarray(gt_mesh.vertex_colors) * 255
    my_color_arr[:10]

    vertex_label_arr = gt_mesh_pymesh.get_attribute('vertex_label')

    my_parser = LabelParser()
    lut = my_parser.get_nyuid_to_ht_map()
    lut[0] = 0

    for i in range(vertex_label_arr.shape[0]):
        if lut[int(vertex_label_arr[i])] == 1:
            my_color_arr[i] = (1, 0, 0)
        else:
            my_color_arr[i] = (0, 0, 0)

    gt_mesh.vertex_colors = o3d.utility.Vector3dVector(my_color_arr)

    o3d.io.write_triangle_mesh(dst_path, gt_mesh, write_ascii=False)
