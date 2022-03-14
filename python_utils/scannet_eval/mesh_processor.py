import os
import numpy as np
import open3d as o3d

class ra_slam_mesh_reader(object):
    def __init__(self, mesh_dir) -> None:
        super().__init__()

        # Load file into numpy arrays
        self.vertices_file = os.path.join(mesh_dir, "mesh_vertices.bin")
        self.indices_file = os.path.join(mesh_dir, "mesh_indices.bin")
        self.ht_prob_file = os.path.join(mesh_dir, "mesh_vertices_prob.bin")
        self.vertices_arr = np.fromfile(self.vertices_file, dtype = np.float32).reshape(-1, 3)
        self.indices_arr = np.fromfile(self.indices_file, dtype = np.int32).reshape(-1, 3)
        self.ht_prob_arr = np.fromfile(self.ht_prob_file, dtype = np.float32).reshape(-1,)

        # Sanity checks
        assert np.max(self.ht_prob_arr) <= 1
        assert np.min(self.ht_prob_arr) >= 0
        assert self.ht_prob_arr.shape[0] == self.vertices_arr.shape[0]

        self.init_mesh()
    
    def init_mesh(self):
        self.construct_geometric_mesh()
        self.fill_mesh_w_raw_prob()
    
    def construct_geometric_mesh(self):
        # Grayscale only
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(self.vertices_arr)
        self.mesh.triangles = o3d.utility.Vector3iVector(self.indices_arr)
        self.mesh.compute_vertex_normals()
    
    def fill_mesh_w_raw_prob(self):
        # Color mesh using high-touch probability
        color_arr = np.zeros((self.ht_prob_arr.shape[0], 3), dtype = np.float32)
        color_arr[:, 0] = self.ht_prob_arr

        self.mesh.vertex_colors = o3d.utility.Vector3dVector(color_arr)
    
    def vertex_clustering_down_sample(self, voxel_size=0.05):
        # Vertex Clustering
        self.mesh = self.mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average
        )

    def decimation_downsample(self, target_num_triangles=100000):
        self.mesh = self.mesh.simplify_quadric_decimation(target_number_of_triangles=target_num_triangles)
    
    def visualize_mesh(self):
        o3d.visualization.draw_geometries([self.mesh])
    
    def get_num_vertices(self):
        return len(self.mesh.vertices)
    
    def get_num_triangles(self):
        return len(self.mesh.triangles)
    
    def save_to_file(self, file_path):
        o3d.io.write_triangle_mesh(file_path, self.mesh, write_ascii=False)
