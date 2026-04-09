import open3d as o3d

mesh = o3d.io.read_triangle_mesh("RignetDataset/fbx/13.fbx")
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])
import numpy as np
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

print("Vertices:", vertices.shape)
print("Triangles:", triangles.shape)

pcd = mesh.sample_points_uniformly(number_of_points=1024)

o3d.visualization.draw_geometries([pcd])