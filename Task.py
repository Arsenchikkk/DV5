import open3d as o3d
import numpy as np

# === 1. ЗАГРУЗКА И ВИЗУАЛИЗАЦИЯ ===
print("=== 1. Загрузка и визуализация модели ===")
mesh = o3d.io.read_triangle_mesh("cat_fixed.ply")
o3d.visualization.draw_geometries([mesh])
print("Вершины:", np.asarray(mesh.vertices).shape[0])
print("Треугольники:", np.asarray(mesh.triangles).shape[0])
print("Есть цвета:", mesh.has_vertex_colors())
print("Есть нормали:", mesh.has_vertex_normals())

mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], window_name="1. Исходная модель")

# === 2. ПРЕОБРАЗОВАНИЕ В ОБЛАКО ТОЧЕК ===
print("\n=== 2. Преобразование в облако точек ===")

pcd = mesh.sample_points_uniformly(number_of_points=20000)
print("Количество точек:", np.asarray(pcd.points).shape[0])
print("Есть цвета:", pcd.has_colors())

o3d.visualization.draw_geometries([pcd], window_name="2. Облако точек")

# === 3. РЕКОНСТРУКЦИЯ ПОВЕРХНОСТИ ===
print("\n=== 3. Реконструкция поверхности ===")

mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9)
bbox = pcd.get_axis_aligned_bounding_box()
mesh_crop = mesh_poisson.crop(bbox)

print("Вершины:", np.asarray(mesh_crop.vertices).shape[0])
print("Треугольники:", np.asarray(mesh_crop.triangles).shape[0])
print("Есть цвета:", mesh_crop.has_vertex_colors())

o3d.visualization.draw_geometries([mesh_crop], window_name="3. Реконструкция поверхности")

# === 4. ВОКСЕЛИЗАЦИЯ ===
print("\n=== 4. Вокселизация ===")

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
print("Количество вокселей:", len(voxel_grid.get_voxels()))
o3d.visualization.draw_geometries([voxel_grid], window_name="4. Вокселизация")

# === 5. ДОБАВЛЕНИЕ ПЛОСКОСТИ ===
print("\n=== 5. Добавление плоскости ===")

# Создаём большую плоскость под моделью (примерно как пол)
plane = o3d.geometry.TriangleMesh.create_box(width=30,height=1., depth=45.0)
plane.translate((-3, -0.5, -3))
plane.paint_uniform_color([0.7, 0.7, 0.7])  # немного светлее для контраста


o3d.visualization.draw_geometries([mesh_crop, plane], window_name="5. Объект с плоскостью")

# === 6. ОБРЕЗКА ПО ПОВЕРХНОСТИ ===
print("\n=== 6. Обрезка по плоскости ===")

points = np.asarray(pcd.points)
mask = points[:, 1] > -0.2  # оставим точки выше плоскости
pcd_clipped = o3d.geometry.PointCloud()
pcd_clipped.points = o3d.utility.Vector3dVector(points[mask])
pcd_clipped.paint_uniform_color([1, 0.6, 0])


print("Оставшиеся точки:", np.asarray(pcd_clipped.points).shape[0])
o3d.visualization.draw_geometries([pcd_clipped], window_name="6. После обрезки")

# === 7. ЦВЕТ И ЭКСТРЕМУМЫ ===
print("\n=== 7. Работа с цветом и экстремумами ===")

points = np.asarray(pcd_clipped.points)
z_vals = points[:, 2]
z_min, z_max = z_vals.min(), z_vals.max()

# Градиент по оси Z
colors = (z_vals - z_min) / (z_max - z_min)
pcd_clipped.colors = o3d.utility.Vector3dVector(np.c_[colors, 0.3*colors, 1 - colors])

# Найдём экстремальные точки
min_idx = np.argmin(z_vals)
max_idx = np.argmax(z_vals)
p_min, p_max = points[min_idx], points[max_idx]

# Добавим сферы
sphere_min = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
sphere_min.translate(p_min)
sphere_min.paint_uniform_color([0, 1, 0])  # зелёная — минимум

sphere_max = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
sphere_max.translate(p_max)
sphere_max.paint_uniform_color([1, 0, 0])  # красная — максимум

print("Минимум по Z:", p_min)
print("Максимум по Z:", p_max)

o3d.visualization.draw_geometries(
    [pcd_clipped, sphere_min, sphere_max],
    window_name="7. Цвет и экстремумы"
)
