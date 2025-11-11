import trimesh

# Загружаем OBJ
mesh = trimesh.load('12221_Cat_v1_l3.obj', force='mesh')

# Проверим, что mesh корректен
print("Вершины:", len(mesh.vertices))
print("Грани:", len(mesh.faces))

# Сохраняем в PLY
mesh.export('cat_fixed.ply')
print("✅ Файл успешно сохранён: cat_fixed.ply")
