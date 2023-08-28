import open3d as o3d

from nearl.features import fingerprint


def convex_ratio(): 
  shapes = {
    'arrow': o3d.geometry.TriangleMesh.create_arrow(),
    'box': o3d.geometry.TriangleMesh.create_box(),
    'cone':o3d.geometry.TriangleMesh.create_cone(),
    'coord_frame':o3d.geometry.TriangleMesh.create_coordinate_frame(),
    'cylinder':o3d.geometry.TriangleMesh.create_cylinder(),
    'icosahedron':o3d.geometry.TriangleMesh.create_icosahedron(),
    'mobius':o3d.geometry.TriangleMesh.create_mobius(),
    'octahedron':o3d.geometry.TriangleMesh.create_octahedron(),
    'sphere':o3d.geometry.TriangleMesh.create_sphere(),
    'tetrahedron':o3d.geometry.TriangleMesh.create_tetrahedron(),
    'torus':o3d.geometry.TriangleMesh.create_torus(),
    'icosahedron':o3d.geometry.TriangleMesh.create_icosahedron(),
  }

  for sname, mesh in shapes.items():
    mesh.compute_vertex_normals();
    mesh.scale(10, center=mesh.get_center())
    convex = fingerprint.computeconvex(mesh)
    ratio_c_p = len(convex[1].points)/len(convex[0].points)
    print(f"Ratio(C/S) {sname:15}: {ratio_c_p:6.3f} ({len(convex[1].points):>4d}/{len(convex[0].points):<4d})")

if __name__ == "__main__":
  convex_ratio()
