import meshio
def readObj(path):
    print("[ INFO] Reading obj mesh", path)
    obj_mesh = meshio.read(path)
    points = obj_mesh.points
    facets = obj_mesh.cells[0].data
    print("Points:")
    print(points)
    print("Triangles:")
    print(facets)
    return points, facets

from meshpy.tet import MeshInfo, build
def writeTetVtk(points, facets, opath):
    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    print("[ INFO] ready to build tet mesh")
    tet_mesh = build(mesh_info)

    # print("[ INFO] Tet mesh generated:")
    # print("Points:")
    # for i, p in enumerate(tet_mesh.points):
    #     print (i, p)
    # print("Elements:")
    # for i, t in enumerate(tet_mesh.elements):
    #     print (i, t)
    print("[ INFO] write vtk to local", opath)

    tet_mesh.write_vtk(opath)

obj_points, obj_facets = readObj("plate.obj")

writeTetVtk(obj_points, obj_facets, "plate.vtk")