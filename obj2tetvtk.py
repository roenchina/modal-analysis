import meshio
import argparse, os


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
    print("[ INFO] writing vtk to:", opath)
    tet_mesh.write_vtk(opath)

def normalize(points):
    max_val = max(abs(points.max()), abs(points.min()))
    points = points / max_val * 0.5
    return points

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='./model/r01.obj', help='Input file path [default: \'./r01.obj\']')
parser.add_argument('-o', '--output', type=str, default='./model/r01.vtk', help='Output file path [default: \'./r01.vtk\']')
FLAGS = parser.parse_args()

obj_points, obj_facets = readObj(FLAGS.input)
obj_points = normalize(obj_points)
writeTetVtk(obj_points, obj_facets, FLAGS.output)