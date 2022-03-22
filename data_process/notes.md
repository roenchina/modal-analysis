使用meshio读取obj的方法：

```cmd
>>> python
Python 3.7.7 (default, May  6 2020, 11:45:54) [MSC v.1916 64 bit (AMD64)] :: 
Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import meshio
>>> mesh = meshio.read("./r02.obj")
>>> mesh.points
array([[-0.70710678, -0.70710678,  0.        ], 
       [-0.70710678,  0.70710678,  0.        ], 
       [ 0.70710678,  0.70710678,  0.        ], 
       [ 0.70710678, -0.70710678,  0.        ], 
       [ 0.        ,  0.        , -1.        ], 
       [ 0.        ,  0.        ,  1.        ]])
>>> mesh.cells
[<meshio CellBlock, type: triangle, num cells: 8, tags: []>]
>>> mesh.cells[0]
<meshio CellBlock, type: triangle, num cells: 8, tags: []>
>>> mesh.cells[0].data
array([[0, 1, 4],
       [0, 4, 3],
       [0, 3, 5],
       [0, 5, 1],
       [1, 2, 4],
       [1, 5, 2],
       [2, 3, 4],
       [2, 5, 3]])
>>> mesh.cells[1].data 
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: list index out of range
```

