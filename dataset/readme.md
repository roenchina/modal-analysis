## 我要做什么-basic

1. 修改10K的脚本，使得它可以给每个模型文件名前缀其对应的ID
2. 写一个脚本给每个stl模型建一个子文件夹，并把stl模型移进该文件夹
3. 写一个脚本对每个文件夹内的stl模型执行”obj2tetvtk.py“，即读取-归一化-tet网格化，同名.vtk保存到该文件夹内
4. 写一个脚本对每个文件夹内的vtk模型执行M，K的构造，默认材料0（陶瓷）



## 我要做什么-test

> 我不确定要固定多少个点

1. 写一个脚本，固定前k个点，对调整后的M，K执行广义特征值分解
   - 广义特征值分解用什么库？
     - 可以限定特征值范围
2. 写一个脚本，在模型文件夹内建立fixk文件夹，把分解后的特征值和对应的特征向量保存到该目录下



## 最后的文件目录样式

```
dataset
├─37627_laser
│  ├─37627_laser.stl
│  ├─37627_laser.vtk
│  ├─mass.txt
│  ├─stiff.txt
│  ├─fix10
│  │  ├─evals.txt
│  │  └─evecs.txt
│  └─fix20
│     ├─evals.txt
│     └─evecs.txt
│
├─42727_DP
│  ├─42727_DP.stl
│  ├─42727_DP.vtk
│  ├─mass.txt
│  ├─stiff.txt
│  ├─fix10
│  │  ├─evals.txt
│  │  └─evecs.txt
│  └─fix20
│     ├─evals.txt
│     └─evecs.txt
...
|
└─43780_WIMM
   ├─43780_WIMM.stl
   ├─43780_WIMM.vtk
   ├─mass.txt
   ├─stiff.txt
   ├─fix10
   │  ├─evals.txt
   │  └─evecs.txt
   └─fix20
      ├─evals.txt
      └─evecs.txt
```





