import os
import shutil


# 获取目录下的stl文件名
def osWalk(path):
    stl_list = []
    dir_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            print('[ FILE]', file)
            postfix = os.path.splitext(file)[1]
            if postfix == '.stl' or postfix == '.STL':
                stl_list.append(os.path.splitext(file)[0])
        for dir in dirs:
            print('[ DIR]', dir)
            dir_list.append(dir)
    print('[ INFO] We found stl_list and dir_list:')
    print(stl_list)
    print(dir_list)
    # return stl_list
    return stl_list, dir_list

def examination(path):
    print('[ EXAM] Exmination begins at', path)
    stl_list, dir_list = osWalk(path)
    if( not stl_list == []):
        print('[ EXAM] raw stl file exists:', stl_list)
    
    for dir in dir_list:
        source_stl = os.path.join(path, dir, dir+'.stl')
        target_vtk = os.path.join(path, dir, dir+'.vtk')
        mass_txt = os.path.join(path, dir, 'mass.txt')
        stiff_txt = os.path.join(path, dir, 'stiff.txt')

        if( not os.path.isfile(source_stl)):
            print('[ EXAM] stl file missing for ', dir)

        if( not os.path.isfile(target_vtk)):
            print('[ EXAM] vtk file missing for ', dir)

        if(os.path.isfile(mass_txt) and os.path.isfile(stiff_txt)):
            pass
        else:
            print('[ EXAM] M K matrix missing for ', dir)

    print('[ EXAM] Exmination Over')


# 给每个stl模型建一个子文件夹，并把stl模型移进该文件夹
base_dir = os.getcwd()
stl_list, dir_list = osWalk(base_dir)
for stl in stl_list:
    new_dir = os.path.join(base_dir, stl)
    print('[ INFO] making new dir:', new_dir)
    os.mkdir(new_dir)

    print('[ INFO] moving stl file to new dir')
    shutil.move(stl+'.stl', new_dir)

    dir_list.append(new_dir)


print('[ INFO] now all dir:', dir_list)
for dir in dir_list:
    source_stl = os.path.join(base_dir, dir, dir+'.stl')
    target_vtk = os.path.join(base_dir, dir, dir+'.vtk')
    mass_txt = os.path.join(base_dir, dir, 'mass.txt')
    stiff_txt = os.path.join(base_dir, dir, 'stiff.txt')

    if(os.path.isfile(mass_txt) and os.path.isfile(stiff_txt)):
        print('[ INFO] M and K already exist at ', dir)
        continue

    if( os.path.isfile(source_stl) and not os.path.isfile(target_vtk)):
        print('[ INFO] transforming stl to vtk for ', dir)
        ##### 执行vtk操作 #####
        open(target_vtk, 'w')

    if(os.path.isfile(target_vtk)):
        print('[ INFO] construting M and K for ', dir)
        ##### 执行modal analysis #####
        open(mass_txt, 'w')
        open(stiff_txt, 'w')
    else:
        print('[ ERROR] no vtk exists at', dir)


examination(base_dir)
