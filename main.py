from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import os
from ModalAnalysis import *

_file_model = ''
_dir_output = ''
ma_ins = ModalAnalysis()


def modal_ana():
    # args pre process
    material_path = './material/material-{}.cfg'.format(val_material.get())
    fixed_vtx = [i for i in range(int(val_fixnum.get()))]

    print('[ INFO] vtk file path', _file_model)
    print('[ INFO] material', material_path)
    print('[ INFO] output path', _dir_output)
    print('[ INFO] fix vtx num', val_fixnum.get())

    global ma_ins
    ma_ins.setVtkFile(_file_model)
    ma_ins.setMaterial(material_path)
    ma_ins.setOutputPath(_dir_output)
    ma_ins.setFixedVtx(fixed_vtx)

    ma_ins.constructM_ori()
    ma_ins.constructK_ori()
    ma_ins.getM_fix()
    ma_ins.getK_fix()
    ma_ins.eignDecom()

    messagebox.showinfo("Message title", "模态分析成功")


def generate_sound():
    print('[ INFO] contact point', val_cpoint.get())
    print('[ INFO] force x: ', val_forcex.get())
    print('[ INFO] force y: ', val_forcey.get())
    print('[ INFO] force z: ', val_forcez.get())

    global ma_ins
    ma_ins.setOutputPath(_dir_output)
    ma_ins.setDuration(3.0)
    ma_ins.setSampRate(44100)
    ma_ins.setForce(int(val_cpoint.get()), float(val_forcex.get()), float(val_forcey.get()), float(val_forcez.get()))
    ma_ins.genSound()

    messagebox.showinfo("Message title", "声音生成成功")

# def play_sound():
#     global ma_ins
#     ma_ins.playSound()

def save_data():
    global ma_ins
    ma_ins.saveAllData()
    ma_ins.saveSound()
    # ma_ins.saveEachMode()

    messagebox.showinfo("Message title", "保存数据成功")
    

root = Tk()
root.title('Modal Analysis demo')
root.geometry('300x300')


# num of fixed vertex
lb_fixnum = Label(root, text='固定点数目：')
lb_fixnum.grid(column=0, row=0)

def_fixnum = IntVar()
def_fixnum.set(0)
val_fixnum = Entry(root, width=5, textvariable=def_fixnum)
val_fixnum.grid(column=1, row=0)

# contact point index
lb_cpoint = Label(root, text='外力作用节点下标：')
lb_cpoint.grid(column=0, row=1)

def_cpoint = IntVar()
def_cpoint.set(3)
val_cpoint = Entry(root, width=5, textvariable=def_cpoint)
val_cpoint.grid(column=1, row=1)


# force x y z
lb_force = Label(root, text='外力大小：')
lb_force.grid(column=0, row=2)

def_forcex = DoubleVar()
def_forcex.set(0.5)
def_forcey = DoubleVar()
def_forcey.set(0.3)
def_forcez = DoubleVar()
def_forcez.set(0.8)

val_forcex = Entry(root, width=5, textvariable=def_forcex)
val_forcex.grid(column=1, row=2)
val_forcey = Entry(root, width=5, textvariable=def_forcey)
val_forcey.grid(column=2, row=2)
val_forcez = Entry(root, width=5, textvariable=def_forcez)
val_forcez.grid(column=3, row=2)

# material num
lb_material = Label(root, text='材料编号：')
lb_material.grid(column=0, row=3)

val_material = Spinbox(root, from_=0, to=7, width=3)
val_material.grid(column=1, row=3)

# model path
def select_model():
    file_model = filedialog.askopenfilename(initialdir=os.path.dirname(__file__))
    if (file_model != ''):
        model_base = '.../' + os.path.basename(file_model)
        lb_model.configure(text=model_base)
    global _file_model
    _file_model = file_model

btn_model = Button(root, text="选择vtk模型", command=select_model)
btn_model.grid(column=0, row=4)

lb_model = Label(root, text='未选择')
lb_model.grid(column=1, row=4)

# output path
def select_output():
    dir_output = filedialog.askdirectory(initialdir=os.path.dirname(__file__))
    if (dir_output != ''):
        dir_base = '.../' + os.path.basename(dir_output)
        lb_output.configure(text=dir_base)
    global _dir_output
    _dir_output = dir_output

btn_output = Button(root, text="选择输出路径", command=select_output)
btn_output.grid(column=0, row=5)

lb_output = Label(root, text='未选择')
lb_output.grid(column=1, row=5)

# ModalAnalysis btn
btn_output = Button(root, text="执行模态分析", command=modal_ana)
btn_output.grid(column=0, row=6)

# Generate sound btn
btn_generate = Button(root, text="生成声音", command=generate_sound)
btn_generate.grid(column=0, row=7)

# # Playsound btn
# btn_play = Button(root, text="播放声音", command=play_sound)
# btn_play.grid(column=1, row=8)

# save btn
btn_save = Button(root, text="保存数据", command=save_data)
btn_save.grid(column=0, row=8)


root.mainloop()

