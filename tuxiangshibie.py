# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tuxiangshibie.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
import tkinter as tk
from tkinter import filedialog
import os
from PyQt5.QtWidgets import QFileDialog
import torch
import cv2
import torch.nn.functional as F
from shizhan import Net
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(494, 429)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(130, 180, 131, 71))
        self.pushButton.setAutoRepeatDelay(600)
        self.pushButton.setObjectName("pushButton")
        self.pushButton1 = QtWidgets.QPushButton(Form)
        self.pushButton1.setGeometry(QtCore.QRect(130, 180, 131, 71))
        self.pushButton1.setAutoRepeatDelay(600)
        self.pushButton1.setObjectName("pushButton1")
        self.pushButton1.move(130,10)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(170, 110, 101, 51))
        self.label.setObjectName("label")
        self.retranslateUi(Form)
        self.pushButton.clicked.connect(self.shibie)
        self.pushButton1.clicked.connect(self.daoru1)
        QtCore.QMetaObject.connectSlotsByName(Form)
    def daoru1(self):
        application_window = tk.Tk()
        # 设置文件对话框会显示的文件类型
        my_filetypes = [('all files', '.*'), ('text files', '.txt')]
        # 请求选择文件
        answer = filedialog.askopenfilename(parent=application_window,initialdir=os.getcwd(),title="Please select a file:",filetypes=my_filetypes)
        global k
        k=answer
    def shibie(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load('model.pth')  # 加载模型
        model = model.to(device)
        model.eval()  # 把模型转为test模式
        global k
        
        img = cv2.imread(k)  # 读取要预测的图片
        img=cv2.resize(img,(32,32))
        trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        # 扩展后，为[1，1，28，28]
        output = model(img)
        prob = F.softmax(output,dim=1) #prob是10个分类的概率
        value, predicted = torch.max(output.data, 1)
        pred_class = classes[predicted.item()]
        self.label.setText(pred_class)
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton1.setText(_translate("Form", "导入图片"))
        self.pushButton.setText(_translate("Form", "开始识别"))
        self.label.setText(_translate("Form", "TextLabel"))
if __name__ == "__main__":
 app = QtWidgets.QApplication(sys.argv)
 widget = QtWidgets.QWidget()
 ui = Ui_Form()
 ui.setupUi(widget)
 widget.show()
 sys.exit(app.exec_())
