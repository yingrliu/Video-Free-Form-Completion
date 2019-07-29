"""
Author: Yingru Liu
Write an Qt5 GUI for image editing.
"""
import sys
import cv2
import yaml
import torch
import argparse
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from QT5.mouse_event import GraphicsScene
from copy import deepcopy
# load the model
from ImgModels.SC_FEGAN import SC_FEGAN_Trainer
from Toolkit.PStoolkit import colorGenerator_by_Filter

# TODO: Fix the bug that mask and image are not matched.
class WINDOW(QMainWindow):
    """
    The main window.
    """
    def __init__(self, args_path, checkpoint_name):
        """

        :param args_path: the path of yaml file that saves the checkpoint configuration.
        :param checkpoint_name: the path that save the parameters of model.
        """
        super().__init__()
        """Set the generative model!"""
        # load the configuration.
        with open(args_path, 'r') as stream:
            args_yaml = yaml.load(stream)
            args = argparse.Namespace(**args_yaml)
            print(args)
        trainer = SC_FEGAN_Trainer(args)
        trainer.loadG(PATH=checkpoint_name)
        self.model = deepcopy(trainer.netG)
        self.model.eval()
        del trainer
        """Set the generative model!"""
        # set the general option of the window.
        self.setGeometry(150, 150, 1300, 700)
        self.setWindowTitle('User-Guided Image Editing')
        self.setWindowIcon(QIcon("logos\ImgLogo.jpg"))
        # set the status bar.
        self.statusBar().showMessage('Ready')
        # set the menu.
        exitAct = MENUITEM('&Exit', self, qApp.quit, 'Ctrl+Q', 'Exit App.')
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)
        #
        self.ld_mask = None
        self.ld_sk = None
        self.h, self.w = 512, 512
        # set tool bar.
        toobar = self.addToolBar('Edit')
        toobar.setFixedHeight(85)
        toobar.setIconSize(QSize(75, 75))
        item = MENUITEM('&Open', self, self.openImage, None, 'Open image.', "logos/add.png")
        toobar.addAction(item)
        item = MENUITEM('&Sketch', self, self._sketch_mode, None, 'Draw sketch.', "logos/sketch.png")
        toobar.addAction(item)
        item = MENUITEM('&Color', self, self._stroke_mode, None, 'Draw color.', "logos/color.jpg")
        toobar.addAction(item)
        item = MENUITEM('&Erase', self, self._mask_mode, None, 'Draw mask.', "logos/mask.png")
        toobar.addAction(item)
        item = MENUITEM('&Erase', self, self.synthesize, None, 'Run synthesis.', "logos/run.png")
        toobar.addAction(item)
        # set the area of image editing.
        mainArea = QWidget()
        grid = QGridLayout()
        # graphics view of input image.
        self.input_GraphicsView = QGraphicsView()
        grid.addWidget(self.input_GraphicsView, 0, 0)
        self.output_GraphicsView = QGraphicsView()
        grid.addWidget(self.output_GraphicsView, 0, 1)
        mainArea.setLayout(grid)
        self.mainArea = mainArea
        self.setCentralWidget(mainArea)
        # set the GraphicsScene.
        self.modes = [0, 0, 0]
        self.mouse_clicked = False
        self.input_scene = GraphicsScene(self.modes)
        self.input_GraphicsView.setScene(self.input_scene)
        self.input_GraphicsView.setFixedSize(self.h, self.w)
        self.input_GraphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.input_GraphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.input_GraphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        #
        self.output_scene = GraphicsScene(self.modes)
        self.output_GraphicsView.setScene(self.output_scene)
        self.output_GraphicsView.setFixedSize(self.h, self.w)
        self.output_GraphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.output_GraphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.output_GraphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # color board.
        self.dlg = QColorDialog(self.input_GraphicsView)
        self.colorPlate = PUSHBUTTON('', self, "Color Plate")
        self.colorPlate.clicked.connect(self._color_change_mode)
        self.statusBar().addPermanentWidget(self.colorPlate)
        self.color = None
        self.show()
        return

    def closeEvent(self, event):
        """
        a remind box when quit.
        :param event:
        :return:
        """
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?", QMessageBox.Yes|QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
        return

    def openImage(self):
        """
        open a image when click the corresponding icon.
        :return:
        """
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        if fileName:
            image = QPixmap(fileName)
            mat_img = cv2.imread(fileName)
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return
            self.image = image.scaled(self.h, self.w)
            mat_img = cv2.resize(mat_img, (self.h, self.w), interpolation=cv2.INTER_CUBIC)
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_BGR2RGB)
            mat_img = mat_img / 127.5 - 1.
            mat_img = np.transpose(mat_img, [2, 0, 1])
            self.mat_img = np.expand_dims(mat_img, axis=0)
            self.input_scene.reset()
            if len(self.input_scene.items()) > 0:
                self.input_scene.reset_items()
            self.input_scene.addPixmap(self.image)
            self.input_GraphicsView.setAlignment(Qt.AlignCenter)
            # if len(self.output_scene.items()) > 0:
            #     self.output_scene.removeItem(self.output_scene.items()[-1])
            # self.output_scene.addPixmap(self.image)
            # self.output_GraphicsView.setAlignment(Qt.AlignCenter)
        return

    def _mode_select(self, mode):
        for i in range(len(self.modes)):
            self.modes[i] = 0
        self.modes[mode] = 1

    def _mask_mode(self):
        """
        draw mask when click the corresponding icon.
        :return:
        """
        self._mode_select(0)

    def _sketch_mode(self):
        """
        draw sketch when click the corresponding icon.
        :return:
        """
        self._mode_select(1)

    def _stroke_mode(self):
        """
        add color when click the corresponding icon.
        :return:
        """
        if not self.color:
            self._color_change_mode()
        self.input_scene.get_stk_color(self.color)
        self._mode_select(2)

    def _color_change_mode(self):
        """
        change the color when click the corresponding icon.
        :return:
        """
        self.dlg.exec_()
        self.color = self.dlg.currentColor().name()
        self.colorPlate.setStyleSheet("background-color: %s;" % self.color)
        self.input_scene.get_stk_color(self.color)
        return

    def _generate_mask(self, pts):
        """
        :param pts:
        :return: mask (0 for masked). shape = (1, 1, self.h, self.w)
        """
        if len(pts)>0:
            mask = np.zeros((self.h, self.w, 3))
            for pt in pts:
                cv2.line(mask, pt['prev'], pt['curr'], (255, 255, 255), 12)
            mask = np.asarray(mask[:, :, 0]/255, dtype=np.uint8)
            mask = np.expand_dims(mask, axis=0)
            mask = np.expand_dims(mask, axis=0)
        else:
            mask = np.zeros((self.h, self.w, 3))
            mask = np.asarray(mask[:, :, 0]/255, dtype=np.uint8)
            mask = np.expand_dims(mask, axis=0)
            mask = np.expand_dims(mask, axis=0)
        return 1 - mask


    def _generate_color(self, pts):
        if len(pts) > 0:
            stroke = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            for pt in pts:
                c = pt['color'].lstrip('#')
                color = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
                color = (color[2], color[1], color[0])
                cv2.line(stroke, pt['prev'], pt['curr'], color, 6)
            stroke = cv2.cvtColor(stroke, cv2.COLOR_BGR2RGB)
            stroke = stroke/255
            stroke = np.expand_dims(stroke, axis=0)
        else:
            stroke = np.zeros((1, self.h, self.w, 3))
        stroke = np.transpose(stroke, [0, 3, 1, 2])
        return stroke

    def _generate_sketch(self, pts):
        """
        :param pts:
        :return: sketch (1 for sketch). shape = (1, 1, self.h, self.w)
        """
        if len(pts) > 0:
            sketch = np.zeros((self.h, self.w, 3))
            # sketch = 255*sketch
            for pt in pts:
                cv2.line(sketch, pt['prev'], pt['curr'], (255, 255, 255), 3)
            sketch = np.asarray(sketch[:, :, 0] / 255, dtype=np.uint8)
            sketch = np.expand_dims(sketch, axis=0)
            sketch = np.expand_dims(sketch, axis=0)
        else:
            sketch = np.zeros((self.h, self.w, 3))
            # sketch = 255*sketch
            sketch = np.asarray(sketch[:, :, 0] / 255, dtype=np.uint8)
            sketch = np.expand_dims(sketch, axis=0)
            sketch = np.expand_dims(sketch, axis=0)
        return sketch

    def synthesize(self):
        """

        :return:
        """
        """Exist if not image is loaded."""
        if not hasattr(self, 'mat_img'):
            reply = QMessageBox.question(self, 'Message', "You should load an image first.", QMessageBox.Ok)
            if reply == QMessageBox.Ok:
                return
        """Prepare the input data."""
        sketch = self._generate_sketch(self.input_scene.sketch_points)
        stroke = self._generate_color(self.input_scene.stroke_points)
        mask = self._generate_mask(self.input_scene.mask_points)
        noise = np.random.normal(0., 1., size=mask.shape)
        # merge with previous mask. (0 for masked).
        if not type(self.ld_mask) == type(None):
            ld_mask = np.expand_dims(self.ld_mask[:, :, 0:1], axis=0)           # shape= [1, self.h, self.w, 1].
            mask = mask+ld_mask
            mask[mask < 2] = 0
            mask[mask == 2] = 1
            mask = np.asarray(mask,dtype=np.uint8)
        self.ld_mask = mask[0]

        if not type(self.ld_sk)==type(None):
            sketch = sketch+self.ld_sk
            sketch[sketch>0] = 1

        #
        sketch = sketch*(1 - mask)
        stroke = stroke*(1 - mask)
        noise = noise*(1 - mask)
        Img = self.mat_img * mask
        data = np.concatenate([Img, sketch, stroke, mask, noise], axis=-3)
        data = torch.from_numpy(data.astype(dtype=np.float32))
        if torch.cuda.is_available():
            data = data.cuda()
        """Compute the synthesized output."""
        syn_img = self.model.forward(data)
        syn_img= syn_img.detach().cpu().numpy()
        syn_img = Img * mask + syn_img * (1 - mask)
        syn_img = np.transpose(syn_img[0], [1, 2, 0]).copy()
        syn_img = np.asarray((syn_img + 1.0) * 127.5, dtype=np.uint8)
        """Visualize the synthesized output."""
        if len(self.output_scene.items()) > 0:
            self.output_scene.reset_items()
        qim = QImage(syn_img, self.h, self.w, self.w * 3, QImage.Format_RGB888)
        image = QPixmap.fromImage(qim)
        self.output_scene.addPixmap(image)
        self.output_GraphicsView.setAlignment(Qt.AlignCenter)
        #
        return


    # todo:
    def reset(self):
        return

class PUSHBUTTON(QPushButton):
    def __init__(self, name, widget, tips):
        super(PUSHBUTTON, self).__init__(name, widget)
        if tips:
            self.setToolTip(tips)
        self.resize(self.sizeHint())
        self.move(50, 50)
        self.show()

class MENUITEM(QAction):
    def __init__(self, name, widget, action, shortcut=None, tips=None, icon=None):
        super(MENUITEM, self).__init__(name, widget)
        self.triggered.connect(action)
        if shortcut:
            self.setShortcut(shortcut)
        if tips:
            self.setStatusTip(tips)
        if icon:
            self.setIcon(QIcon(icon))
        return

args_path = "D:\Projects\VideoEditing\parameters.yaml"
checkpoint_path = "D:\Projects\VideoEditing\CheckPoint.pth-0013.tar"
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = WINDOW(args_path, checkpoint_path)
    sys.exit(app.exec_())
