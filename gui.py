# -*- coding: utf-8 -*-

from transform import Transform, Histogram, Math
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import ntpath
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy


def desktopGeometry():
    return QtWidgets.QDesktopWidget().availableGeometry()


defonfigure...
 adjustWindowSize(window, widgetWidth, widgetHeight):
    centralWidget = window.centralWidget()
    maxAvailableSize = desktopGeometry().size()
    maxSize = maxAvailableSize - window.frameSize() + centralWidget.size()
    size = QtCore.QSize(min(maxSize.width(), widgetWidth), min(maxSize.height(), widgetHeight))

    scrollbarWidth = QtWidgets.qApp.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
    if maxSize.height() < widgetHeight and maxSize.width() > widgetWidth:
        size.setWidth(size.width() + scrollbarWidth)
    if maxSize.width() < widgetWidth and maxSize.height() > widgetHeight:
        size.setHeight(size.height() + scrollbarWidth)

    centralWidget.setMinimumSize(size)
    window.adjustSize()
    centralWidget.setMinimumSize(200, 100)


class DialogLog(QtWidgets.QDialog):
    c = None

    def __init__(self, parentWindow):
        QtWidgets.QDialog.__init__(self, parentWindow)
        layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle("Log transform")

        layout.addWidget(QtWidgets.QLabel("Function: c * Log(i+1)"))

        self.center = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QHBoxLayout(self.center)
        self.center.setLayout(self.layout1)
        self.layout1.addWidget(QtWidgets.QLabel("c ="))
        self.textfieldC = QtWidgets.QLineEdit(self.center)
        self.textfieldC.setValidator(
            QtGui.QDoubleValidator(self.textfieldC))
        self.layout1.addWidget(self.textfieldC)
        layout.addWidget(self.center)

        self.buttons = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QHBoxLayout(self.buttons)
        self.buttons.setLayout(self.layout2)
        self.buttonOk = QtWidgets.QPushButton("OK")
        self.buttonOk.clicked.connect(self.ok)
        self.buttonCancel = QtWidgets.QPushButton("Cancel")
        self.buttonCancel.clicked.connect(self.close)
        self.layout2.addWidget(self.buttonOk)
        self.layout2.addWidget(self.buttonCancel)
        layout.addWidget(self.buttons)

        self.resize(300, self.size().height())

    def ok(self):
        self.c = float(self.textfieldC.text())
        self.close()

    def keyPressEvent(self, e):
        k = e.key()
        if k == QtCore.Qt.Key_Escape:
            self.close()
        elif k == QtCore.Qt.Key_Enter or k == QtCore.Qt.Key_Return:
            self.ok()


class DialogPower(QtWidgets.QDialog):
    c = None
    p = None

    def __init__(self, parentWindow):
        QtWidgets.QDialog.__init__(self, parentWindow)
        layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle("Power transform")

        layout.addWidget(QtWidgets.QLabel("Function: c * i ^ p"))

        self.center = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QHBoxLayout(self.center)
        self.center.setLayout(self.layout1)
        self.layout1.addWidget(QtWidgets.QLabel("c ="))
        self.textfieldC = QtWidgets.QLineEdit(self.center)
        self.textfieldC.setValidator(
            QtGui.QDoubleValidator(self.textfieldC))
        self.layout1.addWidget(self.textfieldC)
        layout.addWidget(self.center)

        self.center1 = QtWidgets.QWidget(self)
        self.layout3 = QtWidgets.QHBoxLayout(self.center1)
        self.center1.setLayout(self.layout3)
        self.layout3.addWidget(QtWidgets.QLabel("p ="))
        self.textfieldP = QtWidgets.QLineEdit(self.center1)
        self.textfieldP.setValidator(
            QtGui.QDoubleValidator(self.textfieldP))
        self.layout3.addWidget(self.textfieldP)
        layout.addWidget(self.center1)

        self.buttons = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QHBoxLayout(self.buttons)
        self.buttons.setLayout(self.layout2)
        self.buttonOk = QtWidgets.QPushButton("OK")
        self.buttonOk.clicked.connect(self.ok)
        self.buttonCancel = QtWidgets.QPushButton("Cancel")
        self.buttonCancel.clicked.connect(self.close)
        self.layout2.addWidget(self.buttonOk)
        self.layout2.addWidget(self.buttonCancel)
        layout.addWidget(self.buttons)

        self.resize(300, self.size().height())

    def ok(self):
        self.c = float(self.textfieldC.text())
        self.p = float(self.textfieldP.text())
        self.close()

    def keyPressEvent(self, e):
        k = e.key()
        if k == QtCore.Qt.Key_Escape:
            self.close()
        elif k == QtCore.Qt.Key_Enter or k == QtCore.Qt.Key_Return:
            self.ok()


class DialogShape(QtWidgets.QDialog):
    function = None
    center = None
    width = None
    times = None
    distance = None
    direction = None

    def __init__(self, parentWindow):
        QtWidgets.QDialog.__init__(self, parentWindow)
        layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle("Histogram shaping")

        layout.addWidget(QtWidgets.QLabel("Target histogram"))

        self.center = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QGridLayout(self.center)
        self.center.setLayout(self.layout1)
        self.layout1.addWidget(QtWidgets.QLabel("Function: "), 0, 0)
        self.boxFunction = QtWidgets.QComboBox(self.center)
        self.boxFunction.addItem("Uniform")
        self.boxFunction.addItem("Triangle")
        self.boxFunction.addItem("Normal (Gaussian)")
        self.layout1.addWidget(self.boxFunction, 0, 1)
        #
        self.layout1.addWidget(QtWidgets.QLabel("Center (Mean): "), 1, 0)
        self.textfieldCenter = QtWidgets.QLineEdit(self.center)
        self.textfieldCenter.setText("127")
        self.textfieldCenter.setValidator(
            QtGui.QIntValidator(0, 255, self.textfieldCenter))
        self.layout1.addWidget(self.textfieldCenter, 1, 1)
        #
        self.layout1.addWidget(QtWidgets.QLabel("Width (Variance): "), 2, 0)
        self.textfieldWidth = QtWidgets.QLineEdit(self.center)
        self.textfieldWidth.setText("256")
        self.textfieldWidth.setValidator(
            QtGui.QIntValidator(1, 256, self.textfieldWidth))
        self.layout1.addWidget(self.textfieldWidth, 2, 1)
        #
        self.layout1.addWidget(QtWidgets.QLabel("Repeat times: "), 3, 0)
        self.textfieldTimes = QtWidgets.QLineEdit(self.center)
        self.textfieldTimes.setText("0")
        self.textfieldTimes.setValidator(
            QtGui.QIntValidator(0, 127, self.textfieldTimes))
        self.layout1.addWidget(self.textfieldTimes, 3, 1)
        #
        self.layout1.addWidget(QtWidgets.QLabel("Repeat distance: "), 4, 0)
        self.textfieldDistance = QtWidgets.QLineEdit(self.center)
        self.textfieldDistance.setText("0")
        self.textfieldDistance.setValidator(
            QtGui.QIntValidator(0, 255, self.textfieldDistance))
        self.layout1.addWidget(self.textfieldDistance, 4, 1)
        #
        self.layout1.addWidget(QtWidgets.QLabel("Repeat direction: "), 5, 0)
        self.boxDirection = QtWidgets.QComboBox(self.center)
        self.boxDirection.addItem("Both")
        self.boxDirection.addItem("Right")
        self.boxDirection.addItem("Left")
        self.layout1.addWidget(self.boxDirection, 5, 1)
        layout.addWidget(self.center)

        self.buttons = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QHBoxLayout(self.buttons)
        self.buttons.setLayout(self.layout2)
        self.buttonOk = QtWidgets.QPushButton("OK")
        self.buttonOk.clicked.connect(self.ok)
        self.buttonCancel = QtWidgets.QPushButton("Cancel")
        self.buttonCancel.clicked.connect(self.close)
        self.layout2.addWidget(self.buttonOk)
        self.layout2.addWidget(self.buttonCancel)
        layout.addWidget(self.buttons)

        self.resize(300, self.size().height())

    def ok(self):
        functions = ["uniform", "triangle", "gaussian"]
        directions = ["both", "right", "left"]
        self.function = functions[self.boxFunction.currentIndex()]
        self.center = int(self.textfieldCenter.text())
        self.width = int(self.textfieldWidth.text())
        self.times = int(self.textfieldTimes.text())
        self.distance = int(self.textfieldDistance.text())
        self.direction = directions[self.boxDirection.currentIndex()]
        self.close()

    def keyPressEvent(self, e):
        k = e.key()
        if k == QtCore.Qt.Key_Escape:
            self.close()


class DialogMatch(QtWidgets.QDialog):
    f = ''
    path = ''

    def __init__(self, parentWindow):
        QtWidgets.QDialog.__init__(self, parentWindow)
        layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle("Histogram matching")

        layout.addWidget(QtWidgets.QLabel("Target image"))

        self.center = QtWidgets.QWidget(self)
        self.layout1 = QtWidgets.QHBoxLayout(self.center)
        self.center.setLayout(self.layout1)
        self.buttonBrowse = QtWidgets.QPushButton("Choose File")
        self.buttonBrowse.clicked.connect(self.browse)
        self.layout1.addWidget(self.buttonBrowse)
        self.label = QtWidgets.QLabel("No File Chosen")
        self.label.setStyleSheet("QLabel {color: gray;}")
        self.layout1.addWidget(self.label)
        layout.addWidget(self.center)

        self.buttons = QtWidgets.QWidget(self)
        self.layout2 = QtWidgets.QHBoxLayout(self.buttons)
        self.buttons.setLayout(self.layout2)
        self.buttonOk = QtWidgets.QPushButton("OK")
        self.buttonOk.clicked.connect(self.ok)
        self.buttonCancel = QtWidgets.QPushButton("Cancel")
        self.buttonCancel.clicked.connect(self.close)
        self.layout2.addWidget(self.buttonOk)
        self.layout2.addWidget(self.buttonCancel)
        layout.addWidget(self.buttons)

        self.resize(300, self.size().height())

    def browse(self):
        self.f = QtWidgets.QFileDialog.getOpenFileName(self, 'Select File')[0]
        if self.f:
            self.label.setText(ntpath.basename(str(self.f)))
            self.label.setStyleSheet("QLabel {color: black;}")

    def ok(self):
        self.path = str(self.f)
        self.close()

    def keyPressEvent(self, e):
        k = e.key()
        if k == QtCore.Qt.Key_Escape:
            self.close()


class WindowHistogram(QtWidgets.QMainWindow):
    closed = False
    image = None

    def __init__(self, parent):
        super(WindowHistogram, self).__init__(parent)
        self.parentWindow = parent
        self.setWindowTitle("Intensity Transformer - Histogram")
        parent.actionHistogram.setChecked(True)

        # add image holder
        self.scrollArea = QtWidgets.QScrollArea()
        self.imageLabel = ImageLabel(scrollArea=self.scrollArea)
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea.setAlignment(QtCore.Qt.AlignCenter)
        self.scrollArea.setWidget(self.imageLabel)

        self.setCentralWidget(self.scrollArea)

    def closeEvent(self, e):
        self.closed = True
        self.parentWindow.actionHistogram.setChecked(False)

    def isClosed(self):
        return self.closed

    def setHistogram(self, hist, legend=["Image"]):
        fig = plt.figure()
        ax = plt.subplot(111)
        x = range(256)
        for y in hist:
            plt.plot(x, y, drawstyle='steps', linewidth=0.7)
            plt.bar(x, y, width=1, alpha=0.5)
        ax.set_ylim(ymin=0)
        ax.autoscale(enable=True, axis='x', tight=True)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        leg = plt.legend(legend,
                         loc='upper center',
                         bbox_to_anchor=(0.5, -0.1),
                         ncol=len(legend) % 5)
        plt.setp(leg.get_lines(), linewidth=4)
        fig.canvas.draw()
        size = fig.get_size_inches() * fig.get_dpi()
        self.width = int(size[0])
        self.height = int(size[1])
        matrix = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        matrix = matrix.reshape(self.height, self.width, 3)
        image = QtGui.QImage(matrix, self.width, self.height, QtGui.QImage.Format_RGB888)
        self.imageLabel.setPixmap(QtGui.QPixmap(image))
        self.imageLabel.adjustSize()
        plt.close(fig)

    def adjustWindowSize(self):
        adjustWindowSize(self, self.width + 2, self.height + 2)

    def adjustWindowPosition(self):
        # move the window to top-right of the main window
        self.move(self.parent().frameGeometry().topRight())
        if not desktopGeometry().contains(self.frameGeometry()):
            point = desktopGeometry().topRight()
            point.setX(point.x() - self.frameSize().width() + 1)
            self.move(point)


class DropFieldWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DropFieldWidget, self).__init__(parent.window)
        self.parent = parent
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        mimeData = e.mimeData()
        if mimeData.hasUrls:
            urls = mimeData.urls()
            if len(urls) == 1:
                e.accept()
                return
        e.ignore()

    def dropEvent(self, e):
        mimeData = e.mimeData()
        if mimeData.hasUrls:
            urls = mimeData.urls()
            if len(urls) == 1:
                path = str(urls[0].toLocalFile())
                e.accept()
                self.parent.openImage(path)
                return
        e.ignore()


class ImageLabel(QtWidgets.QLabel):
    canMoveImage = False
    mousePos = None

    def __init__(self, parent=None, scrollArea=None):
        super(ImageLabel, self).__init__(parent)
        self.scrollArea = scrollArea
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def mousePressEvent(self, e):
        if e.button() != QtCore.Qt.LeftButton:
            return
        self.mousePos = e.globalPos()
        self.canMoveImage = True
        self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if not self.canMoveImage:
            return
        mousePos = e.globalPos()
        d = self.mousePos - mousePos
        hBar = self.scrollArea.horizontalScrollBar()
        vBar = self.scrollArea.verticalScrollBar()
        x = hBar.value() + d.x()
        y = vBar.value() + d.y()
        hBar.setValue(x)
        vBar.setValue(y)
        self.mousePos = mousePos

    def mouseReleaseEvent(self, e):
        if e.button() != QtCore.Qt.LeftButton:
            return
        self.mousePos = None
        self.canMoveImage = False
        self.setCursor(QtCore.Qt.OpenHandCursor)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.history = None
        self.historyIndex = 0
        self.windowHist = None

        self.window = MainWindow
        MainWindow.resize(800, 600)
        MainWindow.closeEvent = self.closeEvent
        self.centralwidget = DropFieldWidget(self)
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowTitle("Intensity Transformer - Image")

        # build menu bar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuTransform = QtWidgets.QMenu(self.menubar)
        self.menuWindow = QtWidgets.QMenu(self.menubar)
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        MainWindow.setMenuBar(self.menubar)

        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionExit = QtWidgets.QAction(MainWindow)

        self.actionUndo = QtWidgets.QAction(MainWindow)
        self.actionRedo = QtWidgets.QAction(MainWindow)

        self.actionZoomIn = QtWidgets.QAction(MainWindow)
        self.actionZoomOut = QtWidgets.QAction(MainWindow)
        self.actionResetZoom = QtWidgets.QAction(MainWindow)

        self.actionNegative = QtWidgets.QAction(MainWindow)
        self.actionLog = QtWidgets.QAction(MainWindow)
        self.actionPower = QtWidgets.QAction(MainWindow)
        self.actionEqualize = QtWidgets.QAction(MainWindow)
        self.actionShape = QtWidgets.QAction(MainWindow)
        self.actionMatch = QtWidgets.QAction(MainWindow)

        self.window.actionHistogram = QtWidgets.QAction(MainWindow)

        self.actionAbout = QtWidgets.QAction(MainWindow)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionExit)

        self.menubar.addAction(self.menuEdit.menuAction())
        self.menuEdit.addAction(self.actionUndo)
        self.menuEdit.addAction(self.actionRedo)

        self.menubar.addAction(self.menuView.menuAction())
        self.menuView.addAction(self.actionZoomIn)
        self.menuView.addAction(self.actionZoomOut)
        self.menuView.addAction(self.actionResetZoom)

        self.menubar.addAction(self.menuTransform.menuAction())
        self.menuTransform.addAction(self.actionNegative)
        self.menuTransform.addAction(self.actionLog)
        self.menuTransform.addAction(self.actionPower)
        self.menuTransform.addAction(self.actionEqualize)
        self.menuTransform.addAction(self.actionShape)
        self.menuTransform.addAction(self.actionMatch)

        self.menubar.addAction(self.menuWindow.menuAction())
        self.menuWindow.addAction(self.window.actionHistogram)

        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuHelp.addAction(self.actionAbout)

        # set menu bar items' text
        self.menuFile.setTitle("&File")
        self.actionOpen.setText("&Open")
        self.actionSave.setText("&Save")
        self.actionExit.setText("&Exit")

        self.menuEdit.setTitle("&Edit")
        self.actionUndo.setText("&Undo")
        self.actionRedo.setText("&Redo")

        self.menuView.setTitle("&View")
        self.actionZoomIn.setText("Zoom &In")
        self.actionZoomOut.setText("Soom &Out")
        self.actionResetZoom.setText("&Reset Zoom")

        self.menuTransform.setTitle("&Transform")
        self.actionNegative.setText("&Negative")
        self.actionLog.setText("&Log")
        self.actionPower.setText("&Power")
        self.actionEqualize.setText("&Equalize")
        self.actionShape.setText("&Shape")
        self.actionMatch.setText("&Match")

        self.menuWindow.setTitle("&Window")
        self.window.actionHistogram.setText("&Histogram")

        self.menuHelp.setTitle("&Help")
        self.actionAbout.setText("&About")

        # set menu bar items' keyboard shortcuts
        self.actionOpen.setShortcut("Ctrl+O")
        self.actionSave.setShortcut("Ctrl+S")
        self.actionExit.setShortcut("Ctrl+W")

        self.actionUndo.setShortcut(QtGui.QKeySequence.Undo)
        self.actionRedo.setShortcut(QtGui.QKeySequence.Redo)

        self.actionZoomIn.setShortcut(QtGui.QKeySequence.ZoomIn)
        self.actionZoomOut.setShortcut(QtGui.QKeySequence.ZoomOut)
        self.actionResetZoom.setShortcut("Ctrl+/")

        self.actionNegative.setShortcut("Ctrl+N")
        self.actionLog.setShortcut("Ctrl+L")
        self.actionPower.setShortcut("Ctrl+P")
        self.actionEqualize.setShortcut("Ctrl+E")
        self.actionShape.setShortcut("Ctrl+A")
        self.actionMatch.setShortcut("Ctrl+M")

        self.window.actionHistogram.setShortcut("Ctrl+H")

        self.actionAbout.setShortcut("F1")

        #
        self.window.actionHistogram.setCheckable(True)

        # disable _Save, Edit+, View+, Transform+ and _Histogram
        self.actionSave.setEnabled(False)
        for a in self.menuEdit.actions():
            a.setEnabled(False)
        self.menuEdit.setEnabled(False)
        for a in self.menuView.actions():
            a.setEnabled(False)
        self.menuView.setEnabled(False)
        for a in self.menuTransform.actions():
            a.setEnabled(False)
        self.menuTransform.setEnabled(False)
        self.window.actionHistogram.setEnabled(False)

        # setup drop field
        layout = QtWidgets.QVBoxLayout(self.centralwidget)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        label1 = QtWidgets.QLabel("Drag and drop file here")
        label1.setStyleSheet("QLabel {font-size: 20px}")
        layout.addWidget(label1, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacing(30)
        label2 = QtWidgets.QLabel("( or )")
        label2.setStyleSheet("QLabel {color: gray; font-size: 12px}")
        layout.addWidget(label2, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacing(30)
        buttonChooseFile = QtWidgets.QPushButton("Choose File")
        buttonChooseFile.setFixedSize(140, 35)
        layout.addWidget(buttonChooseFile, alignment=QtCore.Qt.AlignCenter)

        # set menu bar action listeners
        self.actionOpen.triggered.connect(self.menuActionOpen)
        self.actionSave.triggered.connect(self.menuActionSave)
        self.actionExit.triggered.connect(self.menuActionExit)

        self.actionUndo.triggered.connect(self.menuActionUndo)
        self.actionRedo.triggered.connect(self.menuActionRedo)

        self.actionZoomIn.triggered.connect(self.menuActionZoomIn)
        self.actionZoomOut.triggered.connect(self.menuActionZoomOut)
        self.actionResetZoom.triggered.connect(self.menuActionResetZoom)

        self.actionNegative.triggered.connect(self.menuActionNegative)
        self.actionLog.triggered.connect(self.menuActionLog)
        self.actionPower.triggered.connect(self.menuActionPower)
        self.actionEqualize.triggered.connect(self.menuActionEqualize)
        self.actionShape.triggered.connect(self.menuActionShape)
        self.actionMatch.triggered.connect(self.menuActionMatch)

        self.window.actionHistogram.triggered.connect(self.menuActionHistogram)

        self.actionAbout.triggered.connect(self.menuActionAbout)

        buttonChooseFile.clicked.connect(self.menuActionOpen)

    def openImage(self, path):
        if not path:
            return

        # close current image
        e = QtCore.QEvent(QtCore.QEvent.None_)
        self.closeEvent(e)
        if not e.isAccepted():
            return

        # read image
        matrix = cv2.imread(path, 0)
        if matrix is None:
            print("Could not open file or file is not an image.")
            return

        self.history = [{'matrix': matrix, 'histogram': None, 'saved': True}]
        self.historyIndex = 0

        layout = self.centralwidget.layout()
        # clean layout
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is None:
                layout.removeItem(item)
            else:
                widget.deleteLater()
            del item

        # show image
        self.scrollArea = QtWidgets.QScrollArea()
        self.imageLabel = ImageLabel(scrollArea=self.scrollArea)
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea.setAlignment(QtCore.Qt.AlignCenter)
        self.scrollArea.setWidget(self.imageLabel)
        layout.addWidget(self.scrollArea)
        layout.setContentsMargins(0, 0, 0, 0)

        self.zoom = 1
        self.updateImage(matrix)

        # adjust window's size
        height, width = matrix.shape
        adjustWindowSize(self.window, width + 2, height + 2)

        # center window
        geom = desktopGeometry()
        x = geom.x() + 0.5 * (geom.width() - self.window.frameSize().width())
        y = geom.y() + 0.5 * (geom.height() - self.window.frameSize().height())
        self.window.move(x, y)

        # create histogram window
        if self.windowHist is None or self.windowHist.isClosed():
            if self.windowHist is not None:
                del self.windowHist
            self.windowHist = WindowHistogram(self.window)
            self.windowHist.show()

        hist = [Histogram.hist(matrix)]
        legend = ["Source Image"]
        self.history[0]['histogram'] = {'plots': hist, 'legend': legend}

        self.windowHist.setHistogram(hist, legend)
        self.windowHist.adjustWindowSize()
        self.windowHist.adjustWindowPosition()

        self.window.activateWindow()
        self.window.setFocus()

        # enable _Save, Edit, View+, Transform+ and _Histogram
        self.actionSave.setEnabled(True)
        self.menuEdit.setEnabled(True)
        for a in self.menuView.actions():
            a.setEnabled(True)
        self.menuView.setEnabled(True)
        for a in self.menuTransform.actions():
            a.setEnabled(True)
        self.menuTransform.setEnabled(True)
        self.window.actionHistogram.setEnabled(True)

    def updateImage(self, matrix):
        height, width = matrix.shape
        image = QtGui.QImage(matrix, width, height, width, QtGui.QImage.Format_Indexed8)
        self.imageLabel.setPixmap(QtGui.QPixmap(image))
        self.imageLabel.adjustSize()
        self.imageLabel.resize(self.zoom * self.imageLabel.pixmap().size())

    def saveImage(self, path):
        if not path:
            return False
        try:
            matrix = self.history[self.historyIndex]['matrix']
            cv2.imwrite(path, matrix)
            return True
        except:
            print("Error while saving the image.")
            return False

    def appendHistory(self, item):
        self.historyIndex += 1
        self.history[self.historyIndex:] = [item]
        self.actionUndo.setEnabled(True)
        self.actionRedo.setEnabled(False)

    def closeEvent(self, e):
        if self.history is None \
                or len(self.history) == 0 \
                or self.history[self.historyIndex]['saved']:
            e.accept()
            return
        while True:
            msgBox = QtWidgets.QMessageBox(self.window)
            msgBox.setWindowTitle("Intensity Transformer")
            msgBox.setText("The image has been modified.")
            msgBox.setInformativeText("Do you want to save your changes?")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
            msgBox.setDefaultButton(QtWidgets.QMessageBox.Save)
            ret = msgBox.exec_()
            if ret == QtWidgets.QMessageBox.Save:
                ret = self.menuActionSave()
                if ret:
                    e.accept()
                    break
                continue
            elif ret == QtWidgets.QMessageBox.Discard:
                e.accept()
            else:
                e.ignore()
            break

    def menuActionOpen(self):
        path = str(QtWidgets.QFileDialog.getOpenFileName(self.window, 'Open File')[0])
        self.openImage(path)

    def menuActionSave(self):
        path = str(QtWidgets.QFileDialog.getSaveFileName(self.window, 'Save File')[0])
        ret = self.saveImage(path)
        if not ret:
            return False
        self.history[self.historyIndex]['saved'] = True
        return True

    def menuActionExit(self):
        self.window.close()

    def menuActionUndo(self):
        if self.historyIndex < 1:
            return
        self.historyIndex -= 1
        # update matrix and histogram
        item = self.history[self.historyIndex]
        self.updateImage(item['matrix'])
        if self.historyIndex == 0:
            hist = item['histogram']['plots']
            legend = item['histogram']['legend']
        else:
            prevHist = self.history[self.historyIndex - 1]['histogram']['plots'][0]
            hist = [prevHist] + item['histogram']['plots']
            legend = ["Old"] + item['histogram']['legend']
        self.windowHist.setHistogram(hist, legend)
        # enable/disable actions
        if self.historyIndex < 1:
            self.actionUndo.setEnabled(False)
        self.actionRedo.setEnabled(True)

    def menuActionRedo(self):
        if self.historyIndex + 1 >= len(self.history):
            return
        self.historyIndex += 1
        # update matrix and histogram
        item = self.history[self.historyIndex]
        self.updateImage(item['matrix'])
        prevHist = self.history[self.historyIndex - 1]['histogram']['plots'][0]
        hist = [prevHist] + item['histogram']['plots']
        legend = ["Old"] + item['histogram']['legend']
        self.windowHist.setHistogram(hist, legend)
        # enable/disable actions
        if self.historyIndex + 1 >= len(self.history):
            self.actionRedo.setEnabled(False)
        self.actionUndo.setEnabled(True)

    def menuActionZoomIn(self):
        self.zoom += 0.25
        self.imageLabel.resize(self.zoom * self.imageLabel.pixmap().size())
        self.actionZoomOut.setEnabled(True)

    def menuActionZoomOut(self):
        self.zoom -= 0.25
        self.imageLabel.resize(self.zoom * self.imageLabel.pixmap().size())
        self.actionZoomOut.setEnabled(self.zoom > 0.25)

    def menuActionResetZoom(self):
        self.zoom = 1
        self.imageLabel.resize(self.imageLabel.pixmap().size())
        self.actionZoomOut.setEnabled(True)

    def menuActionNegative(self):
        # copy current state
        old = self.history[self.historyIndex]
        matrix = old['matrix'].copy()
        histogram = deepcopy(old['histogram']['plots'][0])
        # apply negative function
        matrix = Transform.negative(matrix)
        # compute histogram
        histogram = {
            'plots': [Histogram.hist(matrix)],
            'legend': ["Negative"]
        }
        # add new state
        self.appendHistory({
            'matrix': matrix,
            'histogram': histogram,
            'saved': False
        })
        self.updateImage(matrix)
        hist = [old['histogram']['plots'][0]] + histogram['plots']
        legend = ["Old"] + histogram['legend']
        self.windowHist.setHistogram(hist, legend)

    def menuActionLog(self):
        # get parameters
        popup = DialogLog(self.window)
        popup.exec_()
        if popup.c is None:
            return
        # copy current state
        old = self.history[self.historyIndex]
        matrix = old['matrix'].copy()
        histogram = deepcopy(old['histogram']['plots'][0])
        # apply log function
        matrix = Transform.log(matrix, popup.c)
        # compute histogram
        histogram = {
            'plots': [Histogram.hist(matrix)],
            'legend': [str(popup.c) + "*log(i+1)"]
        }
        # add new state
        self.appendHistory({
            'matrix': matrix,
            'histogram': histogram,
            'saved': False
        })
        self.updateImage(matrix)
        hist = [old['histogram']['plots'][0]] + histogram['plots']
        legend = ["Old"] + histogram['legend']
        self.windowHist.setHistogram(hist, legend)

    def menuActionPower(self):
        # get parameters
        popup = DialogPower(self.window)
        popup.exec_()
        if popup.c is None or popup.p is None:
            return
        # copy current state
        old = self.history[self.historyIndex]
        matrix = old['matrix'].copy()
        histogram = deepcopy(old['histogram']['plots'][0])
        # apply power function
        matrix = Transform.power(matrix, popup.c, popup.p)
        # compute histogram
        histogram = {
            'plots': [Histogram.hist(matrix)],
            'legend': [str(popup.c) + "*i^" + str(popup.p)]
        }
        # add new state
        self.appendHistory({
            'matrix': matrix,
            'histogram': histogram,
            'saved': False
        })
        self.updateImage(matrix)
        hist = [old['histogram']['plots'][0]] + histogram['plots']
        legend = ["Old"] + histogram['legend']
        self.windowHist.setHistogram(hist, legend)

    def menuActionEqualize(self):
        # copy current state
        old = self.history[self.historyIndex]
        matrix = old['matrix'].copy()
        histogram = deepcopy(old['histogram']['plots'][0])
        # apply histogram equalization
        matrix = Transform.equalize(matrix, histogram)
        # compute histogram
        histogram = {
            'plots': [Histogram.hist(matrix)],
            'legend': ["Equalized"]
        }
        # add new state
        self.appendHistory({
            'matrix': matrix,
            'histogram': histogram,
            'saved': False
        })
        self.updateImage(matrix)
        hist = [old['histogram']['plots'][0]] + histogram['plots']
        legend = ["Old"] + histogram['legend']
        self.windowHist.setHistogram(hist, legend)

    def menuActionShape(self):
        # get parameters
        popup = DialogShape(self.window)
        popup.exec_()
        if popup.function is None \
                or popup.center is None or popup.width is None \
                or popup.times is None or popup.distance is None \
                or popup.direction is None:
            return
        # copy current state
        old = self.history[self.historyIndex]
        matrix = old['matrix'].copy()
        histogram = deepcopy(old['histogram']['plots'][0])
        # apply histogram shaping
        matrix, target = Transform.shape(matrix, histogram, popup.function,
                                         popup.center, popup.width, popup.times, popup.distance, popup.direction)
        # compute histogram
        h = Histogram.hist(matrix)
        target = Math.scale(target, max(h) / float(max(target)))
        histogram = {
            'plots': [h, target],
            'legend': ["New", "Target"]
        }
        # add new state
        self.appendHistory({
            'matrix': matrix,
            'histogram': histogram,
            'saved': False
        })
        self.updateImage(matrix)
        hist = [old['histogram']['plots'][0]] + histogram['plots']
        legend = ["Old"] + histogram['legend']
        self.windowHist.setHistogram(hist, legend)

    def menuActionMatch(self):
        # get parameters
        popup = DialogMatch(self.window)
        popup.exec_()
        if not popup.path:
            return
        # read file at popup.path
        target = cv2.imread(popup.path, 0)
        if target is None:
            print("Could not open file or file is not an image.")
            return
        # copy current state
        old = self.history[self.historyIndex]
        matrix = old['matrix'].copy()
        histogram = deepcopy(old['histogram']['plots'][0])
        # apply histogram matching
        histogram1 = Histogram.hist(target)
        matrix = Transform.match(matrix, histogram, histogram1)
        # compute histogram
        histogram = {
            'plots': [Histogram.hist(matrix), histogram1],
            'legend': ["New", "Target"]
        }
        # add new state
        self.appendHistory({
            'matrix': matrix,
            'histogram': histogram,
            'saved': False
        })
        self.updateImage(matrix)
        hist = [old['histogram']['plots'][0]] + histogram['plots']
        legend = ["Old"] + histogram['legend']
        self.windowHist.setHistogram(hist, legend)

    def menuActionHistogram(self):
        showHistogram = self.window.actionHistogram.isChecked()
        if self.windowHist is None:
            self.window.actionHistogram.setChecked(False)
            return
        if showHistogram:
            self.windowHist.show()
        else:
            self.windowHist.hide()

    def menuActionAbout(self):
        msgBox = QtWidgets.QMessageBox(self.window)
        msgBox.setWindowTitle("Intensity Transformer")
        msgBox.setText("Team: Image-ine Dragons")
        msgBox.setInformativeText("Lichao Duan, Bhavishya Kasarla, Sankalp Mogulothu, Mouad Rifai, Akhila Velma")
        msgBox.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    form = Ui_MainWindow()
    form.setupUi(window)
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
