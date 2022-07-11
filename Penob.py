import scipy.spatial.transform._rotation_groups
from aplikasi import Ui_MainWindow
from scipy.spatial import distance as dist
import mahotas
import imutils
import sys
import cv2
import math
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import numpy as np
from matplotlib import pyplot as plt
import ctypes

def konvolusi(image, kernel): #fungsi konvolusi dengan dua input image dan kernel
    X_h, X_w = image.shape[:2] #untuk mencari lebar dan tinggi image
    F_h, F_w = kernel.shape[:2] #untuk mencari lebar dan tinggi kernel
    H = F_h // 2 #hasil pembagian tinggi kernel di inputkan ke dalam variabel H
    W = F_w // 2 #hasil pembagian lebar kernel di inputkan ke dalam variabel W
    out = np.zeros((X_h, X_w)) #variabel keluaran sebagai array image
    for i in np.arange(H + 1, X_h - H): #perulangan untuk tinggi citra
        for j in np.arange(W + 1, X_w - W): #perulangan untuk lebar citra
            sum = 0 #inisialisasi variabel sum
            for k in np.arange(-H, H + 1): #perulangan untuk tinggi kernel
                for l in np.arange(-W, W + 1): #perulangan untuk lebar kernel
                    a = image[i + k, j + l] #rumus perhitungan nilai pixel citra
                    w = kernel[H + k, W + l] #rumus perhitungan bobot kernel
                    sum = sum + (w * a) #perhitungan variabel w kali a disimpan dalam sum
            out[i, j] = sum #memasukkan nilai sum ke array out

    return out #kembalikan ke prosedur

def describe_shapes(image):
    # initialize the list of shape features
    shapeFeatures = []

    # convert the image to grayscale, blur it, and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    image = blurred  # import citra
    bins_num = 256  # set jumlah total bins dalam histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)  # untuk mendapatkan histogram gambar
    hist = np.divide(hist.ravel(), hist.max())  # menormalisasi histogram
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2  # mencari nilai tengah bins
    # Iterasi semua ambang (indeks) dan dapatkan probabilitas w1 (t), w2 (t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_mids) / weight1  # Dapatkan kelas means mu0 (t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]  # Dapatkan kelas means mu1 (t)
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    index_of_max_val = np.argmax(inter_class_variance)  # Maksimalkan val fungsi inter_class_variance
    threshold = bin_mids[:-1][index_of_max_val]
    threshold, otsu = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # perform a series of dilations and erosions to close holes
    # in the shapes
    closing = cv2.dilate(otsu, None, iterations=4)
    closing = cv2.erode(closing, None, iterations=2)
    # detect contours in the edge map
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #value = mahotas.features.zernike_moments(thresh, 10)
    # loop over the contours
    for c in cnts:
        # create an empty mask for the contour and draw it
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # extract the bounding box ROI from the mask
        (x, y, w, h) = cv2.boundingRect(c)
        roi = mask[y:y + h, x:x + w]

        # compute Zernike Moments for the ROI and update the list
        # of shape features
        features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
        shapeFeatures.append(features)
    #return a tuple of the contours and shapes
    return (blurred),(otsu),(closing),(threshold),(cnts, shapeFeatures)

class Showimage(QMainWindow,Ui_MainWindow): #,Ui_MainWindow
    def __init__(self, parent=None):
        super(Showimage, self).__init__(parent) #parent
        self.setupUi(self)
        self.Image = None
        self.Image2 = None
        self.Image3 = None
        self.list = None
        self.cnts = None

        self.buttonZernike.setEnabled(False)
        #FILE
        self.Load.triggered.connect(self.open)
        self.Save_image.triggered.connect(self.save)
        self.Save_pixel.triggered.connect(self.pixelsave)
        #OPERASI INTERFACE
        self.load_citra.clicked.connect(self.open)
        self.load_citra_2.clicked.connect(self.open2)
        self.buttonCrop.clicked.connect(self.crop)
        self.buttonCrop_2.clicked.connect(self.crop2)
        self.buttonLatih.clicked.connect(self.latih)
        self.buttonZernike.clicked.connect(self.zernike)
        self.buttonReset.clicked.connect(self.reset)
        #HISTOGRAM
        self.actionRGB.triggered.connect(self.hisRgb)
        self.actionGrayscale.triggered.connect(self.hisGrayscale)
        self.actionHistogramEqualization.triggered.connect(self.EqualHistogram)
        # SLIDER
        self.SliderBrightness.valueChanged.connect(self.sliderbrightness)
        self.SliderContrast.valueChanged.connect(self.slidercontrast)
        # OPERASI TITIK
        self.actionOperasiPencerahan.triggered.connect(self.brightness)
        self.actionSimpleContrast.triggered.connect(self.contrast)
        self.actionContrastStreching.triggered.connect(self.contrastStreching)
        self.actionNegatifImage.triggered.connect(self.negatif)
        self.actionBinerImage.triggered.connect(self.biner)
        # OPERASI GEOMETRI
        self.action_Translasi.triggered.connect(self.translasi)
        self.action_45_Derajat_2.triggered.connect(self.rotasimin45derajat)
        self.action45_Derajat_2.triggered.connect(self.rotasi45derajat)
        self.action_90_Derajat_2.triggered.connect(self.rotasimin90derajat)
        self.action90_Derajat_2.triggered.connect(self.rotasi90derajat)
        self.action180_Derajat_2.triggered.connect(self.rotasi180derajat)
        self.action_Transpose.triggered.connect(self.transpose)
        self.action_Skewed.triggered.connect(self.skew)
        # ZOOMIN ZOOMOUT
        self.action_2x.triggered.connect(self.zoomIn2x)
        self.action_3x.triggered.connect(self.zoomIn3x)
        self.action_4x.triggered.connect(self.zoomIn4x)
        self.action1per2.triggered.connect(self.zoomOutSetengah)
        self.action1per4.triggered.connect(self.zoomOutSeperempat)
        self.action3per4.triggered.connect(self.zoomOutTigaperempat)
        # OPERASI BOOLEAN
        self.actionAND.triggered.connect(self.operasiAND)
        self.actionOR.triggered.connect(self.operasiOR)
        self.actionXOR.triggered.connect(self.operasiXOR)
        # OPERASI SPASIAL
        self.actionKernel_1.triggered.connect(self.filtering1)
        self.actionKernel_2.triggered.connect(self.filtering2)
        self.actionKernel_3.triggered.connect(self.smoothingMean1)
        self.actionKernel_4.triggered.connect(self.smoothingMean2)
        self.actionSmoothing_GaussianFilter.triggered.connect(self.smoothingGaussian)
        self.actionKernel_5.triggered.connect(self.sharpeningKernel1)
        self.actionKernel_6.triggered.connect(self.sharpeningKernel2)
        self.actionKernel_7.triggered.connect(self.sharpeningKernel3)
        self.actionKernel_8.triggered.connect(self.sharpeningKernel4)
        self.actionKernel_9.triggered.connect(self.sharpeningKernel5)
        self.actionKernel_10.triggered.connect(self.sharpeningKernel6)
        self.actionLaplace.triggered.connect(self.sharpeningLaplace)
        self.actionMedian.triggered.connect(self.median)
        self.actionMaxFilter.triggered.connect(self.maxFiltering)
        self.actionMinFilter.triggered.connect(self.minFiltering)
        # OPERASI TRESHOLDING
        self.action_Binary.triggered.connect(self.threshBin)
        self.action_Binary_Invers.triggered.connect(self.threshBinInv)
        self.action_Trunc.triggered.connect(self.threshTrunc)
        self.actionTo_Zero.triggered.connect(self.threshTozero)
        self.actionTo_Zero_Invers.triggered.connect(self.threshTozeroInv)
        self.action_Otsu.triggered.connect(self.otsuThresh)
        self.actionGaussian_Thresholding.triggered.connect(self.threshGaussian)
        self.action_Adaptive_Mean.triggered.connect(self.threshMean)
        # Transformasi
        self.actionDFT_Smoothing_Image.triggered.connect(self.DFTSmoothing)
        self.actionDFT_Edge.triggered.connect(self.DFTEdge)
        # Deteksi Tepi
        self.actionSobel.triggered.connect(self.Sobel)
        self.actionPrewitt.triggered.connect(self.Prewitt)
        self.actionRoberts.triggered.connect(self.Roberts)
        self.actionCany.triggered.connect(self.cany)


    def open(self):
        filename, filter = QFileDialog.getOpenFileName(self, 'Open File','D:\Perkuliahan\semester 4\Pengolahan Citra Digital (PCD)\Tugas Besar\Program RTM4',"Image Files(*.jpg)""\nImage Files(*.jpeg)""\nImage Files(*.png)")
        if filename:
            self.Image = cv2.imread(filename)
            self.displayImage(1)
        else:
            print('Gagal Memuat')
        self.label_4.setText(filename)

    def open2(self):
        filename, filter = QFileDialog.getOpenFileName(self, 'Open File','D:\Perkuliahan\semester 4\Pengolahan Citra Digital (PCD)\Tugas Besar\Program RTM4',"Image Files(*.jpg)""\nImage Files(*.jpeg)""\nImage Files(*.png)")
        if filename:
            self.Image2 = cv2.imread(filename)
            self.displayImage2(1)
        else:
            print('Gagal Memuat')
        self.label_5.setText(filename)

    def crop(self):
        image = self.Image #memanggil citra dan dimasukkan ke dalam variabel image
        roi = cv2.selectROI(image) #menggunakan fungsi ROI pada cv2 untuk menentukan daerah yang ingin dilakukan proses crop
        print (roi) #menampilkan jendela ROI
        image_cropped = image[int(roi[1]):int(roi[1] + roi[3]), #proses pemotongan citra sesuai inputan ROI
                        int(roi[0]):int(roi[0] + roi[2])]
        resize_image = cv2.resize(image_cropped, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC) #proses resize citra setelah dilakukan pemotongan
        self.Image = resize_image #hasil resize citra dimasukkan ke dalam variabel
        self.displayImage(1) #menampilkan citra

    def crop2(self):
        image = self.Image2 #memanggil citra dan dimasukkan ke dalam variabel image
        roi = cv2.selectROI(image) #menggunakan fungsi ROI pada cv2 untuk menentukan daerah yang ingin dilakukan proses crop
        print (roi) #menampilkan jendela ROI
        image_cropped = image[int(roi[1]):int(roi[1] + roi[3]), #proses pemotongan citra sesuai inputan ROI
                        int(roi[0]):int(roi[0] + roi[2])]
        resize_image = cv2.resize(image_cropped, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC) #proses resize citra setelah dilakukan pemotongan
        self.Image2 = resize_image #hasil resize citra dimasukkan ke dalam variabel
        self.displayImage2(1) #menampilkan citra

    def save(self):
        filename, filter = QFileDialog.getSaveFileName(self, 'Save File','D:\Perkuliahan\semester 4\Pengolahan Citra Digital (PCD)\Tugas Besar',"JPG Image (*.jpg)")
        if filename:
            cv2.imwrite(filename, self.Image)
        else:
            print('Tidak Dapat Menyimpan')

    def pixelsave(self):
        filename, filter = QFileDialog.getSaveFileName(self, 'Save File', 'D:\Perkuliahan\semester 4\Pengolahan Citra Digital (PCD)\Tugas Besar',"TXT File (*.txt)""\nCSV File (*.csv)")
        if filename:
            np.savetxt(filename, self.Image)
        else:
            print('Tidak Dapat Menyimpan')

    def reset(self):
        self.label.clear()
        self.label_pertama.clear()
        self.label_Thres.clear()
        self.label_Gauss.clear()
        self.label_Closing.clear()
        self.label_hasil.clear()
        self.label_4.clear()
        self.label_6.clear()
        self.label_Hasil.clear()
        self.label_5.clear()
        self.label_14.clear()
        self.buttonZernike.setEnabled(False)

    def hisRgb(self):
        image = self.Image
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([image], [i], None, [256],[0, 256])
            plt.plot(histo, color=col)
        plt.show()

    def hisGrayscale(self):
        image = self.Image
        plt.hist(image.ravel(), 255, [0, 255])
        plt.show()

    def EqualHistogram(self): #nama prosedur yang dapat di panggil oleh button
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256]) #inisialisasi hist dan bins sebagai numpy histogram dengan rentang 0 sampai 256
        cdf = hist.cumsum() #inisialisasi cdf dengan perintah hasil perhitungan kumulatif nilai
        cdf_normalized = cdf * hist.max() / cdf.max() #proses normalisasi dengan rumus
        cdf_m = np.ma.masked_equal(cdf, 0) #proses masking value atau nilai
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) #rumus perataan citra
        cdf = np.ma.filled(cdf_m, 0).astype('uint8') #mengisikan cdf
        self.Image = cdf[self.Image] #inisialisasi kembali cdf
        self.displayImage(5) #menampilkan image dijendela ke 2

        plt.plot(cdf_normalized, color='b') #untuk menormalisasi warna biru
        plt.hist(self.Image.flatten(), 256, [0, 256], color="r") #perhitungan image dengan rentang 0 sampai 256 dengan acuan warna mrah
        plt.xlim([0, 256]) #rentang xlim 0 sampai 256
        plt.legend(('cdf', 'histogram'), loc='upper left') #perintah untuk membuat keterangan pada histogram
        plt.show() #tampilkan histogram

    def grayscale(self):
        H, W = self.Image.shape[:2] #mengukur tinggi dan lebar citra
        gray = np.zeros((H, W), np.uint8) #membuat array dari ukuran citra yang telah diketahui
        for i in range(H): #perulangan tinggi citra
            for j in range(W): #perulangan lebar citra
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] + #proses perhitungan RGB menjadi grayscale
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255) #rentang derajat keabuan dan cropping
        self.Image = gray #citra dimasukkan ke dalam variabel
        self.displayImage(2) #menampilkan citra

    def gaussianFilter(self):
        image = self.Image  # import citra
        blur = cv2.GaussianBlur(image, (5, 5), 0)  # fungsi filter gaussian untuk memperbaiki citra
        self.Image = blur
        self.displayImage(3)  # menampilkan citra

    def threshOtsu(self):
        image = self.Image  # import citra
        bins_num = 256 #set jumlah total bins dalam histogram
        hist, bin_edges = np.histogram(image, bins=bins_num) #untuk mendapatkan histogram gambar
        hist = np.divide(hist.ravel(), hist.max()) #menormalisasi histogram
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2 #mencari nilai tengah bins
        #Iterasi semua ambang (indeks) dan dapatkan probabilitas w1 (t), w2 (t)
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        mean1 = np.cumsum(hist * bin_mids) / weight1 #Dapatkan kelas means mu0 (t)
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1] #Dapatkan kelas means mu1 (t)
        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        index_of_max_val = np.argmax(inter_class_variance) #Maksimalkan val fungsi inter_class_variance
        threshold = bin_mids[:-1][index_of_max_val]
        threshold, thresh = cv2.threshold(image, threshold, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) #operasi thresholding otsu
        self.Image = thresh #memasukkan image yang telah di proses otsu thresholding ke dalam variabel
        self.displayImage(4) #menampilkan citra

        self.label_6.setNum(threshold)

    def biner(self,threshold):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                if a == threshold :
                    a = 0
                elif a < threshold :
                    a = 1
                elif a > threshold :
                    a = 255

                self.Image.itemset((i, j), a)
        print(threshold)
        self.displayImage(5)

    def sliderbrightness(self,value):
        # agar menghindari error ketika melewati proses Grayscaling Citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + value, 0, 255)
                self.Image.itemset((i,j), b)
        self.displayImage(5)

    def slidercontrast(self,value):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + value, 0, 255)
                self.Image.itemset((i, j), b)
        self.displayImage(5)

    def brightness(self):
        # agar menghindari error ketika melewati proses Grayscaling Citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)
                self.Image.itemset((i,j), b)
        self.displayImage(5)

    def contrast(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + contrast, 0, 255)
                self.Image.itemset((i, j), b)
        self.displayImage(5)

    def contrastStreching(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.Image.itemset((i, j), b)

        self.displayImage(5)

    def negatif(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = math.ceil(255 - a)

                self.Image.itemset((i, j), b)

        self.displayImage(5)

    def translasi(self):
        h, w = self.Image.shape[:2]
        quarter_h, quarter_w = h/8, w/4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w, h))
        self.Image = img
        self.displayImage(5)

    def rotasimin45derajat(self):
        self.rotasi(-45)

    def rotasi45derajat(self):
        self.rotasi(45)

    def rotasimin90derajat(self):
        self.rotasi(-90)

    def rotasi90derajat(self):
        self.rotasi(90)

    def rotasi180derajat(self):
        self.rotasi(180)

    def rotasi(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rotationMatrix[0, 2] += (nW / 2) - w /2
        rotationMatrix[1, 2] += (nH / 2) - h /2

        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))
        self.Image = rot_image
        self.displayImage(5)

    def transpose(self):
        tran_image = cv2.transpose(self.Image)
        self.Image = tran_image
        self.displayImage(5)

    def zoomIn2x(self):
        self.zoomIn(2)

    def zoomIn3x(self):
        self.zoomIn(3)

    def zoomIn4x(self):
        self.zoomIn(4)

    def zoomIn(self, skala):
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('original', self.Image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def zoomOutSetengah(self):
        self.zoomOut(1/2)

    def zoomOutSeperempat(self):
        self.zoomOut(1/4)

    def zoomOutTigaperempat(self):
        self.zoomOut(3/4)

    def zoomOut(self, skala):
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('original', self.Image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def skew(self):
        self.skewed(1,3)

    def skewed(self, x, y):
        resize_image = cv2.resize(self.Image, None, fx=x, fy=y, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('original', self.Image)
        cv2.imshow('Skewed', resize_image)
        cv2.waitKey()

    def operasiAND(self):
        image1 = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(self.Image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_and(image1, image2)
        self.Image = image1
        self.displayImage(1)
        self.Image = image2
        self.displayImage(6)
        self.Image = operasi
        self.displayImage(5)

    def operasiOR(self):
        image1 = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(self.Image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_or(image1, image2)
        self.Image = image1
        self.displayImage(1)
        self.Image = image2
        self.displayImage(6)
        self.Image = operasi
        self.displayImage(5)

    def operasiXOR(self):
        image1 = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(self.Image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_xor(image1, image2)
        self.Image = image1
        self.displayImage(1)
        self.Image = image2
        self.displayImage(6)
        self.Image = operasi
        self.displayImage(5)

    def filtering1(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY) #mengambil image dan merubah ke dalam citra grayscale
        kernel = np.array( #kernel yang ingin dimasukkan
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]])
        img_out = konvolusi(image, kernel) #pemanggilan fungsi dan di simpan dalam variabel img_out
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')  # menampilkan hasil citra dari fungsi
        plt.title('Filtering kernel 1')  # untuk memberi title keluaran
        plt.xticks([]), plt.yticks([])  # untuk menghilangakan koordinat garis skala grafik
        plt.show()
        self.Image = img_out

    def filtering2(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY) #mengambil image dan merubah ke dalam citra grayscale
        kernel = np.array( #kernel yang digunakan
            [[6, 0, -6],
             [6, 1, -6],
             [6, 0, -6]])
        img_out = konvolusi(image, kernel) #pemanggilan fungsi dan di simpan dalam variabel img_out
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')  # menampilkan hasil citra dari fungsi
        plt.title('Filtering kernel 2')  # untuk memberi title keluaran
        plt.xticks([]), plt.yticks([])  # untuk menghilangakan koordinat garis skala grafik
        plt.show()
        self.Image = img_out

    def smoothingMean1(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        mean = (1/9) * np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]])
        img_out = konvolusi(image, mean)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.title('Smoothing kernel 1')
        plt.xticks([]), plt.yticks([])
        plt.show()
        self.Image = img_out

    def smoothingMean2(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        mean = (1/4) * np.array(
            [[1, 1, 0],
             [1, 1, 0],
             [0, 0, 0]])
        img_out = konvolusi(image, mean)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.title('Smoothing kernel 2')
        plt.xticks([]), plt.yticks([])
        plt.show()
        self.Image = img_out

    def smoothingGaussian(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY) #import image yang kemudian di grayscaling
        gauss = (1/345) * np.array( #array gaussian filter
            [[1, 5, 7, 5, 1],
             [5, 20, 33, 20, 5],
             [7, 33, 55, 33, 7],
             [5, 20, 33, 20, 5],
             [1, 5, 7, 5, 1]])
        img_out = konvolusi(image, gauss) #pemanggilan fungsi konvolusi dengan variabel inputan image dan kernel
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.title('Smoothing gaussian')
        plt.xticks([]), plt.yticks([])
        plt.show()
        self.Image = img_out #hasil image dimasukkan kedalam variabel

    def sharpeningKernel1(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        kernel1 = np.array(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]])
        img_out = konvolusi(image, kernel1)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.title('Sharpening kernel 1')
        plt.xticks([]), plt.yticks([])
        plt.show()
        self.Image = img_out

    def sharpeningKernel2(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        kernel2 = np.array(
            [[-1, -1, -1],
             [-1,  9, -1],
             [-1, -1, -1]])
        img_out = konvolusi(image, kernel2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.title('Sharpening kernel 2')
        plt.xticks([]), plt.yticks([])
        plt.show()
        self.Image = img_out

    def sharpeningKernel3(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        kernel3 = np.array(
            [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]])
        img_out = konvolusi(image, kernel3)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.title('Sharpening kernel 3')
        plt.xticks([]), plt.yticks([])
        plt.show()
        self.Image = img_out

    def sharpeningKernel4(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        kernel4 = np.array(
            [[ 1, -2,  1],
             [-2,  5, -2],
             [ 1, -2,  1]])
        img_out = konvolusi(image, kernel4)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.title('Sharpening kernel 4')
        plt.xticks([]), plt.yticks([])
        plt.show()
        self.Image = img_out

    def sharpeningKernel5(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        kernel5 = np.array(
            [[ 1, -2,  1],
             [-2,  4, -2],
             [ 1, -2,  1]])
        img_out = konvolusi(image, kernel5)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.title('Sharpening kernel 5')
        plt.xticks([]), plt.yticks([])
        plt.show()
        self.Image = img_out

    def sharpeningKernel6(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        kernel6 = np.array(
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]])
        img_out = konvolusi(image, kernel6)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.title('Sharpening kernel 6')
        plt.xticks([]), plt.yticks([])
        plt.show()
        self.Image = img_out

    def sharpeningLaplace(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        laplace = (1/16) * np.array(
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 16, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]])
        img_out = konvolusi(image, laplace)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.title('Sharpening laplace')
        plt.xticks([]), plt.yticks([])
        plt.show()
        self.Image = img_out

    def median(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY) #mengambil image dan merubah ke dalam citra grayscale
        image_out = image.copy() #inisialisasi img_out seperti variabel image
        h, w = image.shape[:2] #mencari tinggi dan lebar image
        for i in np.arange(3, h - 3): #perulangan untuk mengecek nilai pixel tinggi image
            for j in np.arange(3, w - 3): #perulangan untuk mengecek nilai pixel lebar image
                neighbors = [] #inisialisasi array kosong untuk menyimpan
                for k in np.arange(-3, 4): #perulangan tinggi kernel
                    for l in np.arange(-3, 4): #perulangan lebar kernel
                        a = image.item(i + k, j + l) #perhitungan dari perulangan dimasukan kedalam variabel a
                        neighbors.append(a) #menambahkan nilai a ke array neighbors

                neighbors.sort() #untuk mengurutkan array
                median = neighbors[24] #posisi median di array 24 dimasukan ke variabel median
                b = median #nilai median dimasukan ke variabel
                image_out.itemset((i, j), b) #mengganti nilai array i,j menjadi b

        self.Image = image_out
        self.displayImage(5)

    def maxFiltering(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        image_out = image.copy()
        h, w = image.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                max = 0 #inisialisasi nilai variabel max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = image.item(i + k, j + l)
                        if a > max: #jika nilai a lebih besar dari variabel max
                            max = a #maka nilai a dimasukkan ke dalam variabel max
                b = max #nilai max dimasukkan ke dalam variabel b
                image_out.itemset((i, j), b) #mengganti nilai array i,j menjadi b

        self.Image = image_out
        self.displayImage(5)

    def minFiltering(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        image_out = image.copy()
        h, w = image.shape[:2]
        for i in np.arange(3, h - 3):
            for j in np.arange(3, w - 3):
                min = 255 #inisialisasi nilai variabel min = 255
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = image.item(i + k, j + l)
                        if a < min: #jika nilai a lebih besar dari variabel min
                            min = a #maka nilai a dimasukkan ke dalam variabel min
                b = min #nilai min dimasukkan ke dalam variabel b
                image_out.itemset((i, j), b) #mengganti nilai array i,j menjadi b

        self.Image = image_out
        self.displayImage(5)

    def threshBin(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
        self.Image = thresh
        self.displayImage(5)

    def threshBinInv(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)
        self.Image = thresh
        self.displayImage(5)

    def threshTrunc(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_TRUNC)
        self.Image = thresh
        self.displayImage(5)

    def threshTozero(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_TOZERO)
        self.Image = thresh
        self.displayImage(5)

    def threshTozeroInv(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_TOZERO_INV)
        self.Image = thresh
        self.displayImage(5)

    def otsuThresh(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.Image = thresh
        self.displayImage(5)

    def threshGaussian(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
        self.Image = thresh
        self.displayImage(5)

    def threshMean(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
        self.Image = thresh
        self.displayImage(5)

    def DFTSmoothing(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY) #merubah image menjadi grayscale
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT) #merubah citra int 8bit menjadi float32  #proses untuk melakukan transformasi diskrit
        dft_shift = np.fft.fftshift(dft) #proses rearange nilai x untuk menshifting titik 0,0 ke center

        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))) #proses perhitungan spektrum menghasilkan channel bilangan real dan bilangan imaginer

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        #menggunakan low pass filter
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 50 #radius
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1 #nilai center

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift) #mengebalikan titik origin
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]) #mengembalikan data ke bentuk spasial

        fig = plt.figure(figsize=(12, 12))
        axl = fig.add_subplot(2, 2, 1)
        axl.imshow(img, cmap='gray')
        axl.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()

    def DFTEdge(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        # menggunakan high pass filter
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        axl = fig.add_subplot(2, 2, 1)
        axl.imshow(img, cmap='gray')
        axl.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()

    def Sobel(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        img_x = konvolusi(img, Sx)
        img_y = konvolusi(img, Sy)
        img_out = np.sqrt((img_x * img_x) + (img_y * img_y)) #hitung gradient
        img_out = (img_out / np.max(img_out)) * 255 #menormaliasi panjang gradient
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    def Prewitt(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        Px = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
        Py = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
        img_x = konvolusi(img, Px)
        img_y = konvolusi(img, Py)
        img_out = np.sqrt((img_x * img_x) + (img_y * img_y)) #hitung gradient
        img_out = (img_out / np.max(img_out)) * 255 #menormaliasi panjang gradient
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    def Roberts(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        Px = np.array([[1, 0],
                       [0, -1]])
        Py = np.array([[0, 1],
                       [-1, 0]])
        img_x = konvolusi(img, Px)
        img_y = konvolusi(img, Py)
        img_out = np.sqrt((img_x * img_x) + (img_y * img_y))
        img_out = (img_out / np.max(img_out)) * 255
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()

    def cany(self):
        image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        H, W = self.Image.shape[:2]
        # Reduksi Noise
        gauss = (1 / 57) * np.array(
            [[0, 1, 2, 1, 0],
             [1, 3, 5, 3, 1],
             [2, 5, 9, 5, 2],
             [1, 3, 5, 3, 1],
             [0, 1, 2, 1, 0]])
        img_out = konvolusi(image, gauss)
        img_out = img_out.astype("uint8")
        cv2.imshow("Noise Reduction", img_out)

        # Finding Gradien
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        img_x = konvolusi(img_out, Sx)
        img_y = konvolusi(img_out, Sy)
        img_Grad = np.hypot(img_x, img_y)
        img_Grad = (img_Grad / np.max(img_Grad)) * 255
        cv2.imshow("Finding Gradien", img_Grad)

        # non max suppression
        theta = np.arctan2(img_y, img_x)
        angle = theta * 180. / np.pi
        Z = np.zeros((H, W), dtype=np.int32)
        angle[angle < 0] += 180
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img_out[i, j + 1]
                        r = img_out[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_out[i + 1, j]
                        r = img_out[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_out[i - 1, j - 1]
                        r = img_out[i + 1, j + 1]
                    if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                        Z[i, j] = img_out[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        img_N = Z.astype("uint8")
        cv2.imshow("non max supression", img_N)

        #menentukan nilai ambang atas dan bawah
        weak = 100
        strong = 150
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak):  # weak
                    b = weak
                    if (a > strong):  # strong
                        b = 255
                else:
                    b = 0

                img_N.itemset((i, j), b)
        img_H1 = img_N.astype("uint8")
        cv2.imshow("hysteresis part 1", img_H1)

        # hysteresis Thresholding eliminasi titik tepi lemah jika tidak terhubungdengantetanggatepikuat
        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or
                                (img_H1[i + 1, j] == strong) or
                                (img_H1[i + 1, j + 1] == strong) or
                                (img_H1[i, j - 1] == strong) or (img_H1[i, j + 1] == strong) or
                                (img_H1[i - 1, j - 1] == strong) or
                                (img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass

        img_H2 = img_H1.astype("uint8")
        cv2.imshow("hysteresis part 2", img_H2)

    def latih(self):
        shapeFeatures = []
        refImage = self.Image

        # convert the image to grayscale, blur it, and threshold it
        gray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        image = blurred  # import citra
        bins_num = 256  # set jumlah total bins dalam histogram
        hist, bin_edges = np.histogram(image, bins=bins_num)  # untuk mendapatkan histogram gambar
        hist = np.divide(hist.ravel(), hist.max())  # menormalisasi histogram
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2  # mencari nilai tengah bins
        # Iterasi semua ambang (indeks) dan dapatkan probabilitas w1 (t), w2 (t)
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        mean1 = np.cumsum(hist * bin_mids) / weight1  # Dapatkan kelas means mu0 (t)
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]  # Dapatkan kelas means mu1 (t)
        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        index_of_max_val = np.argmax(inter_class_variance)  # Maksimalkan val fungsi inter_class_variance
        threshold = bin_mids[:-1][index_of_max_val]
        threshold, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # perform a series of dilations and erosions to close holes
        # in the shapes
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.erode(thresh, None, iterations=2)

        # detect contours in the edge map

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # value = mahotas.features.zernike_moments(thresh, 10)
        # loop over the contours
        for c in cnts:
            # create an empty mask for the contour and draw it
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            # extract the bounding box ROI from the mask
            (x, y, w, h) = cv2.boundingRect(c)
            roi = mask[y:y + h, x:x + w]

            # compute Zernike Moments for the ROI and update the list
            # of shape features
            features = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(c)[1], degree=8)
            shapeFeatures.append(features)
        self.label_6.setNum(threshold)
        self.list = shapeFeatures
        self.cnts = cnts
        self.buttonZernike.setEnabled(True)
        ctypes.windll.user32.MessageBoxW(0, "Data latih berhasil", "Pemberitahuan", 1)

    def zernike(self):
        # load the reference image containing the object we want to detect,
        # then describe the game region
        (_, latFeatures) = self.cnts, self.list
        # load the shapes image, then describe each of the images in the image
        shapesImage = self.Image2
        (blurred),(otsu),(closing),(threshold),(cnts, shapeFeatures) = describe_shapes(shapesImage)
        self.Image3 = blurred
        self.displayImage3(2)
        self.Image3 = otsu
        self.displayImage3(3)
        self.Image3 = closing
        self.displayImage3(4)
        self.label_14.setNum(threshold)
        # compute the Euclidean distances between the video game features
        # and all other shapes in the second image, then find index of the
        # smallest distance
        D = dist.cdist(latFeatures, shapeFeatures)
        i = np.argmin(D) #untuk mendapatkan nilai index minimum

        print("Nilai hasil pencocokan")
        print(D)
        print("Nilai (i)")
        print(i)
        if (i == 4) :
            self.label_Hasil.setText("Gunting")
            # draw the bounding box around the detected shape
            box = cv2.minAreaRect(cnts[i])
            box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
            cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(cnts[i])
            cv2.putText(shapesImage, "Gunting", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        elif (i == 5) :
            self.label_Hasil.setText("Smartphone")
            # draw the bounding box around the detected shape
            box = cv2.minAreaRect(cnts[i])
            box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
            cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(cnts[i])
            cv2.putText(shapesImage, "Smartphone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        elif (i == 6):
            self.label_Hasil.setText("Pisau")
            # draw the bounding box around the detected shape
            box = cv2.minAreaRect(cnts[i])
            box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
            cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(cnts[i])
            cv2.putText(shapesImage, "Pisau", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        elif (i == 7):
            self.label_Hasil.setText("Kunci")
            # draw the bounding box around the detected shape
            box = cv2.minAreaRect(cnts[i])
            box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
            cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(cnts[i])
            cv2.putText(shapesImage, "Kunci!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        elif (i == 8):
            self.label_Hasil.setText("Koin")
            # draw the bounding box around the detected shape
            box = cv2.minAreaRect(cnts[i])
            box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
            cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(cnts[i])
            cv2.putText(shapesImage, "Koin", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        elif (i == 11):
            self.label_Hasil.setText("Kertas")
            # draw the bounding box around the detected shape
            box = cv2.minAreaRect(cnts[i])
            box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
            cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(cnts[i])
            cv2.putText(shapesImage, "Kertas", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        elif (i == 12):
            self.label_Hasil.setText("Pulpen")
            # draw the bounding box around the detected shape
            box = cv2.minAreaRect(cnts[i])
            box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
            cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(cnts[i])
            cv2.putText(shapesImage, "Pulpen", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        else :
            self.label_Hasil.setText("Tidak dapat dikenali")
            # draw the bounding box around the detected shape
            box = cv2.minAreaRect(cnts[i])
            box = np.int0(cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box))
            cv2.drawContours(shapesImage, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(cnts[i])
            cv2.putText(shapesImage, "Detect!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        # show the output images
        self.Image3 = shapesImage
        self.displayImage3(1)


    def displayImage(self,window):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                     self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        if window == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)
        elif window == 2:
            self.label_Closing.setPixmap(QPixmap.fromImage(img))
            self.label_Closing.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_Closing.setScaledContents(True)
        elif window == 3:
            self.label_Gauss.setPixmap(QPixmap.fromImage(img))
            self.label_Gauss.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_Gauss.setScaledContents(True)
        elif window == 4:
            self.label_Thres.setPixmap(QPixmap.fromImage(img))
            self.label_Thres.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_Thres.setScaledContents(True)
        elif window == 5:
            self.label_hasil.setPixmap(QPixmap.fromImage(img))
            self.label_hasil.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_hasil.setScaledContents(True)
        elif window == 6:
            self.label_pertama.setPixmap(QPixmap.fromImage(img))
            self.label_pertama.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_pertama.setScaledContents(True)

    def displayImage2(self, window):
        qformat = QImage.Format_Indexed8

        if len(self.Image2.shape)==3:
            if(self.Image2.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image2, self.Image2.shape[1], self.Image2.shape[0],
                     self.Image2.strides[0], qformat)

        img = img.rgbSwapped()

        if window == 1:
            self.label_pertama.setPixmap(QPixmap.fromImage(img))
            self.label_pertama.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_pertama.setScaledContents(True)

    def displayImage3(self, window):
        qformat = QImage.Format_Indexed8

        if len(self.Image3.shape)==3:
            if(self.Image3.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image3, self.Image3.shape[1], self.Image3.shape[0],
                     self.Image3.strides[0], qformat)

        img = img.rgbSwapped()

        if window == 1:
            self.label_hasil.setPixmap(QPixmap.fromImage(img))
            self.label_hasil.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_hasil.setScaledContents(True)
        elif window == 2:
            self.label_Gauss.setPixmap(QPixmap.fromImage(img))
            self.label_Gauss.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_Gauss.setScaledContents(True)
        elif window == 3:
            self.label_Thres.setPixmap(QPixmap.fromImage(img))
            self.label_Thres.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_Thres.setScaledContents(True)
        elif window == 4:
            self.label_Closing.setPixmap(QPixmap.fromImage(img))
            self.label_Closing.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_Closing.setScaledContents(True)

app = QtWidgets.QApplication(sys.argv)
window = Showimage()
window.setWindowTitle('Aplikasi Pendeteksi Objek Menggunakan Otsu Thresholding')
window.show()
sys.exit(app.exec())