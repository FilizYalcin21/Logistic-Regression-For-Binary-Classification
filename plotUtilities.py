"""
@author: Filiz Yalcin
"""

import matplotlib.pyplot as plt
import numpy
import seaborn as sn
import pandas as pd

'''
Gönderilen verinin 1. sütunu x, 2. sütunu y olacak şekilde noktalar ile (discrete)
grafik çizer. Gönderilen verinin 3. elemanı 1 ise mavi, 0 ise kırmızı çizer
'''
def drawData(data):
  x = data[:, 0]
  y = data[:, 1]
  labels = data[:, 2]

  # 0 (Ret) için kırmızı, 1 (Kabul) için mavi renk seçilir
  colors = numpy.where(labels == 0, 'red', 'blue')

  plt.figure(figsize=(6, 6))
  plt.scatter(x, y, c=colors, s=25)
  plt.xlabel("x1: 1. Sınav")
  plt.ylabel("x2: 2. Sınav")
  plt.title("Veri Dağılım Grafiği")
  plt.grid(True)
  plt.show()

'''
Gönderilen loss listesini sürekli (continuous) grafik olarak çizer
'''
def drawCrossEntropyLoss(crossEntropyLossList):
  sample_indexes = numpy.arange(len(crossEntropyLossList))
  plt.plot(sample_indexes, crossEntropyLossList)
  plt.xlabel("Epok")
  plt.ylabel("Ortalama Cross Entropy Loss")
  plt.title("Ortalama Loss Grafiği")
  plt.grid(True)
  plt.show()

'''
Gönderilen veri seti için verinin 1. sütunu x, 2. sütunu y olacak şekilde noktalar ile (discrete)
grafik çizer. Gönderilen verinin 3. elemanı 1 ise mavi, 0 ise kırmızı çizer
Çizilen grafiğin üstüne gönderilen ağırlık ve bias değerleri kullanılarak karar
çizgisi (hyperplane) doğru olarak çizilir
'''
def drawHyperplane(dataset, weights, bias):
    x = dataset[:, 0]
    y = dataset[:, 1]
    labels = dataset[:, 2]
    
    # 0 (Ret) için kırmızı, 1 (Kabul) için mavi renk seçilir
    colors = numpy.where(labels == 0, 'red', 'blue')
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=colors, s=25)
    
    # Grafikte 0-1 arası 11 değer görüntülenir
    x1_values = numpy.linspace(0, 1, 11)
    print(x1_values)
    
    # Denklemden x2 çıkartılır
    # w1 * x1 + w2 * x2 + b = 0  →  x2 = -(w1*x1 + b) / w2
    x2_values = -(weights[0] * x1_values + bias) / weights[1]
    print(x2_values)
    
    plt.plot(x1_values, x2_values)
    plt.xlabel("x1: 1. Sınav")
    plt.ylabel("x2: 2. Sınav")
    plt.title("Hyperplane")
    plt.grid(True)
    plt.show()

'''
Gönderilen TP, FN, FP, TN değerlerine göre karmaşıklık matrisini çizer
'''
def drawConfusionMatrix(TP, FN, FP, TN):
    array = [[TP, FN], [FP, TN]]
    df_cm = pd.DataFrame(array, index = [i for i in "10"],
                  columns = [i for i in "10"])
    plt.figure(figsize = (10, 7))
    sn.heatmap(df_cm, annot=True, cmap="crest")