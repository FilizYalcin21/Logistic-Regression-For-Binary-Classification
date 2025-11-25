"""
@author: Filiz Yalcin
"""

import numpy
import math
from model import calculateYPredicted

'''
Stochastic Gradient Descent yöntemi kullanılarak bir ağırlığın (w(t - 1))
bir sonraki değerini (w(t)) döndürür.
Argümanlar:
    oldWeight -- ağırlık değeri (w(t - 1))
    learningRate -- öğrenme katsayısı
    yTarget -- modelin tahmin etmesi beklenen etiket değeri
    yPredicted -- modelin tahmin ettiği etiket
    x -- ilgili ağırlığa karşılık gelen x değeri
'''
def calculateNextWeight(oldWeight, learningRate, yTarget, yPredicted, x):
  newWeight = oldWeight + (learningRate * (yTarget - yPredicted) * x)
  return newWeight

'''
log fonksiyonu 0 için tanımsız olduğundan 0 gönderildiğinde 0 dönmesi için
implemente edilmiştir
Argümanlar: 
    value -- logaritma işlemine gönderilecek değer
'''
def log(value):
  if value > 0:
    return math.log(value)
  else:
    return 0

'''
Cross Entropy Loss yöntemi ile kayıp hesabı yapar, loss'u döner
Argümanlar:
    yTarget -- modelin tahmin etmesi beklenen etiket
    yPredicted -- modelin tahmin ettiği etiket
'''
def calculateCrossEntropyLoss(yTarget, yPredicted):
  loss = -(yTarget * log(yPredicted) + (1 - yTarget) * log(1 - yPredicted))
  return loss

'''
Gönderilen boyut kadar 0-1 arası rastgele ağırlık değerleri dönülür
'''
def getRandomWeights(size):
    return numpy.random.rand(size)


def getValidationLoss(validation_dataset, weights, bias):
    crossEntropyLossList = []
    for validation_data in validation_dataset:
        yPredicted = calculateYPredicted(weights, validation_data[:2], bias)
        yTarget = validation_data[2] # 2 indeksinde label'lar var
        loss = calculateCrossEntropyLoss(yTarget, yPredicted)
        crossEntropyLossList.append(loss)
    return numpy.average(crossEntropyLossList)

'''
Gönderilen örnekleri (sample) kullanarak Stochastic Gradient Descent yöntemi
ile ağırlıkları günceller. Her örnek için loss hesabı yapılarak ortalama loss
değerini hesaplar. Yeni ağırlık değerlerini, yeni bias değerini ve ortalama loss
değerlerini döner.
Argümanlar:
    epoch -- epok sayısı
    sampleList -- örnek listesi
    weights -- başlangıç ağırlıkları
    bias -- başlangıç bias değeri
    learningRate -- SGD için kullanılacak öğrenme katsayısı
'''
def train(epoch, sampleList, weights, bias, learningRate, validation_dataset):
  newWeights = numpy.copy(weights) # yeni ağırlıklar tanımlanır
  averageLossList = []
  val_averageLossList = []
  for e in range(epoch):
    crossEntropyLossList = []
    for i in range(len(sampleList)):
      sample = sampleList[i]
      xValues = sample[:2]
      yPredicted = calculateYPredicted(newWeights, xValues, bias)
      yTarget = sample[2] # 2 indeksinde label'lar var, sample'ın label'ı alınır
      for j in range(len(newWeights)):
        x = sample[j] # ağırlığın indeksindeki x değeri alınır
        nextWeight = calculateNextWeight(newWeights[j], learningRate, yTarget, yPredicted, x)
        newWeights[j] = nextWeight # yeni ağırlık set edilir
      bias = calculateNextWeight(bias, learningRate, yTarget, yPredicted, 1) # w0 için x değeri 1'dir

      # yeni ağırlıklarla tekrar tahmin yaptırılır
      yPredicted = calculateYPredicted(newWeights, xValues, bias)
      # Her örnek için loss hesabı yapılır
      loss = calculateCrossEntropyLoss(yTarget, yPredicted)
      crossEntropyLossList.append(loss)

    averageLoss = numpy.average(crossEntropyLossList)
    averageLossList.append(averageLoss)
    val_averageLoss = getValidationLoss(validation_dataset, newWeights, bias)
    val_averageLossList.append(val_averageLoss)

    print("Epoch:", e+1, "- Training Loss:", averageLoss, "- Validation Loss:", val_averageLoss)
  return newWeights, bias, averageLossList, val_averageLossList


''' 1 epok ile eğitim yapılırken her bir örnek için average loss hesaplanmıştır
görsel olarak sonucu '/graphs' klasöründe bulunmaktadır
kullanmak için yorum satırı açılıp main.py dosyasında bulunan yorum satırı açılabilir

def train(epoch, sampleList, weights, bias, learningRate):
  newWeights = numpy.copy(weights) # yeni ağırlıklar tanımlanır
  crossEntropyLossList = []
  averageLossList = []
  for e in range(epoch):
    for i in range(len(sampleList)):
      sample = sampleList[i]
      xValues = sample[:2]
      yPredicted = calculateYPredicted(newWeights, xValues, bias)
      yTarget = sample[2] # 2 indeksinde label'lar var, sample'ın label'ı alınır
      for j in range(len(newWeights)):
        x = sample[j] # ağırlığın indeksindeki x değeri alınır
        nextWeight = calculateNextWeight(newWeights[j], learningRate, yTarget, yPredicted, x)
        newWeights[j] = nextWeight # yeni ağırlık set edilir
      bias = calculateNextWeight(bias, learningRate, yTarget, yPredicted, 1) # w0 için x değeri 1'dir
      loss = calculateCrossEntropyLoss(yTarget, yPredicted)
      crossEntropyLossList.append(loss)
      averageLoss = numpy.average(crossEntropyLossList)
      averageLossList.append(averageLoss)
  return newWeights, bias, averageLossList
'''