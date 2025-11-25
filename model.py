"""
@author: Filiz Yalcin
"""

import numpy

'''
Sigmoid fonksiyonunun implementasyonudur.
'''
def sigmoid(value):
  return 1 / (1 + numpy.exp(-value))

'''
Gönderilen ağırlık, bias ve x değerleri ile bir sonuç elde edilip
sigmoid fonksiyonundan geçirilerek sonuç dönülür.
Argümanlar:
    weights -- ağırlık değerlerini içeren dizi
    xValues -- x değerlerini içeren dizi
    bias -- bias değeri
'''
def calculateYPredicted(weights, xValues, bias):
  transposed_weights = numpy.transpose(weights)
  result = (transposed_weights @ xValues) + bias # wT ile x vektörü dot product işlemine tabi tutularak bias değeri eklenir
  return sigmoid(result)
