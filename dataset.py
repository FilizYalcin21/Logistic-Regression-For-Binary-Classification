"""
@author: Filiz Yalcin
"""

import numpy


def readFile(filename):
  stringDataList = []
  with open(filename, 'r') as f:
    for line in f.readlines():
      stringDataList.append(line.replace('\n', '').split(','))
  return stringDataList

def convertStringToFloatArray(stringDataList):
  all_data = numpy.zeros((len(stringDataList), len(stringDataList[0])))
  for a in range(len(stringDataList)):
    for i in range(len(stringDataList[a])):
      all_data[a][i] = float(stringDataList[a][i])
  return all_data

# Gönderilen veriler 100 üzerinden değerlendirilen sınav sonuçları olduğundan
# normalizasyon için 0-1 arasına getirmek için basitçe 100'e bölünmüştür
def getNormalizedData(data):
  normalized_data = numpy.array(data)
  for i in range(len(data)):
    normalized_data[i][0] = data[i][0] / 100
    normalized_data[i][1] = data[i][1] / 100
  return normalized_data

def getDataSet():
    filename = './dataset/exams-dataset.txt'

    # TXT dosyadan veriler String şekilde alınıp bir listeye eklenir
    stringDataList = readFile(filename)

    # String şeklinde tutulan veriler float'a çevrilir
    all_data = convertStringToFloatArray(stringDataList)
    normalized_data = getNormalizedData(all_data)
    return normalized_data