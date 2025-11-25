"""
@author: Filiz Yalcin
"""
from model import calculateYPredicted


'''
Gönderilen yPredicted ve yTarget'a bakarak tahminin hangi duruma (TP, FP, FN, TN)
ait olduğunu bulur ve bu durum için değeri bir arttırır. TP, FP, FN, TN değerlerinin
son hâlini döner.
'''
def addToMetricVariables(TP, FP, FN, TN, yPredicted, yTarget):
    if yPredicted == 1:
        if yTarget == 1:
            TP += 1
        else:
            FP += 1
    else:
        if yTarget == 1:
            FN += 1
        else:
            TN += 1
    return TP, FP, FN, TN

'''
Gönderilen ağırlık ve bias değerleri kullanılarak gönderilen veri setindeki her
bir veri için tahmin yapılır. Yapılan tahminlere göre toplam TP, FP, FN, TN bulunur
ve döndürülür.
'''
def getMetricVariables(weights, bias, data_set):
    threshold = 0.5
    yPredicted = 0
    TP, FP, FN, TN = 0, 0, 0, 0;
    for data in data_set:
        yTarget = data[2]
        result = calculateYPredicted(weights, data[:2], bias)
        if result > threshold:
            yPredicted = 1
        else:
            yPredicted = 0
        TP, FP, FN, TN = addToMetricVariables(TP, FP, FN, TN, yPredicted, yTarget)
    return TP, FP, FN, TN

'''
Gönderilen TP, FP, FN, TN değerlerine göre metrikleri hesaplar, konsola yazdırır
'''
def printMetrics(TP, FP, FN, TN):
    print("TP:", TP, "FP:", FP, "FN:", FN, "TN:", TN)
    accuracy, precision, recall, f1Score = getMetrics(TP, FP, FN, TN)
    print("Accuracy:", accuracy, "Precision:", precision, "Recall:", recall, "F1 Score:", f1Score)

'''
Gönderilen TP, FP, FN, TN değerlerine göre accuracy, precision, recall, f1 score
değerlerini bulup döndürür.
'''
def getMetrics(TP, FP, FN, TN):
    accuracy = getAccuracy(TP, FP, FN, TN)
    precision = getPrecision(TP, FP, FN, TN)
    recall = getRecall(TP, FP, FN, TN)
    f1Score = getF1Score(precision, recall)
    return accuracy, precision, recall, f1Score

def getAccuracy(TP, FP, FN, TN):
    return (TP + TN) / (TP + FP + FN + TN)

def getPrecision(TP, FP, FN, TN):
    return TP / (TP + FP)

def getRecall(TP, FP, FN, TN):
    return TP / (TP + FN)

def getF1Score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))
