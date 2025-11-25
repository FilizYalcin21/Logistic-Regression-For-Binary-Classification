"""
@author: Filiz Yalcin
"""

from dataset import getDataSet
from plotUtilities import drawData, drawCrossEntropyLoss, drawHyperplane, drawConfusionMatrix
from train import train, getRandomWeights
from metricUtilities import getMetricVariables, printMetrics

data_set = getDataSet()

# Tüm veriler grafik üzerinde çizdirilir
drawData(data_set)

train_dataset = data_set[:60]
validation_dataset = data_set[60:80]
test_dataset = data_set[80:]

epoch = 50 # 100 epok ile denenmiş, 50'de durulması gerektiğine karar verilmiştir
learningRate = 0.8
weights = getRandomWeights(3)
# weights'in 1 ve 2. değerleri ağırlık, 3. değeri bias
bias = weights[2]
weights = weights[:2]

''' 1 epok ile deneme
weightsWith1Epoch, biasWith1Epoch, averageLossListWith1Epoch = train(1, train_dataset, weights, bias, learningRate)
drawCrossEntropyLoss(averageLossListWith1Epoch)
drawHyperplane(data_set, weightsWith1Epoch, biasWith1Epoch)
'''

weights, bias, averageLossList, val_averageLossList = train(epoch, train_dataset, weights, bias, learningRate, validation_dataset)
drawCrossEntropyLoss(averageLossList)
drawCrossEntropyLoss(val_averageLossList)
drawHyperplane(data_set, weights, bias)

print("\nTrain Metrics:")
TP_train, FP_train, FN_train, TN_train = getMetricVariables(weights, bias, train_dataset)
printMetrics(TP_train, FP_train, FN_train, TN_train)

print("\nValidation Metrics:")
TP_val, FP_val, FN_val, TN_val = getMetricVariables(weights, bias, validation_dataset)
printMetrics(TP_val, FP_val, FN_val, TN_val)

print("\nTest Metrics:")
TP_test, FP_test, FN_test, TN_test = getMetricVariables(weights, bias, test_dataset)
printMetrics(TP_test, FP_test, FN_test, TN_test)

drawConfusionMatrix(TP_train, FN_train, FP_train, TN_train)
drawConfusionMatrix(TP_val, FN_val, FP_val, TN_val)
drawConfusionMatrix(TP_test, FN_test, FP_test, TN_test)
