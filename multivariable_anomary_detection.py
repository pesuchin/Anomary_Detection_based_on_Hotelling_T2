#-------------------------------------------------------------------------------
# Name:        multivariable anomary detection
# Purpose:     anomary detection
#
# Author:      pesuchin
#
# Created:     12/04/2015
# Copyright:   (c) pesuchin 2015
#-------------------------------------------------------------------------------
import csv
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def readdata(path):
  dataReader = csv.reader(open(path, 'r'))
  data = [[row[1],row[2]] for row in dataReader]
  data = data[1:]
  return data

class multivariable_anomary_detection:
  def __init__(self,data):
    self.data = np.array(data)
    self.ave = self.means()
    self.Xc = 0.0
    self.cov = self.covariance_matrix()
    self.a = self.anomary_threshold()
    self.th = [scipy.stats.chi2.ppf(0.99,1)]

  def means(self):
    weight_average = sum([num[0] for num in self.data]) / len(self.data)
    height_average = sum([num[1] for num in self.data]) / len(self.data)
    return  np.array([weight_average,height_average])

  def covariance_matrix(self):
    self.Xc = self.data - np.array([self.ave for i in self.data])
    return np.dot(self.Xc.T, self.Xc) / np.shape(self.data)[0]

  def anomary_threshold(self):
    return np.sum(np.dot(self.Xc,np.linalg.inv(self.cov)) * self.Xc,axis=1)

  def main(self):
    x = np.arange(0,len(self.data))
    y = np.array(self.a)
    plt.plot(x,y,"o")
    plt.title("Anomary_threshold")
    plt.hlines(self.th, 0, len(self.data), linestyles="dashed")
    #x = np.array([i[0] for i in self.data])
    #y = np.array([i[1] for i in self.data])
    #plt.plot(x,y,"o")
    plt.show()

if __name__ == '__main__':
  data = readdata("Davis.csv")
  float_data = [[float(j) for j in i] for i in data]
  anomary = multivariable_anomary_detection(float_data)
  anomary.main()
  #print anomary