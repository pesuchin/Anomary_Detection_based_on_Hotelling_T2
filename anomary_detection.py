#-------------------------------------------------------------------------------
# Name:        anormary detection based on a variable
# Purpose:     anomary detection
#
# Author:      pesuchin
#
# Created:     11/04/2015
#-------------------------------------------------------------------------------
import csv
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def read_data(path):
  dataReader = csv.reader(open(path, 'r'))
  data = [row[1] for row in dataReader]
  data = data[1:]
  return data

class one_variable_anomary_detection:
  def __init__(self, data):
    self.data = data
    self.mu = 0
    self.s1 = 0
    self.s2 = 0
    self.a = []
    self.th = [scipy.stats.chi2.ppf(0.99,1)]

  def average(self,data):
    sum = 0.0
    for d in data:
      sum += d
    return sum / len(data)

  def variance(self,ave):
    s1 = [((d - self.mu)**2) for d in self.data]
    return self.average(s1),s1

  def anomary_threshold(self):
    return [i / self.s2 for i in self.s1]

  def main(self):
    self.mu = self.average(self.data)
    self.s2 ,self.s1 = self.variance(self.mu)
    self.a = self.anomary_threshold()
    x = np.arange(0,len(self.data))
    y = np.array(self.a)
    plt.plot(x,y,"o")
    plt.hlines(self.th, 0, len(self.data), linestyles="dashed")
    plt.title("Anomary_threshold")
    plt.ylim(0,50)
    plt.show()

if __name__ == '__main__':
    data = read_data('Davis.csv')
    float_data = [float(i) for i in data]
    anomary =  one_variable_anomary_detection(float_data)
    anomary.main()
