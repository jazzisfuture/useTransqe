# -*- coding:utf-8 -*-
import sys


def addlabel(inputFile, outputFile):
    tran = open(inputFile, 'rb')
    new_tran = open(outputFile, 'wb')

    lines = tran.readlines()
    index = 0
    for line in lines:
        print >> new_tran, line.strip() + ' ' + '(' + str(index) + ')'
        index+=1

if __name__ == '__main__':
    addlabel(sys.argv[1], sys.argv[2])
