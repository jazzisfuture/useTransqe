# -*- coding:utf-8 -*-
import sys


def exactHTER(inputFile, outputFile):
    """从使用TERCOM工具生成的**.ter文本中提取最终所需的HTER值，保留小数点后6位,
    并将HTER的数值范围限定到[0,1]
    **.ter内容如下：
    Hypothesis File: ./cwmt/dev_tokenize_label.tgt
    Reference File: ./cwmt/dev_tokenize_label.pe
    0:1 1.0 82.0 0.012195121951219513
    1:1 21.0 50.0 0.42
    2:1 0.0 41.0 0.0
    3:1 2.0 39.0 0.05128205128205128
    """
    tran = open(inputFile, 'rb')
    new_tran = open(outputFile, 'wb')

    lines = tran.readlines()
    for line in lines[2:]:
        value=line.strip().split(' ')
        if float(value[3])>1.0:
            print >> new_tran, '%.6f' % (1)
        else:
            print >> new_tran, '%.6f'%(float(value[3]))


if __name__ == '__main__':
    exactHTER(sys.argv[1], sys.argv[2])
