from skimage import transform, io
from matplotlib import pyplot as plt
import os

i = 1
smallside = 16
bigside = smallside * 2
#path = "D:\SULI\Önálló laboratórium (F)\Képek\DIV2K_train_LR_bicubic\X4\\"
#for img in os.listdir(path):
#    kep = io.imread(path + img)
#    io.imsave("D:\SULI\Önálló laboratórium (F)\Képek\Low res\\" + str(i) + ".jpg", transform.resize(kep,(smallside,smallside)))
#    io.imsave("D:\SULI\Önálló laboratórium (F)\Képek\High res\\" + str(i) + ".jpg", transform.resize(kep,(bigside,bigside)))
#    i = i+1

i = 1
path = "D:\SULI\Önálló laboratórium (F)\Képek\Tesztek eredeti\\"
for img in os.listdir(path):
    kep = io.imread(path + img)
    io.imsave("D:\SULI\Önálló laboratórium (F)\Képek\Tesztek\\" + str(i) + ".jpg",transform.resize(kep, (smallside, smallside)))
    io.imsave("D:\SULI\Önálló laboratórium (F)\Képek\Expected\\" + str(i) + ".jpg",transform.resize(kep, (bigside, bigside)))
    i=i+1
