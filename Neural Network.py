from skimage import io
import keras
import os
from matplotlib import pyplot as plt
import numpy
import math
import time
import datetime

os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\\bin'

def readdata(path): #beolvassuk a célmappából a képeket egy tömbbe amivel vissza fogunk térni

    data = []
    for img in os.listdir(path):
        data.append(numpy.array(io.imread(os.path.join(path, img)))) #hozzáadjuk a "laposan" beolvasott képet a tömbbe
    return data

#az eredményeket kiírjuk egy új mappába az adott helyen. A mappáknek a neve a dátum lesz.
def output_results(path, epochs, batch_size, history, training_time, model):
    now = datetime.datetime.now().strftime("%Y.%m.%d. %Hh%Mm%Ss.")
    path = os.path.join(path,now)
    os.makedirs(path)
    os.makedirs(os.path.join(path,"input"))
    os.makedirs(os.path.join(path,"predicted"))
    os.makedirs(os.path.join(path,"expected"))


    f = open(os.path.join(path,"statistics.txt"), "w+")
    f.write("Epochs: " + str(epochs) + "\n")
    f.write("Batch size: " + str(batch_size) + "\n")
    f.write("Loss: %.4f" % history.history['loss'][-1] + "\n")  # ezzel írjuk ki az utolsó losst
    f.write("PSNRLoss: %.4f" % history.history['PSNRLoss'][-1] + "\n")  # ezzel írjuk ki az utolsó psnrlosst
    f.write("Training time: " + str(datetime.timedelta(seconds=round(training_time))) + "\n")

    plt.plot(history.history['loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(path, "Loss.png"))

    plt.clf()

    plt.plot(history.history['PSNRLoss'])
    plt.title('PSNR loss')
    plt.ylabel('PSNR Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(path, "PSNRLoss.png"))

    plt.clf()

    keras.utils.plot_model(model, os.path.join(path, "model.png"))

    raw_test = readdata("D:\SULI\Önálló laboratórium (F)\Képek\Tesztek")
    correct_prediction = readdata("D:\SULI\Önálló laboratórium (F)\Képek\Expected")

    for i in range(len(raw_test)):
        plt.imshow(raw_test[i])
        plt.savefig(os.path.join(path,"input\\" + str(i) + ".jpg"))
        plt.imshow(correct_prediction[i])
        plt.savefig(os.path.join(path,"expected\\" + str(i) + ".jpg"))

    test_data = numpy.array(raw_test)

    predict = model.predict(test_data, batch_size=1)
    for i in range(len(predict)):
        predict[i] = predict[i] / 255
        plt.imshow(predict[i])
        plt.savefig(os.path.join(path, "predicted\\" + str(i) + ".jpg"))


def showimage(image): #megmutatja a kért képet
    io.imshow(image)
    plt.show()

def PSNRLoss(guess,truth):
    return 20*math.log(255,10) - 10*keras.backend.log(keras.backend.mean(keras.backend.square(guess-truth))) / keras.backend.log(10.)

def showloss(history):
    plt.plot(history.history['loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def showPSNRloss(history):
    plt.plot(history.history['PSNRLoss'])
    plt.title('PSNR loss')
    plt.ylabel('PSNR Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

###################MAIN xd######################

lowres_images = readdata("D:\SULI\Önálló laboratórium (F)\Képek\Low res") #sima tömbble olvassuk a képeket
higres_images = readdata("D:\SULI\Önálló laboratórium (F)\Képek\High res")

training_data = numpy.array(lowres_images) #numpy arraybe pakoljuk a képeket
expected_output = numpy.array(higres_images)

shape = numpy.shape(training_data)
shape = shape[1:]

print(str(len(lowres_images)) + " images")


#a modelnek generálnia kéne valami trágya képet és azt össze kéne hasonlítani a másikkal ezzel a PSNR segítségével és az alapján tweak
#na jelenlegi állapotában dekonvolvál, tehát nagyobbá csinálja a cuccot meg ilyesmi és amúgy egész nagy lesz a psnr matyi
#de a sima loss meg tök kevés, idk
model = keras.Sequential()
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu',input_shape=shape))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32,activation='relu'))
model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=3,activation='relu'))



model.compile(optimizer='adam', loss='mean_squared_error', metrics=[PSNRLoss])

epochs = 100
batch_size = 1

training_time = time.time() #lemérjük, hogy mennyi ideig tréningelt

history = model.fit(training_data, expected_output, epochs=epochs, batch_size=batch_size)

training_time = time.time()-training_time

#elvileg ezzel trénelve lett a model

output_results("D:\SULI\Önálló laboratórium (F)\Eredmények",epochs,batch_size,history,training_time, model)


#ezzel pedig tesztelve is lett az alapja kindof