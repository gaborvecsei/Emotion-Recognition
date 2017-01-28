from keras.callbacks import Callback
import time


class LogTraining(Callback):

	def __init__(self, filePath):
		self.filePath = filePath

    def on_train_begin(self, logs={}):
        self.startTime = time.clock()
        self.endTime = None
        
    def on_epoch_end(self, epoch, logs={}):
        text = "Epoch: {0}; Loss: {1}, Accuracy: {2}".format(epoch, logs.get('loss'), logs.get('acc'))
        self.appendTextToFile(text)

    def on_train_end(self, logs={})
    	self.endTime = time.clock()
    	text = "Trained in: {0} seconds".format(self.endTime-self.startTime)
    	self.appendTextToFile(text)

    def appendTextToFile(text):
    	with open(self.filePath, "a") as f:
    		f.write(text + "\n")