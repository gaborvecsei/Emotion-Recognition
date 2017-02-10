import threading
import time

from keras.models import model_from_json


class ModelLoader(threading.Thread):
    """
    Loads trained model in a new thread
    """

    def __init__(self, modelStructurePath, modelWeightsPath):
        """
        :param modelStructurePath: path to the model's json file
        :param modelWeightsPath:  path to the model's weights (.h5) file
        """

        super(ModelLoader, self).__init__()
        self.model = None
        self.modelStructurePath = modelStructurePath
        self.modelWeightsPath = modelWeightsPath

    def getModel(self):
        if self.model is None:
            return None
        else:
            return self.model

    def loadModel(self):
        """
        Loads model from the model structure and model weights file
        :return: trained model
        """

        print("Model loading started...")
        s = time.clock()
        with open(self.modelStructurePath, "r") as jsonFile:
            loadedModelStructure = jsonFile.read()
        self.model = model_from_json(loadedModelStructure)
        self.model.load_weights(self.modelWeightsPath)
        e = time.clock()
        print("Model is Loaded: {0}; in {1:.2f} seconds".format(self.model, (e - s)))

    def run(self):
        super(ModelLoader, self).run()
        self.loadModel()
