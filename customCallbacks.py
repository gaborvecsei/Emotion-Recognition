import time

from keras.callbacks import Callback
from slackclient import SlackClient


class LogTraining(Callback):
    """
    Logs the training at each epoch to a txt file
    """

    def __init__(self, filePath):
        """
        :param filePath: Path to the file
        """

        super().__init__()
        self.filePath = filePath
        self.startTime = None
        self.endTime = None

    def on_train_begin(self, logs=None):
        self.startTime = time.clock()
        self.endTime = None

    def on_epoch_end(self, epoch, logs=None):
        text = "Epoch: {0}; Loss: {1}, Accuracy: {2}".format(epoch, logs.get('loss'), logs.get('acc'))
        self.appendTextToFile(text)

    def on_train_end(self, logs=None):
        self.endTime = time.clock()
        text = "Trained in: {0} seconds".format(self.endTime - self.startTime)
        self.appendTextToFile(text)

    def appendTextToFile(self, text):
        with open(self.filePath, "a") as f:
            f.write(text + "\n")


class SlackNotifier(Callback):
    """
    Sends you message at Slack when the training is finished
    """

    def __init__(self, slackToken, botName="Notifier Bot", channelName="Training Notification"):
        """
        :param slackToken: Generate token for your slack team: https://api.slack.com/docs/oauth-test-tokens
        :param botName: Name of your bot (can be anything)
        :param channelName: Name of an existing channel at your slack team
        """

        super().__init__()
        self.botName = botName
        self.channelName = channelName
        self.slackToken = slackToken
        self.slackClient = SlackClient(self.slackToken)

        self.accs = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.accs.append(logs.get('acc'))
        self.losses.append(logs.get('loss'))

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.sendNotification(
            "Training ended! :smile: \n" + "Accuracy: {0}\nLoss: {1}".format(self.accs[-1], self.losses[-1]))

    def sendNotification(self, message):
        self.slackClient.api_call('chat.postMessage', text=message, channel=self.channelName, username=self.botName,
                                  icon_emoji=":robot_face:")
