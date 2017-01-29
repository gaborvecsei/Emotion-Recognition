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

    def on_train_begin(self, logs={}):
        self.startTime = time.clock()
        self.endTime = None

    def on_epoch_end(self, epoch, logs={}):
        text = "Epoch: {0}; Loss: {1}, Accuracy: {2}".format(epoch, logs.get('loss'), logs.get('acc'))
        self.appendTextToFile(text)

    def on_train_end(self, logs={}):
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

    def on_train_end(self, logs={}):
        super().on_train_end(logs)
        self.sendNotification(
            "Training ended! Check out the results: loss:{0:.2f}, acc:{1:.2f}".format(logs.get('loss'),
                                                                                      logs.get('acc')))

    def sendNotification(self, message):
        self.slackClient.api_call('chat.postMessage', text=message, channel=self.channelName, username=self.botName,
                                  icon_emoji=":robot_face:")
