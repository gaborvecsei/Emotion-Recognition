import time

from keras.callbacks import Callback
from slackclient import SlackClient
import configparser


class LogTraining(Callback):
    """
    Logs the training at each epoch to a .txt file
    (loss and accuracy)
    """

    def __init__(self, file_path):
        """
        :param file_path: Path to the file
        """

        super().__init__()
        self.file_path = file_path
        self.start_time = None
        self.end_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.clock()
        self.end_time = None

    def on_epoch_end(self, epoch, logs=None):
        text = "Epoch: {0}; Loss: {1}, Accuracy: {2}".format(epoch, logs.get('loss'), logs.get('acc'))
        self.append_text_to_file(text)

    def on_train_end(self, logs=None):
        self.end_time = time.clock()
        text = "Trained in: {0} seconds".format(self.end_time - self.start_time)
        self.append_text_to_file(text)

    def append_text_to_file(self, text):
        with open(self.file_path, "a") as f:
            f.write(text + "\n")


class SlackNotifier(Callback):
    """
    Sends you a message at Slack when the training is finished
    """

    def __init__(self, bot_name="Notifier Bot", channel_name="Training Notification"):
        """
        :param bot_name: Name of your bot (can be anything)
        :param channel_name: Name of an existing channel at your slack team
        """

        super().__init__()
        self.bot_name = bot_name
        self.channel_name = channel_name
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.slack_token = config.get("slack", "token")
        self.slack_client = SlackClient(self.slack_token)

        self.accs = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.accs.append(logs.get('acc'))
        self.losses.append(logs.get('loss'))

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.send_notification(
            "Training ended! :smile: \n" + "Accuracy: {0}\nLoss: {1}".format(self.accs[-1], self.losses[-1]))

    def send_notification(self, message):
        self.slack_client.api_call('chat.postMessage', text=message, channel=self.channel_name, username=self.bot_name,
                                   icon_emoji=":robot_face:")
