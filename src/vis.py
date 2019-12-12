"""
Visualization using visdom.
"""
import visdom
import numpy as np
from datetime import datetime

class VisdomVisualizer:
    """
    Uses visdom to visualize some data from the training.
    """

    def __init__(self):
        self.visdom = visdom.Visdom()
        self.env = datetime.now().strftime(("exp %m/%d/%Y, %H:%M:%S"))
        self.visdom.fork_env("main", self.env)
        self.windows = dict()
        self.ticks = dict()

    def push_train_loss(self, value):
        self._push_line(value, name="Train loss")

    def push_test_loss(self, value):
        self._push_line(value, name="Test loss")

    def push_train_accuracy(self, value):
        self._push_line(value, name="Train accuracy")

    def push_test_accuracy(self, value):
        self._push_line(value, name="Test accuracy")

    def push_class_images(self, images, label):
        self._push_images(images, name="Samples detected as {}".format(label))

    def _push_images(self, images, name):
        if self.windows.get(name) is None:
            self.ticks[name] = 0
            self.windows[name] = self.visdom.images(images, \
                                                    opts=dict(title=name),\
                                                    env=self.env)
        else:
            self.ticks[name] += 1
            self.visdom.images(images, win=self.windows[name], env=self.env)
 

    def _push_line(self, value, name):
        if self.windows.get(name) is None:
            self.ticks[name] = 0
            self.windows[name] = self.visdom.line(X=np.array([0]), \
                                                  Y=np.array([0]), \
                                                  opts=dict(title=name),\
                                                  env=self.env)
        else:
            self.ticks[name] += 1
            x = np.array([self.ticks[name]])
            y = np.array([value])
            self.visdom.line(X=x, Y=y, win=self.windows[name], env=self.env, update='append')
    
    


