'''
Created on Dec 18, 2014

@author: Carlos Jaramillo
@author: Pablo Munoz
'''

from __future__ import division
from __future__ import print_function

import visvis as vv
import common_plot

class Visualization(object):
    def __init__(self):
        self.running_classifier = False
        print("Running visualization")
        app = vv.use()
        vv.title("Components intervals")
        main_window = vv.gca()
        main_window.position.w = 1000
        main_window.SetLimits((0, 1000), (1, 500), None, 0.02)
        main_window.eventKeyDown.Bind(self.OnKey)
        app.Run()

    def OnKey(self, event):
        """ Called when a key is pressed down in the axes.
        """
        if event.text and event.text.lower() in 'r' and not self.running_classifier:
            print("running classifier")
            self.running_classifier = True
            pass  # test_real_dataset()
