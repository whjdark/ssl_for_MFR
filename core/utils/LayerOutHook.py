'''
Author: whj
Date: 2022-05-31 21:38:24
LastEditors: whj
LastEditTime: 2022-05-31 21:48:02
Description: file content
'''

class LayerOutHook():
    """
    Hook for LayerOut.
    """
    def __init__(self, net, layer_name):
        self.output = None

        for (name, module) in net.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook=self.hook)
    
    def hook(self, module, input, output):
        self.output = output
