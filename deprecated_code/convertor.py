import os
os.environ['KERAS_BACKEND'] = 'torch'
from keras_torch_transport import TorchToKerasConvertor


convertor = TorchToKerasConvertor("./RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth")

convertor.convert2kears("output.keras")