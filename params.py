from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_2048
from model.vgg16 import vgg16

rows = 1024
cols = 1024

max_epochs = 100
batch_size = 1

orig_width = 1918
orig_height = 1280

threshold = 0.5

learning_rate = 1e-2
half_life = 16.

model_factory = get_unet_1024
