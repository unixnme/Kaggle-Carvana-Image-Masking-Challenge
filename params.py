from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_2048
from model.vgg16 import vgg16

rows = 2048
cols = 2048

max_epochs = 100
batch_size = 2

orig_width = 1918
orig_height = 1280

threshold = 0.5

model_factory = get_unet_2048
