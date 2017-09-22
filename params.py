from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024
from model.densenet import dense_net_128

rows = 960
cols = 640

max_epochs = 100
batch_size = 8

orig_width = 1918
orig_height = 1280

threshold = 0.5

model_factory = get_unet_256
