from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024

rows = 1024
cols = 1024 / 2 * 3

max_epochs = 100
batch_size = 2

orig_width = 1918
orig_height = 1280

threshold = 0.5

model_factory = get_unet_1024
