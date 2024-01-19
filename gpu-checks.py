print('## tf check')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
gpu = tf.config.list_physical_devices('GPU')
print(gpu)

print('## jax check')
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

print('## pytorch check')
import torch
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))