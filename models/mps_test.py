import torch

from common.device import detect_device, get_device_priority

print("torch.__version__ =", torch.__version__)
print("preferred_device =", detect_device())
print("device_priority =", " -> ".join(get_device_priority()))
print("mps_available =", getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())
print("cuda_available =", torch.cuda.is_available())

