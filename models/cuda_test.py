import torch

from common.device import detect_device, get_device_priority

print("torch.__version__ =", torch.__version__)
print("preferred_device =", detect_device())
print("device_priority =", " -> ".join(get_device_priority()))
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count =", torch.cuda.device_count())
    print("device_name =", torch.cuda.get_device_name(0))
print("mps_available =", getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())
