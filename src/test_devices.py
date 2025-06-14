import torch
print("CUDA Disponível:", torch.cuda.is_available())
print("Versão PyTorch:", torch.__version__)
print("Versão CUDA:", torch.version.cuda)
print("Dispositivo disponível:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
