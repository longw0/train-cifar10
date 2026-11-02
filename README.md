# Quick Start

create a conda environment
```shell
conda create -n stl10_gpu python=3.10 -y
conda activate stl10_gpu
```

install pytorch (cuda version)
> My Device
> 
> CPU: i7-11700 2.5GHZ
> 
> GPU: GTX 1660Ti 
> 
> Mem: 2×16GB DDR4 3200 
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib numpy pillow tqdm torchinfo torchviz graphviz
```

verify
```python
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

