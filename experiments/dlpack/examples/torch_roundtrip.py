import torch
from torch.utils import dlpack
from gtensor import GTensor, memory_stats

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
g = GTensor.from_dlpack(dlpack.to_dlpack(x))
print(g)

y = dlpack.from_dlpack(g.to_dlpack())
print(y)
y[0, 0] = 999
print("shared with torch input:", x[0, 0].item())

own = GTensor([1024, 1024], dtype="float32")
print(memory_stats().current_bytes, memory_stats().peak_bytes)
