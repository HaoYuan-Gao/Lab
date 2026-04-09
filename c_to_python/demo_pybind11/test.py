import numpy as np
import demo_pybind11

a = np.ones(8, dtype=np.float32)
b = np.ones(8, dtype=np.float32) * 2

out = demo_pybind11.add(a, b)
print(out[:7])  # [3. 3. 3. 3. 3.]
