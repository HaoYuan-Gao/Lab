import torch
import witin  # 触发加载扩展

x = torch.randn(3)
print("x =", x)

meta = {"a": 1, "b": 42, "hello": 7}

y = torch.ops.witin.foo(x, meta)
print("y =", y)


s = "hello_torchscript"
print("input :", s)

out = torch.ops.witin.echo_str(s)
print("output:", out)