import torch
import dispatcher

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
c = 3.0

out = torch.ops.myops.scale_mul(a, b, c)
out.sum().backward()

print("max|grad_a| =", (a.grad), "  b*c =", b.detach() * c)
print("max|grad_b| =", (b.grad), "  a*c =", a.detach() * c)
