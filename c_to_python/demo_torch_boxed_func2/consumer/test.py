import torch
import producer
import consumer

producer_so = "/home/haoyuangao/pinned/cuda_learing/torchclass_demo/producer/producer.cpython-311-x86_64-linux-gnu.so"

x = torch.randn(4).requires_grad_(True)
w = torch.randn(4).requires_grad_(True)

y = consumer.infer_by_producer_so(producer_so, 7, x, w)
print("y =", y)
print("x =", x)
print("w =", w)
print("x.grad =", x.grad)
print("w.grad =", w.grad)

print("real out:", x*w + 7)

st = torch.ops.producer.make_state(7)
#### infer by st
y_st = torch.ops.consumer.infer_st(x, w, st)
print("real out by st:", y_st)

#### infer by py
y_py = consumer.infer_py(x, w, st)
print("real out by py:", y_py)
