try:
    from .gtensor_abi import Tensor, noise_forward
except ImportError:
    from gtensor_abi import Tensor, noise_forward


def main():
    x = Tensor.empty_2d(3, 4, dtype=0)
    print("input:", x)

    outputs = noise_forward(x, output_count=3)
    print("output count:", len(outputs))

    for i, tensor in enumerate(outputs):
        print(f"output[{i}]:", tensor)


if __name__ == "__main__":
    main()
