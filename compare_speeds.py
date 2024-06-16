# import time
# import numpy as np
# import torch

# def measure_time(func, *args, repeats=5):
#     times = []
#     for _ in range(repeats):
#         start = time.time()
#         func(*args)
#         end = time.time()
#         times.append(end - start)
#     return np.mean(times)

# # Python's built-in pow()
# def python_pow(base, exp):
#     return [pow(b, e) for b, e in zip(base, exp)]

# # NumPy's np.power()
# def numpy_pow(base, exp):
#     return np.power(base, exp)

# # PyTorch's torch.pow()
# def pytorch_pow(base, exp):
#     return torch.pow(base, exp)

# # Generate random base and exponent arrays
# size = 100000
# base = np.random.rand(size).tolist()
# exp = np.random.rand(size).tolist()

# # Convert base and exponent arrays to NumPy arrays
# base_np = np.array(base)
# exp_np = np.array(exp)

# # Convert base and exponent arrays to PyTorch tensors
# base_torch = torch.tensor(base_np, dtype=torch.float32)
# exp_torch = torch.tensor(exp_np, dtype=torch.float32)

# # Measure the time taken for each function
# python_time = measure_time(python_pow, base, exp)
# numpy_time = measure_time(numpy_pow, base_np, exp_np)
# pytorch_time = measure_time(pytorch_pow, base_torch, exp_torch)

# print(f"Python pow() average time: {python_time:.6f} seconds")
# print(f"NumPy np.power() average time: {numpy_time:.6f} seconds")
# print(f"PyTorch torch.pow() average time: {pytorch_time:.6f} seconds")


# import time
# import numpy as np
# import torch

# def measure_time(func, *args, repeats=5):
#     times = []
#     for _ in range(repeats):
#         start = time.time()
#         func(*args)
#         end = time.time()
#         times.append(end - start)
#     return np.mean(times)

# # Python's built-in pow()
# def python_pow(base, exp):
#     return pow(base, exp)

# # NumPy's np.power()
# def numpy_pow(base, exp):
#     return np.power(base, exp)

# # PyTorch's torch.pow()
# def pytorch_pow(base, exp):
#     return torch.pow(base, exp)

# # Single base and exponent values
# base = 10000
# exp = 512

# # Convert base and exponent to NumPy array
# base_np = np.array(base)
# exp_np = np.array(exp)

# # Convert base and exponent to PyTorch tensor
# base_torch = torch.tensor(base, dtype=torch.float32)
# exp_torch = torch.tensor(exp, dtype=torch.float32)

# # Measure the time taken for each function
# python_time = measure_time(python_pow, base, exp)
# numpy_time = measure_time(numpy_pow, base_np, exp_np)
# pytorch_time = measure_time(pytorch_pow, base_torch, exp_torch)

# print(f"Python pow() average time: {python_time:.6f} seconds")
# print(f"NumPy np.power() average time: {numpy_time:.6f} seconds")
# print(f"PyTorch torch.pow() average time: {pytorch_time:.6f} seconds")

import time
import numpy as np
import torch

def measure_time(func, *args, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end - start)
    return np.mean(times)

# Python's built-in pow()
def python_mult(a, b):
    return a @ b

# PyTorch's torch.pow()
def pytorch_mult(a, b):
    return torch.matmul(a, b)

# Single base and exponent values
hidden_size = 3584
seq_len = 1024
num_attention_heads = 28
a = torch.randn((1, num_attention_heads, seq_len, int(hidden_size/num_attention_heads)), dtype=torch.float32)
b = torch.randn((1, num_attention_heads, int(hidden_size/num_attention_heads), seq_len), dtype=torch.float32)
print(hidden_size/num_attention_heads)



# Measure the time taken for each function
python_time = measure_time(python_mult, a, b)
pytorch_time = measure_time(pytorch_mult, a, b)

print(f"Python pow() average time: {python_time:.6f} seconds")
print(f"PyTorch torch.pow() average time: {pytorch_time:.6f} seconds")

