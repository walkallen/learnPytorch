import torch


# 基本操作

tensor = torch.tensor([1, 2, 3])
print(tensor + 10)

print(tensor * 10)

# 矩阵乘法
print("\n======================= 矩阵乘法 \n")
a = torch.matmul(tensor, tensor)
print(a)

# 矩阵乘法可以使用 @ 符号
b = tensor @ tensor 
print(b)
print("\n")





"""

神经网络充满了矩阵乘法和点积。

torch.nn.Linear()模块（稍后我们将看到它的实际应用）, 也称为前馈层或全连接层,
实现输入 x 和权重矩阵 A 之间的矩阵乘法。

Y = Ax + b

"""




print("\n======================= 创建一个线性层 \n")

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)


# 设定种子，固定初始参数
torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")
print("\n")





print("\n======================= 求最小值、最大值、平均值、总和等（聚合） \n")

# 求最小值、最大值、平均值、总和等（聚合）

# Create a tensor
x = torch.arange(0, 100, 10)
print(x)

print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
# 将整数类型转换为 float 类型
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")
print("\n")






print("\n======================= 求位置最小/最大 \n")

# Positional min/max 位置最小/最大

"""

您还可以分别使用 torch.argmax() 和 torch.argmin() 找到出现最大值或最小值的张量的索引。

如果您只想要最高 (或最低) 值的位置而不是实际值本身 (我们将在后面使用 softmax 激活
函数的部分中看到这一点），这会很有帮助。
"""

# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")
print("\n")




print("\n======================= 更改张量数据类型 \n")


# 更改张量数据类型

# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
print((tensor, tensor.dtype))


# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)

# Create a int8 tensor
tensor_int8 = tensor.type(torch.int8)
print(tensor_int8)



