
import torch


print(f"torch.__version__ {torch.__version__}")


# 向量
vector = torch.tensor([7, 7])


print(f"vector          {vector}")

# 有几个方括号，维数就是几

print(f"vector.ndim     {vector.ndim}")




# 矩阵  二维矩阵，维数是2
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])



print(f"\nMATRIX\n{MATRIX}\n\n")

print(f"MATRIX.ndim     {MATRIX.ndim}")


print(f"MATRIX.shape    {MATRIX.shape}")



# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])


print(f"\nTENSOR\n{TENSOR}\n\n")


print(f"TENSOR.ndim     {TENSOR.ndim}")
print(f"TENSOR.shape    {TENSOR.shape}")

"""

=========================== Output ===================================

torch.__version__ 2.4.0+cu121
vector          tensor([7, 7])
vector.ndim     1

MATRIX
tensor([[ 7,  8],
        [ 9, 10]])


MATRIX.ndim     2
MATRIX.shape    torch.Size([2, 2])

TENSOR
tensor([[[1, 2, 3],
         [3, 6, 9],
         [2, 4, 5]]])


TENSOR.ndim     3
TENSOR.shape    torch.Size([1, 3, 3])

"""



# rand 是平均分布生成 0 到 1 之间的数字
random_tensor = torch.rand(size=(3, 4))

print((random_tensor, random_tensor.dtype))
print()
print()

# 假设您想要一个常见图像形状为[224, 224, 3] （ [height, width, color_channels ]）的随机张量。

random_image_size_tensor = torch.rand(size=(224, 224, 3))

print(f"random_image_size_tensor.shape    {random_image_size_tensor.shape}")
print(f"random_image_size_tensor.ndim     {random_image_size_tensor.ndim}")
print("\n")

# 创建一个全都是零的 tensor
zeros = torch.zeros(size=(3, 4))
print((zeros, zeros.dtype))
print("\n")

# 创建一个全都是 1 的 tensor
ones = torch.ones(size=(3, 4))
print((ones, ones.dtype))
print("\n")


"""
=========================== Output ===================================


random_image_size_tensor.shape    torch.Size([224, 224, 3])
random_image_size_tensor.ndim     3


(tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]]), torch.float32)


(tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]]), torch.float32)


"""


'''

Sometimes you might want a range of numbers, such as 1 to 10 or 0 to 100.
有时您可能需要一个数字范围，例如 1 到 10 或 0 到 100。

You can use torch.arange(start, end, step) to do so.
您可以使用torch.arange(start, end, step)来执行此操作。


Where: 在哪里：

start = start of range (e.g. 0)
start = 范围的开始（例如 0
end = end of range (e.g. 10)
end = 范围结束（例如 10
step = how many steps in between each value (e.g. 1)
step = 每个值之间有多少步（例如 1

注意: 在Python中, 您可以使用range()来创建范围。然而在 PyTorch 中， 
torch.range()已被弃用，并且将来可能会显示错误。

'''


# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)
print("\n")



'''
有时您可能需要某种类型的一个张量与另一个张量具有相同的形状。
'''

ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
print(ten_zeros)
print("\n")



# 数据类型默认是 float32

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 
print("数据类型默认是 float32")
print((float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device))
print("\n")


# 也可以人为修改成 float16


float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

print("也可以人为修改成 float16")
print(float_16_tensor.dtype)
print("\n")



'''
我们之前已经见过这些，但是您想要了解的有关张量的三个最常见属性是：


shape - 张量是什么形状？ （有些操作需要特定的形状规则）

dtype - 张量中的元素存储在什么数据类型中？

device - 张量存储在什么设备上？  通常是GPU或CPU
'''

# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"张量的形状: {some_tensor.shape}")
print(f"张量的数据类型: {some_tensor.dtype}")
print(f"张量存储的设备: {some_tensor.device}") # will default to CPU
print("\n")


"""
=========================== Output ===================================

tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


数据类型默认是 float32
(torch.Size([3]), torch.float32, device(type='cpu'))


也可以人为修改成 float16
torch.float16


tensor([[0.9545, 0.7235, 0.3124, 0.5020],
        [0.1236, 0.9087, 0.5158, 0.1131],
        [0.9602, 0.9327, 0.4993, 0.5249]])
张量的形状: torch.Size([3, 4])
张量的数据类型: torch.float32
张量存储的设备: cpu

"""


# 基本操作

tensor = torch.tensor([1, 2, 3])
print(tensor + 10)

print(tensor * 10)

# 矩阵乘法
a = torch.matmul(tensor, tensor)
print(a)

# 矩阵乘法可以使用 @ 符号
b = tensor @ tensor 
print(b)
print("\n")


"""
=========================== Output ===================================
tensor([11, 12, 13])
tensor([10, 20, 30])
tensor(14)
tensor(14)

=========================== Output ===================================
"""



"""

神经网络充满了矩阵乘法和点积。

torch.nn.Linear()模块（稍后我们将看到它的实际应用）, 也称为前馈层或全连接层,
实现输入 x 和权重矩阵 A 之间的矩阵乘法。

Y = Ax + b

"""



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



"""
=========================== Output ===================================

Input shape: torch.Size([3, 2])

Output:
tensor([[2.2368, 1.2292, 0.4714, 0.3864, 0.1309, 0.9838],
        [4.4919, 2.1970, 0.4469, 0.5285, 0.3401, 2.4777],
        [6.7469, 3.1648, 0.4224, 0.6705, 0.5493, 3.9716]],
       grad_fn=<AddmmBackward0>)

Output shape: torch.Size([3, 6])


tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
Minimum: 0
Maximum: 90
Mean: 45.0
Sum: 450


Tensor: tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
Index where max value occurs: 8
Index where min value occurs: 0


=========================== Output ===================================
"""




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



"""
重塑、堆叠、挤压和解压
Often times you'll want to reshape or change the dimensions of your tensors without actually changing the values inside them.
很多时候，您需要重塑或更改张量的尺寸，而不实际更改其中的值。


torch.reshape(input, shape)         input重塑为shape 如果兼容  也可以使用torch.Tensor.reshape() 。
Tensor.view(shape)                  返回原始张量的不同shape的视图  但与原始张量共享相同的数据。
torch.stack(tensors, dim=0)         沿新维度 ( dim ) 连接一系列tensors  所有 tensors 必须具有相同的大小。
torch.squeeze(input)                挤压 input 以删除所有值为 1 维度。
torch.unsqueeze(input, dim)         返回在dim处添加维度值为1 input 。
torch.permute(input, dims)          返回原始input的视图 其尺寸排列 重新排列 为dims 。
"""


x = torch.arange(1., 8.)
print(x)
print(x.shape)


# Add an extra dimension
x_reshaped = x.reshape(1, 7)
print(x_reshaped)
print(x_reshaped.shape)



"""
使用torch.view()更改张量的视图实际上只会创建同一张量的新视图。

So changing the view changes the original tensor too.
因此改变视图也会改变原始张量。

"""
z = x.view(1, 7)
print(z)
print(z.shape)


# Changing z changes x
z[:, 0] = 5
print(f"z is {z}")
print(f"x is {x}")


# 如果我们想将新张量堆叠五次，我们可以使用torch.stack()来实现。

# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
print("在第 0 维堆叠")
print(x_stacked)

print()
print(f"堆叠以后的形状 {x_stacked.shape}")
print("\n")

x_stacked = torch.stack([x, x, x, x], dim=1) # try changing dim to dim=1 and see what happens
print("在第 1 维堆叠")
print(x_stacked)

print()
print(f"堆叠以后的形状 {x_stacked.shape}")
print("\n")


"""
从张量中删除所有单一维度怎么样？

To do so you can use torch.squeeze() (I remember this as squeezing the tensor to only have dimensions over 1).
为此，您可以使用torch.squeeze() （我记得这是将张量压缩为仅具有超过 1 的维度）。

"""

print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")
print("\n")


# 要执行与torch.squeeze()相反的操作，您可以使用torch.unsqueeze()在特定索引处添加维度值 1。
print("要执行与torch.squeeze()相反的操作，您可以使用torch.unsqueeze()在特定索引处添加维度值 1")

print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

## Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")
print("\n")


# 您还可以使用torch.permute(input, dims)重新排列轴值的顺序，其中input将转换为具有新dims视图。

print("您还可以使用torch.permute(input, dims)重新排列轴值的顺序，其中input将转换为具有新dims视图")
# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")


