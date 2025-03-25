
import torch



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

print("\n======================= 创建从 0 到 9 的张量\n")

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)
print("\n")






'''
有时您可能需要某种类型的一个张量与另一个张量具有相同的形状。
'''


print("\n======================= 根据一个张量的形状创建另一个张量\n")
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
print(ten_zeros)
print("\n")





# 数据类型默认是 float32

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 
print("\n======================= 数据类型默认是 float32 \n")
print((float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device))
print("\n")





# 也可以人为修改成 float16


float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

print("\n======================= 也可以人为修改成 float16 \n")
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



