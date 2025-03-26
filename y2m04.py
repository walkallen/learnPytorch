

import torch


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


print("\n======================= 增加一个维度 \n")
# Add an extra dimension
x_reshaped = x.reshape(1, 7)
print(x_reshaped)
print(x_reshaped.shape)



print("\n======================= 使用torch.view()更改张量的视图 \n")


"""
使用torch.view()更改张量的视图，实际上只会创建同一张量的新视图。

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






print("\n======================= 如果我们想将新张量堆叠五次，我们可以使用torch.stack()来实现 \n")

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

print("\n======================= 使用 torch.squeeze() 删除多余的维度 \n")


print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")
print("\n")





# 要执行与torch.squeeze()相反的操作，您可以使用torch.unsqueeze()在特定索引处添加维度值 1。
print("\n======================= 要执行与torch.squeeze()相反的操作，您可以使用torch.unsqueeze()在特定索引处添加维度值 1 \n")

print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

## Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")
print("\n")





# 您还可以使用torch.permute(input, dims)重新排列轴值的顺序，其中input将转换为具有新dims视图。

print("\n======================= 您还可以使用torch.permute(input, dims)重新排列轴值的顺序, 其中input 将转换为具有新 dims 视图 \n")
# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")







