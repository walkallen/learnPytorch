
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


