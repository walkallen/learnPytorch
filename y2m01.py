
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











