

'''

二元分类
    目标可以是两个选项之一,例如是或否
    eg. 根据某人的健康参数预测其是否患有心脏病。

多类分类
    目标可以是两个以上选项之一
    eg. 确定一张照片是食物、人还是狗。


多标签分类
    可以为目标分配多个选项
    eg. 预测应该为维基百科文章分配哪些类别(例如数学、科学和哲学）。


我们要介绍的内容

0. 分类神经网络的架构
    神经网络几乎可以有任何形状或大小,但它们通常遵循类似的平面图。

1. 准备好二分类数据
    数据几乎可以是任何东西,但首先我们将创建一个简单的二元分类数据集。

2. 构建PyTorch分类模型
    在这里,我们将创建一个模型来学习数据中的模式,我们还将选择损失函数、优化器并构建特定于分类的训练循环。

3. 将模型拟合到数据(训练）
    我们已经有了数据和模型,现在让我们让模型(尝试）在(训练）数据中查找模式。

4. 进行预测并评估模型(推理）
    我们的模型在数据中发现了模式,让我们将其发现与实际(测试）数据进行比较。

5. 改进模型(从模型角度）
    我们已经训练并评估了一个模型,但它不起作用,让我们尝试一些方法来改进它。

6. 非线性
    到目前为止,我们的模型只能模拟直线,那么非线性(非直线）线呢？

7. 复制非线性函数
    我们使用非线性函数来帮助对非线性数据进行建模,但它们是什么样子的呢？

8. 将所有内容与多类分类结合起来
    让我们将迄今为止为二元分类所做的一切与多类分类问题放在一起。

'''



'''
0. 分类神经网络的架构

输入层形状
    与特征数量相同(例如,心脏病预测中的年龄、性别、身高、体重、吸烟状况为 5)

隐藏层
    特定于问题,最小值 = 1, 最大值 = 无限制


每个隐藏层的神经元
    具体问题,一般为 10 到 512

输出层形状 ( out_features )
    1 (一类或另一类)


隐藏层激活
    通常是ReLU (修正线性单元）,但也可以是许多其他单元


输出激活
    Sigmoid (PyTorch 中的torch.sigmoid )

损失函数
    二元交叉熵 (PyTorch 中的torch.nn.BCELoss )

优化器
    SGD (随机梯度下降）, Adam (有关更多选项, 请参阅torch.optim )



'''







'''
1. 制作分类数据并准备好

我们将使用 Scikit-Learn 中的 make_circles() 方法生成两个具有不同颜色点的圆圈。


'''

from sklearn.datasets import make_circles

'''
sklearn.datasets.make_circles 是 scikit-learn 库中一个用于生成合成数据集的函数，
特别用于二分类问题。它会创建两组数据点，这些数据点在二维平面上形成两个同心圆的形状，
一组点位于内圆上或附近，另一组点位于外圆上或附近。
'''

print("\n======================= 制作分类数据并准备好 \n")


# Make 1000 samples 
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values

print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")


# 可视化

# Make DataFrame of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})

print(
circles.head(10)
)


# 每个类有多少个值？


print("\n======================= 每个类有多少个值 \n")

# Check different labels
print( circles.label.value_counts() )



# 可视化
# Visualize with a plot
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0], 
            y=X[:, 1], 
            c=y, 
            cmap=plt.cm.RdYlBu);

plt.show()



print("\n======================= X 和 y 的形状, X 是二维坐标点, y 是标签 \n")

# Check the shapes of our features and labels
print( X.shape, y.shape )



print("\n======================= 这表明我们有两个输入,一个输出 \n")

# 这表明我们有两个输入,一个输出

# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")




# 1.2 将数据转换为张量并创建训练和测试分割

'''
将我们的数据转换为张量(现在我们的数据位于 NumPy 数组中,而 PyTorch 更喜欢使用 PyTorch 张量）。

将我们的数据分为训练集和测试集(我们将在训练集上训练模型以学习X和y之间的模式,然后在测试数据集上评估这些学习到的模式）。

'''


print("\n======================= 将数据转换为张量并创建训练和测试分割 \n")

import torch

# 读取 numpy，创建 tensor ，再转换为 float32 格式

X = torch.from_numpy(X).type(torch.float) 
y = torch.from_numpy(y).type(torch.float)

# 查看前五个数据

print(X[:5], y[:5])




'''
现在我们的数据是张量格式,让我们将其分为训练集和测试集。
为此,我们使用 Scikit-Learn 中有用的函数train_test_split() 。

我们将使用test_size=0.2 (80% 训练,20% 测试）,
并且由于分割在数据中随机发生,因此我们使用random_state=42 ,以便分割是可重现的。
'''

# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible

print(X_train.shape)

print(y_train.shape)

print( len(X_train), len(X_test), len(y_train), len(y_test)  )





'''
好的！看起来我们现在有 800 个训练样本和 200 个测试样本。

第二步. 建立模型

1. 设置与设备无关的代码(因此我们的模型可以在 CPU 或 GPU 上运行(如果可用）。
2. 通过子类化 nn.Module 构建模型。
3. 定义损失函数和优化器。
4. 创建训练循环(这将在下一节中介绍）


'''



print("\n======================= 建立模型 \n")

# Standard PyTorch imports
import torch
from torch import nn

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)



'''

我们需要一个能够处理X数据作为输入并生成y数据形式的模型作为输出的模型。

换句话说, 给定X 特征 , 我们希望模型能够预测y (标签）。

这种具有特征和标签的设置称为监督学习。因为你的数据告诉你的模型在给定特定输入时应该输出什么。

让我们创建一个模型类：

1. 子类nn.Module (几乎所有 PyTorch 模型都是nn.Module的子类) 。
2. 在构造函数中创建 2 个 nn.Linear层, 能够处理X和y的输入和输出形状。
3. 定义一个包含模型前向传递计算的 forward()方法。
4. 实例化模型类并将其发送到目标 device

'''

# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features, produces 1 feature (y)
    
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x)) # computation goes through layer_1 first then the output of layer_1 goes through layer_2

# 4. Create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)
print(model_0)




