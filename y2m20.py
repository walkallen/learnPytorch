


import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
print(torch.__version__)


draw_matplotlib = False


print("\n======================= 创建一些数据 \n")



# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print( X.shape, y.shape , X.size() , y.size(),X[:10], y[:10],  )


# 创建训练集，测试集


print("\n======================= 创建训练集，测试集 \n")


train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print( len(X_train), len(y_train), len(X_test), len(y_test) )

print()

print(X_train.shape , X_test.shape)



import matplotlib.pyplot as plt

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None,
                     draw = False):
    
    if not draw:
        return
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()
    return

plot_predictions(draw=draw_matplotlib)






# Create a Linear Regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(
                        torch.randn(
                            1,                  # <- start with random weights (this will get adjusted as the model learns)
                            dtype=torch.float), # <- PyTorch loves float32 by default
                        requires_grad=True)     # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(
                        torch.randn(
                            1,                  # <- start with random bias (this will get adjusted as the model learns)
                            dtype=torch.float), # <- PyTorch loves float32 by default
                        requires_grad=True)     # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias # <- this is the linear regression formula (y = m*x + b)
    


# 好吧，上面已经发生了很多事情，但让我们一点一点地分解它。
    


# 使用.parameters()检查其参数。

# Set manual seed since nn.Parameter are randomly initialzied
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

# Check the nn.Parameter(s) within the nn.Module subclass we created
print(list(model_0.parameters()))

print()

print()

for param in model_0.parameters():
    print(type(param), param.size())



print("\n======================= 我们还可以使用.state_dict()获取模型的状态（模型包含的内容）集 \n")

# 我们还可以使用.state_dict()获取模型的状态（模型包含的内容）。

# List named parameters 
print( model_0.state_dict() )




print("\n======================= 使用torch.inference_mode()进行预测 \n")

# 使用torch.inference_mode()进行预测

# Make predictions with model
with torch.inference_mode(): 
    y_preds = model_0(X_test)

# Note: in older PyTorch code you might also see torch.no_grad()
# with torch.no_grad():
#   y_preds = model_0(X_test)
    
# 您可能注意到我们使用torch.inference_mode()作为上下文管理器（这就是with torch.inference_mode():的作用）来进行预测。

# 顾名思义， torch.inference_mode()用于使用模型进行推理（进行预测）。
    
# torch.inference_mode()关闭了很多东西（比如梯度跟踪，这对于训练是必需的，但对于推理不是必需的）以使前向传递（数据通过forward()方法）更快。
    



# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

print(f"type of y_pred is {type(y_preds)}")


# 初始权重是随机的，所以预测结果很差

plot_predictions(predictions=y_preds, draw=draw_matplotlib)



"""

Loss function 损失函数   

测量模型预测 ( 例如 y_preds )与真实标签 ( 例如y_test ）相比的错误程度。越低越好。  
PyTorch 在torch.nn中有大量内置损失函数。   
回归问题的平均绝对误差 (MAE) ( torch.nn.L1Loss() )。二元分类问题的二元交叉熵（ torch.nn.BCELoss() ）。   


Optimizer 优化器   

告诉您的模型如何更新其内部参数以最好地降低损失。   
您可以在torch.optim中找到各种优化函数的实现。   
随机梯度下降（ torch.optim.SGD() ）。 Adam 优化器（ torch.optim.Adam() ）。  

"""



# 在 PyTorch 中创建损失函数和优化器
# 根据您正在处理的问题类型，取决于您使用的损失函数和优化器。

# 然而，有一些已知效果良好的常见值，例如 SGD（随机梯度下降）或 Adam 优化器。
# 以及用于回归问题（预测数字）的 MAE（平均绝对误差）损失函数
# 或用于分类问题（预测一件事或另一件事）的二元交叉熵损失函数。



# 平均绝对误差（MAE，在 PyTorch 中： torch.nn.L1Loss ）测量两点（预测和标签）之间的绝对差，然后取所有示例的平均值。

# 我们将使用 SGD， torch.optim.SGD(params, lr)其中：
# params是你想要优化的目标模型参数（例如我们之前随机设置的weights和bias值）。

# lr是您希望优化器更新参数的学习率，较高意味着优化器将尝试较大的更新
# （这些更新有时可能太大，优化器将无法工作），较低意味着优化器将尝试较小的更新
# （这些有时可能太小，优化器将花费很长时间才能找到理想值）。学习率被认为是一个超参数
# （因为它是由机器学习工程师设置的）。学习率的常见起始值为0.01 、 0.001 、 0.0001 ，
# 但是，这些值也可以随着时间的推移进行调整（这称为学习率调度）。






# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(
    params=model_0.parameters(), # parameters of target model to optimize
    lr=0.01) # learning rate (how much the optimizer should change parameters at each step, 
             # higher=more (less stable), lower=less (might take a long time))




'''
对于训练循环，我们将构建以下步骤：

1. foward pass 前向计算
    该模型一次遍历所有训练数据, 执行其forward()函数计算。
    model(x_train)

2. calculate loss 计算损失
    将模型的输出（预测）与真实情况进行比较并进行评估，以了解它们的错误程度。
    loss = loss_fn(y_pred, y_train)

3. zero gradients 清零梯度
    优化器梯度设置为零（默认情况下会累积），因此可以针对特定训练步骤重新计算它们。
    optimizer.zero_grad()

4. Perform backpropagation on the loss  对损失执行反向传播
    计算要更新的每个模型参数的损失梯度 (每个参数requires_grad=True ）。这称为反向传播，因此是“向后”。
    loss.backward()

5. Update the optimizer (gradient descent) 更新优化器（梯度下降）
    对于损失梯度, 使用requires_grad=True更新参数以改进它们。
    optimizer.step()

'''


"""
注意以下几点

在对其执行反向传播 ( loss.backward() ) 之前计算损失 ( loss = ... )。

在逐步执行（ optimizer.step() ）之前将梯度为零（ optimizer.zero_grad() ）。

对损失执行反向传播 ( loss.backward() )后，步进优化器 ( optimizer.step() )。

"""

'''
测试循环步骤如下


1. forward pass 前向计算
    该模型一次遍历所有测试数据, 执行其forward()函数计算。
    model(x_test)

2. Calculate the loss 计算损失
    将模型的输出（预测）与真实情况进行比较并进行评估，以了解它们的错误程度。
    loss = loss_fn(y_pred, y_test)
    
3. Calulate evaluation metrics (optional) 计算评估指标（可选）
    除了损失值之外，您可能还需要计算其他评估指标，例如测试集的准确性。
    这部分函数自己写

请注意，测试循环不包含执行反向传播（ loss.backward() ）或步进优化器（ optimizer.step() ），这是因为模型中的参数在测试期间没有更改，它们已经被计算过。对于测试，我们只对模型前向传递的输出感兴趣。
'''



'''

让我们将上述所有内容放在一起，并训练我们的模型 100 个epoch （前向传递数据），
我们将每 10 个 epoch 对其进行评估。

'''

torch.manual_seed(42)

# 设置训练的轮数
epochs = 100


# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(
    params=model_0.parameters(), # parameters of target model to optimize
    lr=0.01) # learning rate (how much the optimizer should change parameters at each step, 
             # higher=more (less stable), lower=less (might take a long time))


# 创建空的损失数组，记录每一轮训练的损失数据
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    # 将模型设置为训练模式，默认就是训练模式
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    # 前向计算一次
    y_pred = model_0(X_train)
    # print(y_pred)


    # 2. 根据前向计算的结果与标签计算一次损失
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    # 清零梯度
    optimizer.zero_grad()


    # y_pred 是由 X_train @ W 矩阵计算出来的，
    # 由于所有的参数都设置了 requires_grad = True, 所以 从 X_train 这个 tensor 一直到 y_pred 这个 tensor，
    # 计算过程中经历的所有 tensor 的计算关系被保存在一张图中
    # 当 backward 时，W 里所有的梯度会被更新, 就是说 W 里每一个参数对应的梯度 G 都会被更新
    loss.backward()

    # 根据反向传播计算到的梯度，以及相应的学习率，以及相应策略，更新参数
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    # 将模型设置为评估模式
    model_0.eval()

    with torch.inference_mode():
      # 1. Forward pass on test data
      # 前向计算，得到基于测试数据的结果
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      # 测试结果与测试标签，计算损失
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      # Print out what's happening
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")




# 看起来我们的损失随着每个时期的推移而下降，让我们绘制它来找出答案。

# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();
if draw_matplotlib:
    plt.show()



print("\n======================= 让我们检查模型的.state_dict()来看看我们的模型与我们为权重和偏差设置的原始值有多接近 \n")

# 让我们检查模型的.state_dict()来看看我们的模型与我们为权重和偏差设置的原始值有多接近。

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")




# 我们的模型非常接近计算权weight和bias的精确原始值（如果我们训练它更长时间，它可能会更接近）。

'''

4. 使用经过训练的 PyTorch 模型进行预测（推理）

使用 PyTorch 模型进行预测（也称为执行推理）时需要记住三件事：

1. 将模型设置为评估模式 ( model.eval() )。
2. 使用推理模式上下文管理器进行预测（ with torch.inference_mode(): ... ）。
3. 所有预测都应使用同一设备上的对象进行（例如仅在 GPU 上的数据和模型或仅在 CPU 上的数据和模型）。

前两项确保 PyTorch 在训练期间, 在幕后使用但推理不需要的所有有用的计算和设置都被关闭（这会导致更快的计算）。
第三个确保您不会遇到跨设备错误。



'''



print("\n======================= 使用经过训练的 PyTorch 模型进行预测（推理） \n")


# 1. Set the model in evaluation mode
model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(X_test)

print(y_preds)


plot_predictions(predictions=y_preds, draw=draw_matplotlib)




'''
5. 保存和加载 PyTorch 模型


要在 PyTorch 中保存和加载模型, 您应该注意三种主要方法 (以下所有内容均取自PyTorch 保存和加载模型指南）：


torch.save
    使用 Python 的 pickle 实用程序将序列化对象保存到磁盘。
    模型、张量和各种其他 Python 对象 (例如字典) 可以使用torch.save保存。

torch.load
    使用pickle的 unpickling 功能将 pickle 的 Python 对象文件（如模型、张量或字典）
    反序列化并加载到内存中。您还可以设置将对象加载到哪个设备 (CPU、GPU 等）。

torch.nn.Module.load_state_dict
    使用保存的 state_dict() 对象加载模型的参数字典 ( model.state_dict() )。

    
保存 PyTorch 模型的 state_dict()

保存和加载模型以进行推理 (进行预测) 的推荐方法是保存和加载模型的 state_dict() 。

让我们看看如何通过几个步骤来做到这一点：

1. 我们将创建一个目录，用于使用 Python 的pathlib模块将模型保存到调用的models中。

2. 我们将创建一个文件路径来保存模型。

3. 我们将调用 torch.save(obj, f) , 其中obj是目标模型的 state_dict() , f 是保存模型的文件名。



'''




print("\n======================= 保存模型 \n")

from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 

print(model_0.state_dict())



'''
加载已保存的 PyTorch 模型的 state_dict()

因为我们现在已经保存了模型 state_dict() models/01_pytorch_workflow_model_0.pth 
我们现在可以使用它来加载它 torch.nn.Module.load_state_dict(torch.load(f)) 
其中f是我们保存的模型state_dict()的文件路径。

为什么要在 torch.nn.Module.load_state_dict() 里面调用 torch.load() ?


因为我们只保存了模型的 state_dict() , 它是学习参数的字典, 而不是整个模型, 
所以我们首先必须使用 torch.load() 加载state_dict() , 
然后将该state_dict()传递给我们的新实例模型 (它是nn.Module的子类)。

为什么不保存整个模型？

这种方法（保存整个模型）的缺点是序列化数据绑定到特定的类以及保存模型时使用的确切目录结构......
因此，在其他项目中使用或重构后，您的代码可能会以各种方式损坏。


因此, 我们使用灵活的方法仅保存和加载state_dict() ，它基本上也是模型参数的字典。

让我们通过创建 LinearRegressionModel() 的另一个实例来测试它, 
它是torch.nn.Module的子类, 因此将具有内置方法load_state_dict() 。

'''



print("\n======================= 加载模型 \n")



# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))





print("\n======================= 验证加载模型的预测值 \n")


loaded_model_0.eval()

with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

# 让我们看看它们是否与之前的预测相同。
    
# Compare previous model predictions with loaded model predictions (these should be the same)
    
print(y_preds == loaded_model_preds)



