

# 看起来加载的模型预测与之前的模型预测（保存之前进行的预测）相同。
# 这表明我们的模型正在按预期保存和加载。


'''
6. 将所有内容放在一起





'''

# Import PyTorch and matplotlib
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt





draw_matplotlib = True

    




# Check PyTorch version
torch.__version__



# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



print("\n======================= 创建一些数据 \n")


# Create weight and bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and labels)
# without unsqueeze, errors will happen later on (shapes within linear layers)
# 如果没有 unsqueeze ， 那么会报错误
X = torch.arange(start, end, step).unsqueeze(dim=1) 
y = weight * X + bias 

print(X[:10], y[:10], X.shape, y.shape)





print("\n======================= 分割数据 \n")


# 我们将使用 80/20 分割，其中 80% 的训练数据和 20% 的测试数据。

# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print( len(X_train), len(y_train), len(X_test), len(y_test), X_train.shape )






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


# Note: If you've reset your runtime, this function won't work, 
# you'll have to rerun the cell above where it's instantiated.

plot_predictions(X_train, y_train, X_test, y_test, draw=draw_matplotlib)






print("\n======================= 使用 nn.Linear \n")

'''

不再使用 nn.Parameter(), 作为替代，我们使用 nn.Linear(in_features, out_features) 

其中in_features是输入数据的维度数, out_features 是您希望将其输出到的维度数。

在我们的例子中，这两个值都是1因为我们的数据每个标签 ( y ) 有1输入特征 ( X )。



'''

# 还有很多torch.nn模块具有预构建计算的示例，包括许多流行且有用的神经网络层。

# Subclass nn.Module to make our model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, 
                                      out_features=1)
    
    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Set the manual seed when creating the model (this isn't always need but is used for demonstrative purposes, try commenting it out and seeing what happens)
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print( model_1, model_1.state_dict() )




print("\n======================= 将模型加载到 gpu 上 \n")

# 注意model_1.state_dict()的输出， nn.Linear()层为我们创建了一个随机权weight和bias参数。
# 现在让我们将模型放在 GPU 上（如果可用）。
# 我们可以使用.to(device)更改 PyTorch 对象所在的设备。
# 首先让我们检查这个模型所处的当前设备。

# Check model device
print( next(model_1.parameters()).device )


# 让我们将其更改为在 GPU 上（如果可用）。

# Set model to GPU if it's availalble, otherwise it'll default to CPU
model_1.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not


print( next(model_1.parameters()).device )




'''
训练

首先，我们需要一个损失函数和一个优化器。

让我们使用之前使用过的相同函数nn.L1Loss()和torch.optim.SGD() 。

我们必须将新模型的参数（ model.parameters() ）传递给优化器，以便它在训练期间调整它们。

0.01的学习率以前也很有效，所以让我们再次使用它。
'''

# Create loss function
loss_fn = nn.L1Loss()

# Create optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), # optimize newly created model's parameters
                            lr=0.01)





'''

与之前的训练循环相比, 我们在此步骤中要做的唯一不同的事情是将数据放在目标device上。
我们已经使用model_1.to(device)将模型放在目标device上。
我们可以对数据做同样的事情。
这样，如果模型位于 GPU 上，则数据也位于 GPU 上（反之亦然）。
这次让我们更进一步并设置epochs=1000 。
'''


torch.manual_seed(42)

# Set the number of epochs 
epochs = 1000 

# Put data on the available device
# Without this, error will happen (not all model/data on device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    ### Training
    model_1.train() # train mode is on by default after construction

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad optimizer
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    ### Testing
    model_1.eval() # put the model in evaluation mode for testing (inference)
    # 1. Forward pass
    with torch.inference_mode():
        test_pred = model_1(X_test)
    
        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")




print("\n======================= 让我们检查我们的模型已经学习的参数，并将它们与我们硬编码的原始参数进行比较 \n")

# 让我们检查我们的模型已经学习的参数，并将它们与我们硬编码的原始参数进行比较。

# Find our model's learned parameters
from pprint import pprint # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html 
print("The model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")






'''
进行预测

现在我们已经有了一个经过训练的模型，让我们打开它的评估模式并做出一些预测。



'''

# Turn model into evaluation mode
model_1.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_1(X_test)
print(y_preds)

# plot_predictions(predictions=y_preds) # -> won't work... data not on CPU

# Put data on the CPU and plot it
plot_predictions(predictions=y_preds.cpu(), draw=draw_matplotlib)









print("\n======================= 保存和加载模型 \n")






'''
6.5 保存和加载模型

我们对模型预测感到满意，因此将其保存到文件中，以便以后使用。


'''

from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 




# 为了确保一切正常，让我们重新加载它。

# Instantiate a fresh instance of LinearRegressionModelV2
loaded_model_1 = LinearRegressionModelV2()

# Load model state dict 
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
loaded_model_1.to(device)

print(f"Loaded model:\n{loaded_model_1}")
print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")


# 现在我们可以评估加载的模型，看看它的预测是否与保存之前所做的预测一致。

# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)

print(y_preds == loaded_model_1_preds)
