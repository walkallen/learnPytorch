


'''
Reference

https://huggingface.co/learn/nlp-course/chapter4/1?fw=pt



Hugging Face Hub ——我们的主要网站 —— 是一个中心平台，使任何人都可以发现、使用和贡献最新的模型和数据集。
它托管了各种模型，其中超过 10,000 个可供公开使用。我们将专注于本章中的模型，并在第 5 章中查看数据集。

中心库中的模型不仅限于 🤗 Transformers 或甚至 NLP。还包括 Flair 和 AllenNLP 的 NLP 模型、
Asteroid 和 pyannote 的语音模型以及 timm 的视觉模型等。


每个模型都托管在 Git 仓库中，这允许版本控制和可重复性。在 Hub 上共享模型意味着将其向社区开放，
并使其对任何希望轻松使用它的人可访问，从而消除他们自己训练模型的需求，并简化了共享和使用。

此外，在 Hub 上共享模型会自动部署该模型的托管推理 API。
社区中的任何人都可以直接在模型页面上使用自定义输入和适当的控件进行测试。

最好的部分是，在 Hub 上分享和使用任何公共模型都是完全免费的！
如果您希望私下分享模型，也存在付费计划。



模型中心使选择合适的模型变得简单，因此在使用任何下游库时，只需几行代码即可完成。
让我们看看如何实际使用这些模型之一，以及如何回馈社区。

让我们假设我们正在寻找一个基于法语模型，能够执行遮掩填充。

我们选择 camembert-base 检查点来尝试它。我们需要的标识符 camembert-base 就可以开始使用它了！
正如你在前面的章节中看到的，我们可以使用 pipeline() 函数来实例化它：

'''



from transformers import pipeline

camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")

print(results)


'''
[
    {'score': 0.5239014625549316, 'token': 7200, 'token_str': 'délicieux', 'sequence': 'Le camembert est délicieux :)'}, 
    {'score': 0.09637846797704697, 'token': 2183, 'token_str': 'excellent', 'sequence': 'Le camembert est excellent :)'}, 
    {'score': 0.03632454574108124, 'token': 26202, 'token_str': 'succulent', 'sequence': 'Le camembert est succulent :)'}, 
    {'score': 0.029079986736178398, 'token': 528, 'token_str': 'meilleur', 'sequence': 'Le camembert est meilleur :)'}, 
    {'score': 0.02721838839352131, 'token': 1654, 'token_str': 'parfait', 'sequence': 'Le camembert est parfait :)'}
]


如您所见，在管道中加载模型非常简单。您需要注意的唯一一点是所选的检查点适合用于其将要执行的任务。
例如，在这里我们正在加载 camembert-base 管道中的 fill-mask 检查点，这是完全正常的。
但如果我们把此检查点加载到 text-classification 管道中，结果将没有任何意义，
因为 camembert-base 的头部不适合这项任务！我们建议在 Hugging Face Hub 界面中使用任务选择器来选择适当的检查点：


您也可以直接使用模型架构实例化检查点：


'''


from transformers import CamembertTokenizer, CamembertForMaskedLM

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")



'''
然而，我们建议使用 Auto* 类，因为这些类的设计是架构无关的。
虽然之前的代码示例限制了用户只能加载 CamemBERT 架构中的检查点，但使用 Auto* 类可以使切换检查点变得简单：

'''


from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")

'''
使用预训练模型时，请确保检查其训练方式、所使用的数据集、其局限性以及其偏差。
所有这些信息都应在模型卡片上标明。

'''