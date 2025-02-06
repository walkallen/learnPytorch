

'''

Reference
    https://huggingface.co/docs/transformers/v4.48.2/zh/pipeline_tutorial

推理pipeline

pipeline() 让使用Hub上的任何模型进行任何语言、计算机视觉、
语音以及多模态任务的推理变得非常简单。即使您对特定的模态没有经验，
或者不熟悉模型的源码，您仍然可以使用pipeline()进行推理！本教程将教您：

如何使用pipeline() 进行推理。
如何使用特定的tokenizer(分词器)或模型。
如何使用pipeline() 进行音频、视觉和多模态任务的推理。


Pipeline 使用

虽然每个任务都有一个关联的 pipeline(), 但使用通用的抽象的 pipeline() 更加简单，
其中包含所有特定任务的 pipelines 。 pipeline() 会自动加载一个默认模型和一个能够
进行任务推理的预处理类。让我们以使用 pipeline() 进行自动语音识别 (ASR) 或语音转文本为例。




'''


import gc
import torch


# 1. 首先，创建一个pipeline()并指定推理任务：


from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition")




# 2. 将您的输入传递给 pipeline()。对于语音识别，这通常是一个音频输入文件：

print(
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
)



''' 

让我们尝试来自 OpenAI 的 Whisper large-v2 模型。Whisperb 比 Wav2Vec2 晚 2 年发布，
使用接近 10 倍的数据进行了训练。因此, 它在大多数下游基准测试上击败了 Wav2Vec2。 
它还具有预测标点和大小写的附加优势, 而 Wav2Vec2 则无法实现这些功能。



'''


# # 完成操作后，删除 pipeline 对象
# del transcriber

# # 手动触发垃圾回收
# gc.collect()

# # 清空 PyTorch 缓存
# torch.cuda.empty_cache()






'''

您没有得到您期望的结果? 可以在Hub上查看一些最受欢迎的自动语音识别模型 ，看看是否可以获得更好的转录。

让我们尝试来自 OpenAI 的Whisper large-v2 模型。Whisperb 比 Wav2Vec2 晚 2 年发布，
使用接近10倍的数据进行了训练。因此，它在大多数下游基准测试上击败了 Wav2Vec2。 
它还具有预测标点和大小写的附加优势，而 Wav2Vec2 则无法实现这些功能。

让我们在这里尝试一下，看看它的表现如何：

'''



transcriber = pipeline(model="openai/whisper-large-v2")

print(
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
)


# 完成操作后，删除 pipeline 对象
del transcriber

# 手动触发垃圾回收
gc.collect()

# 清空 PyTorch 缓存
torch.cuda.empty_cache()

'''
现在这个结果看起来更准确了！要进行深入的Wav2Vec2与Whisper比较，请参阅音频变换器课程。 
我们鼓励您在 Hub 上查看不同语言的模型，以及专业领域的模型等。您可以在Hub上直接
查看并比较模型的结果，以确定是否适合或处理边缘情况是否比其他模型更好。
如果您没有找到适用于您的用例的模型，您始终可以训练自己的模型！

如果您有多个输入，您可以将输入作为列表传递：


transcriber(
    [
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    ]
)


Pipelines 非常适合用于测试，因为从一个模型切换到另一个模型非常琐碎；
但是，还有一些方法可以将它们优化后用于大型工作负载而不仅仅是测试。
请查看以下指南，深入探讨如何迭代整个数据集或在 Web 服务器中使用 Pipelines :


在数据集上使用流水线
在 Web 服务器中使用流水线



参数

pipeline() 支持许多参数；有些是适用于特定任务的，而有些适用于所有pipeline。
通常情况下，您可以在任何地方指定对应参数：


transcriber = pipeline(model="openai/whisper-large-v2", my_parameter=1)

out = transcriber(...)  # This will use `my_parameter=1`.
out = transcriber(..., my_parameter=2)  # This will override and use `my_parameter=2`.
out = transcriber(...)  # This will go back to using `my_parameter=1`.


让我们查看其中的三个重要参数：

设备

如果您使用 device=n, pipeline 会自动将模型放在指定的设备上。无论您使用 PyTorch 还是 Tensorflow, 这都可以工作。


如果模型对于单个 GPU 来说过于庞大，并且您正在使用 PyTorch, 
您可以设置 device_map="auto" 以自动确定如何加载和存储模型权重。
使用 device_map 参数需要安装🤗 Accelerate 软件包：


以下代码会自动在各个设备上加载和存储模型权重：


transcriber = pipeline(model="openai/whisper-large-v2", device_map="auto")

请注意，如果传递了 device_map="auto"，在实例化您的 pipeline 时不需要添加 device=device 参数，否则可能会遇到一些意外的状况！



批量大小
默认情况下，pipelines不会进行批量推理，原因在这里详细解释。因为批处理不一定更快，实际上在某些情况下可能会更慢。

但如果在您的用例中起作用，您可以使用：


'''


transcriber = pipeline(model="openai/whisper-large-v2", device=0, batch_size=2)
audio_filenames = [f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)]
texts = transcriber(audio_filenames)


# 完成操作后，删除 pipeline 对象
del transcriber
# 手动触发垃圾回收
gc.collect()
# 清空 PyTorch 缓存
torch.cuda.empty_cache()

'''
以上代码会在提供的 4 个音频文件上运行 pipeline, 它会将它们以 2 个一组的批次传递给模型 (模型在GPU上,
此时批处理更有可能有所帮助），而您无需编写额外的代码。输出应始终与没有批处理时收到的结果相一致。
它只是一种帮助您更快地使用 pipeline 的方式。


pipeline 也可以减轻一些批处理的复杂性, 因为对于某些 pipeline, 
需要将单个项目 (如长音频文件) 分成多个部分以供模型处理。 pipeline 为您执行这种 chunk batching。


任务特定参数

所有任务都提供了特定于任务的参数，这些参数提供额外的灵活性和选择，以帮助您完成工作。 
例如, transformers.AutomaticSpeechRecognitionPipeline.call() 方法具有
一个 return_timestamps 参数，对于字幕视频似乎很有帮助：


'''

transcriber = pipeline(model="openai/whisper-large-v2", return_timestamps=True)

print(
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
)

# 完成操作后，删除 pipeline 对象
del transcriber
# 手动触发垃圾回收
gc.collect()
# 清空 PyTorch 缓存
torch.cuda.empty_cache()


'''
{
    'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.', 
    'chunks': [
        {'timestamp': (0.0, 11.88), 'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its'}, 
        {'timestamp': (11.88, 12.38), 'text': ' creed.'}
    ]
}

每个任务都有许多可用的参数, 因此请查看每个任务的 API 参考, 以了解您可以进行哪些调整！
例如, AutomaticSpeechRecognitionPipeline 具有 chunk_length_s 参数，
对于处理非常长的音频文件（例如，为整部电影或长达一小时的视频配字幕）非常有帮助，这通常是模型无法单独处理的：

'''

transcriber = pipeline(model="openai/whisper-large-v2", chunk_length_s=30, return_timestamps=True)
print(
transcriber("https://huggingface.co/datasets/sanchit-gandhi/librispeech_long/resolve/main/audio.wav")
)












