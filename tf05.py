

'''
Reference
    https://huggingface.co/docs/transformers/v4.48.2/zh/preprocessing


预处理


在您可以在数据集上训练模型之前, 数据需要被预处理为期望的模型输入格式。
无论您的数据是文本、图像还是音频, 它们都需要被转换并组合成批量的张量。
🤗 Transformers 提供了一组预处理类来帮助准备数据以供模型使用。在本教程中, 您将了解以下内容：

    对于文本, 使用分词器 (Tokenizer) 将文本转换为一系列标记 (tokens), 并创建 tokens 的数字表示, 将它们组合成张量。
    对于语音和音频, 使用特征提取器 (Feature extractor) 从音频波形中提取顺序特征并将其转换为张量。
    图像输入使用图像处理器(ImageProcessor) 将图像转换为张量。
    多模态输入, 使用处理器(Processor) 结合了 Tokenizer 和 ImageProcessor 或 Processor。



处理文本数据的主要工具是Tokenizer。Tokenizer根据一组规则将文本拆分为tokens。
然后将这些tokens转换为数字, 然后转换为张量, 成为模型的输入。模型所需的任何附加输入都由Tokenizer添加。

如果您计划使用预训练模型, 重要的是使用与之关联的预训练Tokenizer。
这确保文本的拆分方式与预训练语料库相同, 并在预训练期间使用相同的标记-索引的对应关系（通常称为词汇表-vocab）。


'''


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


# 然后将您的文本传递给tokenizer：

encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input)

'''

{
    'input_ids': [101, 2091, 1136, 1143, 13002, 1107, 1103, 5707, 1104, 16678, 1116, 117, 1111, 1152, 1132, 11515, 1105, 3613, 1106, 4470, 119, 102], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

tokenizer返回一个包含三个重要对象的字典：

    input_ids 是与句子中每个token对应的索引。
    attention_mask 指示是否应该关注一个token。
    token_type_ids 在存在多个序列时标识一个token属于哪个序列。

通过解码 input_ids 来返回您的输入：
'''

print(
tokenizer.decode(encoded_input["input_ids"])
)



'''
如您所见, tokenizer向句子中添加了两个特殊token - CLS 和 SEP（分类器和分隔符）。
并非所有模型都需要特殊token, 但如果需要, tokenizer会自动为您添加。

如果有多个句子需要预处理, 将它们作为列表传递给tokenizer：


'''


batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences)
print(encoded_inputs)

'''

{
    'input_ids': 
        [
            [101, 1252, 1184, 1164, 1248, 6462, 136, 102], 
            [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
            [101, 1327, 1164, 5450, 23434, 136, 102]
        ], 
    'token_type_ids': 
        [
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0]], 
    'attention_mask': 
        [
            [1, 1, 1, 1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1, 1, 1]]
}



填充

    句子的长度并不总是相同, 这可能会成为一个问题, 因为模型输入的张量需要具有统一的形状。
    填充是一种策略, 通过在较短的句子中添加一个特殊的 padding token, 以确保张量是矩形的。

    将 padding 参数设置为 True, 以使批次中较短的序列填充到与最长序列相匹配的长度：



'''
print('\n')
print('将 padding 参数设置为 True, 以使批次中较短的序列填充到与最长序列相匹配的长度：')
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True)
print(encoded_input)


'''

{
    'input_ids': 
        [
            [101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
            [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
            [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
    'token_type_ids': 
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
    'attention_mask': 
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
}

第一句和第三句因为较短, 通过0进行填充, 。


截断

另一方面, 有时候一个序列可能对模型来说太长了。在这种情况下, 您需要将序列截断为更短的长度。

将 truncation 参数设置为 True, 以将序列截断为模型接受的最大长度：


'''

print('\n')
print('将 truncation 参数设置为 True, 以将序列截断为模型接受的最大长度：')
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
print(encoded_input)

'''

{
    'input_ids': 
        [
            [101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
            [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
            [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
    'token_type_ids': 
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
    'attention_mask': 
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
}


查看填充和截断概念指南, 了解更多有关填充和截断参数的信息。

构建张量

最后, tokenizer可以返回实际输入到模型的张量。

将 return_tensors 参数设置为 pt（对于PyTorch）或 tf（对于TensorFlow）：


'''


print('\n')
print('将 return_tensors 参数设置为 pt（对于PyTorch）或 tf（对于TensorFlow）')
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_input)


'''

{
    'input_ids': 
        tensor(
            [
                [  101,  1252,  1184,  1164,  1248,  6462,   136,   102,     0,     0,    0,     0,     0,     0,     0],
                [  101,  1790,   112,   189,  1341,  1119,  3520,  1164,  1248,  6462,  117, 21902,  1643,   119,   102],
                [  101,  1327,  1164,  5450, 23434,   136,   102,     0,     0,     0,    0,     0,     0,     0,     0]]), 
    'token_type_ids': 
        tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
    'attention_mask': 
        tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
}




音频
    对于音频任务, 您需要 feature extractor 来准备您的数据集以供模型使用。
    feature extractor 旨在从原始音频数据中提取特征, 并将它们转换为张量。

    加载 MInDS-14 数据集（有关如何加载数据集的更多详细信息, 请参阅🤗 Datasets教程）
    以了解如何在音频数据集中使用 feature extractor：



'''



from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")


# 访问 audio 列的第一个元素以查看输入。调用 audio 列会自动加载和重新采样音频文件：

print('\n')
print('访问 audio 列的第一个元素以查看输入。调用 audio 列会自动加载和重新采样音频文件：')
print(
dataset[0]["audio"]
)













