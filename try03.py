
from datasets import load_dataset
from datasets import load_from_disk

# 加载一个示例数据集（包含多个 split）
# dataset_dict = load_dataset('imdb')

# # 保存 DatasetDict 到本地
# dataset_dict.save_to_disk('/mnt/SD1T/suhao/learnPytorchbak/learnPytorch/data_pku_msd')

# 加载保存的 DatasetDict
# loaded_dataset_dict = load_from_disk('/mnt/SD1T/suhao/learnPytorchbak/learnPytorch/data_visa_mvtec_pkuPcb_pkuPhone')
loaded_dataset_dict = load_from_disk('/mnt/SD1T/suhao/learnPytorchbak/learnPytorch/data_visa_mvtec_pkuPcb_pkuPhone_250120_arrow')
# loaded_dataset_dict = load_from_disk('/mnt/SD1T/suhao/learnPytorchbak/learnPytorch/data_pku_msd')

# 查看数据集
print(loaded_dataset_dict)
print('\n')

print(loaded_dataset_dict['train'][0])
print('\n')

cppe5 = load_dataset("cppe-5")

print(cppe5)
print("\n")

print(cppe5['train'][0])

cppe5['train'][0]['image'].show()




