# FineTuning 资源说明

本目录保存与微调流程相关的脚本，以及体积较大的数据集和模型资源说明。

## 目录内容

- `build_safety_prompts_dataset.py`: 构建输入侧数据集
- `build_output_pku_saferlhf_dataset.py`: 构建输出侧数据集
- `fine_tune_lora.py`: 输入侧 LoRA 微调脚本
- `fine_tune_output_pku_zh_lora.py`: 输出侧 LoRA 微调脚本
- `export_onnx.py`: 导出 ONNX 模型脚本
- `translate_output_dataset_to_zh.py`: 输出数据集中文处理脚本
- `datafiles/`: 训练、验证、测试等数据文件
- `models/`: 基座模型与微调后的模型文件

## 大文件存放

`datafiles/` 和 `models/` 由于体积较大，不建议直接提交到 GitHub。
建议将这两部分单独上传到 Hugging Face Hub，并在本地按以下方式恢复目录：

```text
FineTuning/
  datafiles/
  models/
```

## 与 GitHub 仓库的关系

仓库中的其他代码文件都保留在 GitHub：

[JasmineWA/ai-chat-safety](https://github.com/JasmineWA/ai-chat-safety)

GitHub 主要用于代码版本管理，Hugging Face 主要用于存放大体积的数据集和模型文件。
