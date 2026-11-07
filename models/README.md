# 模型权重

包含efficientnetv2s_best.pth（默认）与densenet121_best.pth（可选替代），固定类别顺序normal、pneumonia、tuberculosis。它们不是双模型投票。

字节数、SHA256和架构见manifest.json。EfficientNet内嵌配置与当前512×512输入及归一化一致，DenseNet训练依据仍需核实。metadata见../docs/evidence/model-metadata.json。

.pth通过Git LFS追踪。克隆后运行git lfs pull；检查文件不是LFS指针。运行python run.py --check将检测指针或缺失权重。

本地文件包含checkpoint参数，但未附完整训练日志、原始训练/测试划分或作者授权。公开或向评审分享前须确认权利来源。不能将外部模型卡的Apache-2.0声明自动应用到这两套本地权重。

设置其他模型时需同时核对MEDICAL223_MODEL_PATH和MEDICAL223_MODEL_ARCH，不任意更改类别顺序或预处理。
