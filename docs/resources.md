# 资源来源核验补充

核验日期：2026-11-07。区分“仓库发布者声明”“本地文件内可见事实”和“资源权利已确认”；本次不代替作者签署原创或授权声明。

## 本地权重

“evidence/model-metadata.json”记录两套权重的SHA256及checkpoint字段。DenseNet含epoch=30；EfficientNet含epoch=4与嵌套config。epoch的索引约定不明，不能直接当作完整训练轮次证明。

EfficientNet内嵌配置确认：类别顺序normal/pneumonia/tuberculosis，目标512×512，归一化mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]，与现有推理代码一致。训练配置包含AdamW、学习率0.0003、batch_size=16、目标epochs=30。这是checkpoint中的配置记录，不是训练完成日志或独立准确率证明。两套权重的作者、训练数据授权与完整实验过程仍待核实。

## 演示图

原交付版demo_manifest.json提供muhammadrehan00_chest_xray_dataset目录及原始文件名，这是来源线索。候选数据集：[Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset)。本机原数据目录已不存在；公开页面未取得可核验的许可正文。因此尚不能确认这六图与上游文件一致、授权完整或患者级独立性。

## 可选外部模型

发布者的模型卡均声明Apache-2.0；记录见“evidence/external-model-metadata.json”，已保存模型ID、revision和文件清单：

- [sukhmani1303/tuberculosis-vit-model](https://huggingface.co/sukhmani1303/tuberculosis-vit-model)：已确认仓库存在model.pt与config.json，与加载器所需文件名相符；未下载或执行远程模型，故不能宣称已验证可运行。模型卡说明仅限教育/研究用途。
- [radiovision/Pneumonia-Detection](https://huggingface.co/radiovision/Pneumonia-Detection)：公开模型卡说明其为胸片肺炎分类ViT；未下载或验证其当前输出兼容性。发布者指标不属于本项目自己的指标。

仓库许可证声明不能自动证明本地两套CNN权重或演示图的权利来源。

## 知识摘要的出处线索

- ATS/IDSA成人社区获得性肺炎指南（2019）：[ATS官方指南入口](https://www.thoracic.org/statements/guideline-implementation-tools/diagnosis-and-treatment-of-cap.php)，原文DOI 10.1164/rccm.201908-1581ST。
- WHO结核病事实页：[Tuberculosis](https://www.who.int/news-room/fact-sheets/detail/tuberculosis)，当前可见页面日期2026-03-24；不自动证明项目原摘要使用了此版本。
- WHO胸片资料：[Chest radiography in tuberculosis detection](https://www.who.int/publications/i/item/9789241511506)，2016-12-09，ISBN 9789241511506。
- Local triage summary：项目本地摘要，没有对应外部原文；需由作者与医学审核人员确认。

本轮补充出处，不改变RAG_DOCUMENTS文本和检索排序。链接确认与完整医学内容审查是不同工作，不能宣称这些资料是截至参赛时所有最新临床指南。

## 软件许可证与展示素材

已从本机已安装版本收集14项Python直接依赖及React/react-dom/scheduler的许可证文件与版本，见“licenses/许可索引.json”。这不是所有传递依赖的完整法律审计。开发构建依赖由package-lock.json记录，相关包附带许可仍需在再分发时保留。Google Fonts保持原页面引用方式，没有将字体文件重新打包。

原PPT、Word中的第三方图片、字体、模板与图表尚未逐项获得授权证明。为避免把旧材料中的未复现指标当作新证据，本代码仓库不收录这些历史Word/PPT；原文件仍保留在旧交付目录中供核对。
