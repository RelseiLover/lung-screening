# 独立评测准备

当前仅有六张精选演示图，没有完整训练/验证/测试集或逐样本历史预测记录。原manifest指向的E:/kaggleapi数据目录当前不存在。因此本轮没有生成独立准确率、AUC或“系统提升”结论。

## 已完成的功能实验

见“evidence/demo-ablation.json”：两套本地CNN均对六图完成推理；完整流程和“文档召回为空”的消融均运行成功。六例最终筛查类别均未因关闭召回而改变，第5例置信度从0.79变为0.78。该消融仍保留症状规则、通用兜底知识与下游保护，不等同于删除整个RAG模块。

两套模型在第1、6张图上的最高分标签不同；这需要独立数据进一步评估，不据此修改模型选择或推理阈值。运行耗时包含可能的首次加载，不作为稳定性能基准。

## 历史指标核验

旧报告生成脚本中存在Accuracy从0.840到0.900、Macro F1从0.844到0.937等数字；当前没有对应的测试集清单、逐例标签/预测或原始实验日志。这些属于待核验的历史陈述，不是本次复现结果。正式答辩不应把它们作为已验证提升引用。

## 数据准备与命令

1. 提供有授权的独立数据集，按患者划分，避免同一患者的相关影像跨训练与测试集。
2. CSV清单使用UTF-8，列为path,label,split；path相对CSV目录，label只取normal、pneumonia、tuberculosis，待评测集split全部填test。
3. 如有训练集清单，使用相同列名但split填train。脚本检查图像内容SHA256重复及训练/测试完全重复，不能识别同患者的不同影像。
4. 在仓库根目录执行：

```powershell
python scripts/evaluate_dataset.py --manifest "D:/authorized-data/test.csv" --train-manifest "D:/authorized-data/train.csv" --model "models/efficientnetv2s_best.pth" --arch tf_efficientnetv2_s.in21k_ft_in1k --output "artifacts/evaluation-efficientnet"
```

评测DenseNet时将model改为densenet121_best.pth、arch改为densenet121并使用不同output目录。缺少train-manifest时可省略，但报告会标记未检查训练重叠。

输出predictions.csv和metrics.json，包含逐例概率、混淆矩阵、分类报告、Accuracy、样本与权重哈希；仅在三类均存在时计算Macro OVR AUC。推理失败会停止，不用启发式回退填充指标。该脚本评估单个CNN，不能直接代替完整系统（症状+影像）的有标签评测。

## 真正的系统/RAG评测还需

配对症状、影像、目标风险标签、引用证据的标注病例；固定云端模型版本和配置；相同病例比较完整流程、无检索、无复核等条件。报告病例数量、划分、失败率与统计方法，不从六个精选演示样例推断临床性能。
