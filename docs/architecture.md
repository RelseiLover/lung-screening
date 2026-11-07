# 技术架构与边界

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 页面 | frontend/src/App.tsx、components.tsx、Results.tsx | 原布局、问诊状态、文件上传、结果与四标签 |
| 本地API | backend/web_api.py | 表单与图像校验、临时文件、同源资源及结果适配 |
| 展示适配 | backend/presentation.py | 与原页面相同的症状拼接和模型/路由展示函数 |
| 编排 | backend/medical223_langgraph.py | 条件路径、节点顺序和有限反馈 |
| 核心能力 | backend/medical223_core.py | CNN、轻量RAG、路由、复核、LLM及规则回退 |
| 配置 | backend/runtime_config.py | 根目录.env字面值加载，系统环境优先 |
| 兼容页面 | backend/streamlit_app.py | 保留原Streamlit入口 |

主流程为 symptom → image → knowledge → route → 可选专病节点 → diagnosis → review → check；满足条件时 feedback 返回 diagnosis。模型及索引保留进程内缓存，API用锁保护共享模型与LLM状态。

默认EfficientNetV2-S，DenseNet是替代模型而不是集成投票。图像转RGB、缩放512×512并使用现有归一化。RAG为四条摘要的TF-IDF召回，不是大规模向量数据库。大部分Agent节点为确定性规则，不宣称LLM自主规划或MCP调用。

POST /api/screen 使用multipart/form-data，字段answers为JSON字符串，image为可选PNG/JPEG。响应对应原页面显示字段，不返回本机临时路径。GET /api/health返回服务状态与构建产物是否存在。

图片用于本地模型推理；启用云端LLM时会发送症状、派生分数与知识证据，不发送原图。原始图像在成功或异常后清理，进程强杀可能遗留临时文件。原型没有生产级账户、长期病例存储、审计与医疗系统集成。

本次仓库整理只调整路径、工具入口、文档和格式，不重写筛查算法与页面布局。
