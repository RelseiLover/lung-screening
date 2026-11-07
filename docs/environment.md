# 环境配置

## 支持范围与依赖

建议 Python 3.12、Node.js 22.12+ 或 24+。Python 直接依赖固定在 requirements.txt；开发/测试增加 requirements-dev.txt。它们不是所有传递依赖的完整哈希锁文件。前端 npm 依赖由 package-lock.json 锁定，使用 npm ci 安装。

本机验证环境为 Windows 64 位、Python 3.12.9。仓库不包含 Python、Node、wheel 或在线专病模型的离线安装包。运行 React 构建产物无需 Node，但克隆后需先构建被忽略的 frontend/dist。

## .env 语法与优先级

将根目录 .env.example 复制为同目录 .env。程序自动读取，支持逐行 KEY=value、整行注释和成对单/双引号。空值忽略，不执行 shell 命令，不支持变量展开、多行值或行尾注释。未知键会报错。修改配置后重启服务。

优先级：系统环境变量 > 根目录 .env > 代码默认值。旧式 backend/deepseek_local_config.json 仅保留兼容逻辑，不推荐使用且必须忽略。路径变量不展开 `~`；自定义路径建议填写绝对路径。不要把密钥填进 .env.example。

| 变量 | 默认与作用 |
| --- | --- |
| DEEPSEEK_API_KEY | 空；不调用云端，使用本地整合 |
| DEEPSEEK_BASE_URL | https://api.deepseek.com/v1 |
| DEEPSEEK_MODEL | deepseek-chat |
| OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL | 可选兼容服务；使用时移除全部 DEEPSEEK 配置，并显式设置 base URL 和 model |
| HF_HUB_OFFLINE | 1；只用已缓存专病模型，0允许下载 |
| HF_TB_REPO | sukhmani1303/tuberculosis-vit-model |
| HF_PNEUMONIA_REPO | radiovision/Pneumonia-Detection |
| MEDICAL223_REPO_DIR | 默认仓库根目录，其下须有 models 子目录 |
| MEDICAL223_MODEL_PATH | 空；自动选 models/efficientnetv2s_best.pth，再尝试 DenseNet |
| MEDICAL223_MODEL_ARCH | tf_efficientnetv2_s.in21k_ft_in1k；显式选DenseNet时配densenet121 |
| MEDICAL223_IMAGE_SIZE | 512，与当前主模型checkpoint配置一致 |

## 常见问题

- 权重只是几行文字：Git LFS 文件尚未拉取，运行 git lfs pull，再检查 models/manifest.json。
- 缺少 frontend/dist：在 frontend 执行 npm ci、npm run build。
- 包导入失败：确认使用同一虚拟环境安装依赖和启动；运行 python run.py --check 查看具体模块。
- 端口占用：使用 python run.py --port 8503；前端开发代理仍指向8502，需同步修改vite配置。
- 没有DeepSeek结果：本地模式本来不需要密钥；配置私有凭据后重启。联网验证只用虚构资料，见scripts/verify_llm.py --help。

不要把 .env、缓存、日志或患者数据作为排错附件上传到公共仓库。
