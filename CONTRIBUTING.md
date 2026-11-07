# 开发与代码规范

## 目录职责

后端 API、工作流、模型与展示辅助函数位于 backend；React 页面位于 frontend/src；工具放 scripts，测试放 tests。模型权重只放 models，演示输入只放 examples/images。实验输出写 artifacts，不将临时文件、私有配置或虚拟环境提交到源码。

## Python

- 使用 Python 3.12，四空格缩进；由 pyproject.toml 中的 Ruff 规则统一格式、导入顺序与基础错误检查。
- 函数/变量使用 snake_case，类使用 PascalCase，常量使用 UPPER_SNAKE_CASE。
- API 边界和共享数据结构尽量提供类型；解释有业务含义的阈值、模型回退和隐私边界，不为显而易见的赋值添加重复注释。
- 保留原模型预处理、类别顺序、路由和一致性保护。涉及其行为的变更须提供对应验证和变更理由。
- 异常提示与日志不得暴露密钥、原始患者资料或私有服务响应。输出到云端的数据范围须在文档与页面保持一致。
- 提交前执行 `python -m ruff check .` 和 `python -m ruff format --check .`；需要格式化时运行 `python -m ruff format .`。

## React / TypeScript

组件使用 PascalCase，Props 使用明确类型，复用问诊和结果组件；避免在前端复制医学路由规则。界面只展示后端结果，不将密钥放到 Vite 环境变量或浏览器资源中。

执行 `npm run format` 统一 Prettier 格式；`npm run format:check` 检查格式；`npm run build` 执行 TypeScript 和生产构建。修改依赖时一并更新 package.json 和 package-lock.json。

## 测试与提交

默认 `python scripts/test.py` 为不下载模型的快速测试；涉及路径、模型或流程的变更还需执行 `python scripts/test.py --models`。真实云调用必须单独显式启用，不能在普通测试或 CI 中调用收费 API。独立数据评测与六图功能演示严格分开。

使用真实提交，例如 `docs: clarify reviewer setup`、`fix: handle model download failure`、`test: cover missing image input`。每次提交描述实际问题、变更及验证，不编造既往开发过程。

Ruff 与 Prettier 只统一格式，不替代代码审查。不要仅为增加提交数量把一次导入拆成伪历史。
