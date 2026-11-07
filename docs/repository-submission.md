# 代码仓库提交检查

本文件说明提交检查与操作方式。首次导入及后续真实提交以git log为准，不补造既往历史。

## 平台与访问

在GitHub/Gitee托管，设置Public或授予评审访问权限。须同时核对代码、Git LFS模型文件及所需附件的访问，不只检查仓库首页。GitHub CI配置只在GitHub运行，Gitee可在本地执行同一组检查。

## 首次提交前

1. 补齐README中的学校、成员职责、组别、老师及实际仓库地址。
2. 阅读development-history.md；先寻找已有开发仓库，不能丢弃真实历史。
3. 确认本地权重和样例图可被公开或分享给评审，见resources.md。
4. 完成README安装、前端构建、Ruff、前端格式检查和测试；确认没有真实.env、密钥、日志或私有病例。
5. 安装并启用Git LFS，再添加models中的.pth。.gitattributes已提供规则，但文件本身不能替代Git LFS安装与实际对象上传。

如果确认没有旧Git历史，后续由实际作者执行：

```shell
git init -b main
git lfs install --local
git config user.name "真实作者姓名"
git config user.email "真实作者邮箱或平台noreply邮箱"
git add .
git status
git diff --cached --stat
git lfs ls-files
git commit -m "chore: import existing screening prototype with documented provenance"
```

以上首次导入不是历史开发记录，不回填日期。配置实际远程地址并推送前，再次确认公开权限及资源授权。不要把示例身份或占位地址原样写入配置。

## 评审复现验收

在干净目录克隆真实仓库，执行git lfs pull、安装依赖、复制.env.example、构建frontend并运行python run.py。确认下载到的是完整权重；演示原有六例与无图像路径；核对结果显示实际模型还是规则回退。无云密钥的本地模式也应运行。

## 对照赛事五项要求

| 要求 | 本轮准备状态 | 仍需实际操作 |
| --- | --- | --- |
| GitHub/Gitee可访问 | 已给出访问与LFS核对规范 | 创建/配置真实仓库及评审权限 |
| 代码结构与规范 | 分离目录、Ruff、Prettier、类型构建与测试入口 | 后续开发继续执行检查 |
| README | 已含介绍、架构、安装运行及团队表格 | 填写真实团队信息 |
| Git提交历史 | 已如实说明无既往历史可提供 | 找回原历史或如实首次导入，保留后续真实提交 |
| 环境配置 | 根.env.example、固定依赖、锁文件、环境文档 | 干净机器与实际平台复现 |
