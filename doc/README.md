# 文档索引

`doc/` 目录已按“总览、接口、数据库、建模”拆分，避免多份文档重复维护同一套内容。
当前总览文档已同步纳入独立 `live` 实时演示模块与 PDF-only 报告导出口径。

## 推荐阅读顺序

1. `chinese_project.md`
   - 项目目标、功能范围、模块职责、部署边界、live 模块定位
2. `api_design.md`
   - 前端、Go 主服务、Python 服务之间的接口契约
3. `database_design.md`
   - MySQL 表结构与数据职责边界
4. `model_data_pipeline.md`
   - 训练与推理依赖的数据口径
5. `schema.sql`
   - 数据库初始化脚本
6. `defense_qa.md`
   - 答辩常见问题、参考回答与临场表达模板

## 当前分工原则

- 总体目标、功能边界、部署说明，只写在 `chinese_project.md`
- 接口字段、错误码、请求响应结构，只写在 `api_design.md`
- 表结构、级联关系、元数据职责，只写在 `database_design.md`
- 训练样本、标签、输入窗口、特征定义，只写在 `model_data_pipeline.md`
