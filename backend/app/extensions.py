from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy


# SQLAlchemy 扩展实例。
# 在这里先创建对象，实际 Flask app 会在 create_app 中通过 db.init_app(app) 绑定。
db = SQLAlchemy()
