from app import create_app
from config import Config

# Flask 应用入口。
# create_app 负责完成配置加载、数据库初始化、目录创建和路由注册；
# main.py 只保留启动逻辑，便于开发环境直接运行后端服务。
app = create_app(Config)

if __name__ == "__main__":
    app.run(host=Config.APP_HOST, port=Config.APP_PORT, debug=Config.DEBUG)
