from flask import Flask;

def create_app():
    app = Flask(__name__);
    app.config['SECRET_KEY'] = 'firstpythonwebapp';
    from .signs import views;
    app.register_blueprint(views, url_prefix='/');
    from .lights import auth;
    app.register_blueprint(auth, url_prefix='/')
    return app;
