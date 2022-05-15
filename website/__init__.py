from flask import Flask;

def create_app():
    app = Flask(__name__);
    app.config['SECRET_KEY'] = 'firstpythonwebapp';
    from .signs import views;
    app.register_blueprint(views, url_prefix='/');
    from .lights import auth;
    app.register_blueprint(auth, url_prefix='/')
    from .object_output import object;
    app.register_blueprint(object, url_prefix='/')
    from .lane import lane;
    app.register_blueprint(lane, url_prefix='/')
    return app;
