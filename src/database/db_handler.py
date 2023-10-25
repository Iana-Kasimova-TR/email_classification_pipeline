from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Sample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input = db.Column(db.String(100), nullable=False)
    result = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"Sample('{self.input}', '{self.result}')"


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()


def save_to_database(input: str, result: str) -> bool:
    """
    input: input text
    output: result of classification
    save preditions to database
    """
    sample = Sample(input=input, result=result)
    try:
        db.session.add(sample)
        db.session.commit()
    except Exception as e:
        print(e)
        return False
    return True


def get_all_predictions() -> list:
    """
    output: list of all predictions
    """
    samples = Sample.query.all()
    return samples
