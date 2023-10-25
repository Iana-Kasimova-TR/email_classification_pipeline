sql_uri=sqlite:///email.db

build:
	docker build -f docker/Dockerfile -t flask_email_app .

test:
	SQLALCHEMY_DATABASE_URI=$(sql_uri) py.test
	black src/

run:
	docker run -p 5050:5050 -v email.db:email.db flask_email_app