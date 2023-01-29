#!/usr/bin/env python
import os
import sys
from flask import Flask, render_template, request, redirect, session, url_for
from flask_mysqldb import MySQL

import MySQLdb

app = Flask(__name__)
app.secret_key="1234"

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "root@123"
app.config["MYSQL_DB"] = "myhospital"

db=MySQL(app)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method== 'POST':
        if 'username' in request.form and 'password' in request.form:
            username=request.form['username']
            password=request.form['password']
            cursor =db.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute("SELECT * FROM myhospital WHERE Username=%s AND Password=%s",(username,password))
            info=cursor.fetchone()
            if info['Username']== username and info['Password']== password:
                return "login successful"
            else:
                return "login unsuccessful, Please register"

        render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myHospital.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
