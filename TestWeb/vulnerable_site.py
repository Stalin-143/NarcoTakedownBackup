from flask import Flask, request, render_template_string, session
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'secret'

def init_db():
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, content TEXT)")
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return '<h1>Welcome to the Buggy Website!</h1><a href="/login">Login</a> <a href="/register">Register</a>'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()
        cursor.execute(f"INSERT INTO users (username, password) VALUES ('{username}', '{password}')")  # SQL Injection
        conn.commit()
        conn.close()
        return 'Registered successfully! <a href="/login">Login</a>'
    return '<form method="post">Username: <input type="text" name="username"><br>Password: <input type="password" name="password"><br><input type="submit" value="Register"></form>'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'")  # SQL Injection
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user'] = username
            return f'Logged in as {username}! <a href="/messages">Go to messages</a>'
        return 'Invalid credentials!'
    return '<form method="post">Username: <input type="text" name="username"><br>Password: <input type="password" name="password"><br><input type="submit" value="Login"></form>'

@app.route('/messages', methods=['GET', 'POST'])
def messages():
    if 'user' not in session:
        return 'You need to login first! <a href="/login">Login</a>'
    if request.method == 'POST':
        message = request.form['message']
        conn = sqlite3.connect('test.db')
        cursor = conn.cursor()
        cursor.execute(f"INSERT INTO messages (content) VALUES ('{message}')")  # SQL Injection
        conn.commit()
        conn.close()
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages")
    messages = cursor.fetchall()
    conn.close()
    message_display = "<br>".join([msg[1] for msg in messages])  # XSS Vulnerability
    return f'<form method="post">Message: <input type="text" name="message"><br><input type="submit" value="Post"></form><br>{message_display}'

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(os.path.join('uploads', file.filename))  # Arbitrary File Upload
    return 'File uploaded!'

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
