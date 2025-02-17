from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
from bs4 import BeautifulSoup
import sympy as sp
import sqlite3
import logging
import hashlib
from transformers import pipeline

# --- Настройки ---
app = Flask(__name__)
logging.basicConfig(filename='server.log', level=logging.INFO)

# --- Ограничение запросов (10 подключений/минуту) ---
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["10 per minute"]
)

# --- Инициализация ИИ-моделей ---
nlp_model = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')

# --- База данных для кэша ---
conn = sqlite3.connect('cache.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS tasks (hash TEXT PRIMARY KEY, task TEXT, answer TEXT)')

# --- Шифрование данных (упрощенный пример) ---
def encrypt(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()

# --- Парсинг заданий с ФИПИ ---
def parse_fipi():
    try:
        response = requests.get('https://fipi.ru/ege', headers={'User-Agent': 'MyEgeBot/1.0 (contact: admin@example.com)'})
        soup = BeautifulSoup(response.text, 'html.parser')
        tasks = [task.text for task in soup.select('.task-content')[:10]]  # Пример селектора
        return tasks
    except Exception as e:
        logging.error(f"FIPI parsing error: {e}")
        return []

# --- Генерация задания и ответа ---
def generate_task(subject: str) -> dict:
    # Парсинг или генерация задания
    task = parse_fipi()[0] if 'математика' in subject else "Решите уравнение: 2x^2 + 3x - 5 = 0"
    
    # Генерация ответа через SymPy или ИИ
    if 'математика' in subject:
        x = sp.symbols('x')
        answer = sp.solve(task.split(": ")[1], x)
    else:
        answer = nlp_model(question=task, context="Образовательный контекст")['answer']
    
    return {'task': task, 'answer': answer}

# --- Маршрут для получения задания ---
@app.route('/get_task', methods=['GET'])
@limiter.limit("5 per minute")
def get_task():
    subject = request.args.get('subject', 'математика')
    task_data = generate_task(subject)
    
    # Кэширование
    task_hash = encrypt(task_data['task'])
    cursor.execute('INSERT OR IGNORE INTO tasks VALUES (?, ?, ?)', 
                  (task_hash, task_data['task'], str(task_data['answer'])))
    conn.commit()
    
    return jsonify(task_data)

# --- Запуск сервера ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')  # HTTPS для шифрования
