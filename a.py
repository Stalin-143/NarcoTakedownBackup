import requests
from threading import Thread

def send_request():
    try:
        requests.get("http://127.0.0.1:5000", timeout=1)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

threads = []
for _ in range(200000):
    t = Thread(target=send_request)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
