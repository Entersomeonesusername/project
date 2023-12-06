# gunicorn.py

import multiprocessing

timeout = 300
workers = multiprocessing.cpu_count() * 2 + 1
