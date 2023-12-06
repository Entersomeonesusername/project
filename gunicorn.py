# gunicorn.py

import multiprocessing

timeout = 60
workers = multiprocessing.cpu_count() * 2 + 1
