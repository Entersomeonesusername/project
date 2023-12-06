# gunicorn.py

import multiprocessing
worker_class = 'gevent'

timeout = 300
workers = multiprocessing.cpu_count() * 2 + 1
