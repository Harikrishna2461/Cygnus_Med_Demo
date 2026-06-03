import multiprocessing

bind = "unix:/var/www/chiva/chiva.sock"
workers = 2
worker_class = "sync"
timeout = 120
keepalive = 5
errorlog = "/var/log/chiva/gunicorn-error.log"
accesslog = "/var/log/chiva/gunicorn-access.log"
loglevel = "info"
