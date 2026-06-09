bind = "unix:/var/www/shunt_classification_and_ligation/shunt.sock"
workers = 1
worker_class = "sync"
timeout = 120
keepalive = 5
errorlog = "/var/log/shunt_classification_and_ligation/gunicorn-error.log"
accesslog = "/var/log/shunt_classification_and_ligation/gunicorn-access.log"
loglevel = "info"
