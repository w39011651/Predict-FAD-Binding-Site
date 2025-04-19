import time

def log(msg, end='\n'):
    if type(msg) != str:
        msg = str(msg)
    print(time.strftime('[%Y-%m-%d %H:%M:%S]') + msg, end=end)