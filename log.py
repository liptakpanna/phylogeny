import datetime

def logging(text):
    with open("log.txt", mode='a') as file:
        file.write('%s,  %s.\n' % 
                (datetime.datetime.now(), text))