import datetime

def logging(text):
    with open("d_log.txt", mode='a') as file:
        file.write('%s,  %s.\n' % 
                (datetime.datetime.now(), text))

def loggingAcc(text):
    with open("a_log.txt", mode='a') as file:
        file.write('%s,  %s.\n' % 
                (datetime.datetime.now(), text))
