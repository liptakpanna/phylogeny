import runTrain
import runAccTest

import sarichEnv

acc_py_env = sarichEnv.SarichEnv()
acc_env = tf_py_environment.TFPyEnvironment(acc_py_env)
environment = acc_env

i = 0
while not time_step.is_last():
    action_step = i
    i = i+1
    time_step = environment.step(action_step.action)

'''
discount = 0.95
sizes = [300]
balP = [0.1,0.2]
pecP = [0.4,0.3]

for s in sizes:
    for b in balP:
        for p in pecP:
            policy = runTrain.runForAcc(discount,s, b, p)
            runAccTest.acc("pectinate", policy)
            runAccTest.acc("balanced", policy)
            runAccTest.acc("random", policy)
'''
