import runTrain
import runAccTest
import sarichEnv

x = sarichEnv.SarichEnv()

discount = 0.95
s = 500
b = 0.1
p = 0.4

policy = runTrain.runForAcc(discount,s, b, p)
runAccTest.acc("pectinate", policy)
runAccTest.acc("balanced", policy)
runAccTest.acc("random", policy)

runAccTest.sarichEnv(policy)

'''
discounts = [0.05,0.5,0.75,0.95]
sizes = [100,200,300,400,500]
for s in sizes:
    for d in discounts:
        result = runTrain.run(d, s, 0.25, 0.25)


policy = runTrain.runForAcc(0.95,100, 0.25, 0.25)
runAccTest.acc("pectinate", policy)
runAccTest.acc("balanced", policy)
runAccTest.acc("random", policy)
'''