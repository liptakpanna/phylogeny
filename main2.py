import runTrain
import runAccTest

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