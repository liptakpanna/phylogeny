import runTrain
import runAccTest
import sys

sizes = [100,200,300,400,500]

version = (sys.argv[1])
if(version == "1"):
    print("Running discount 0.05 tests")
    d = 0.05
    for s in sizes:
        result = runTrain.run(d, s, 0.25, 0.25)
elif(version == "2"):
    print("Running discount 0.5 tests")
    d = 0.5
    for s in sizes:
        result = runTrain.run(d, s, 0.25, 0.25)
elif(version == "3"):
    print("Running discount 0.75 tests")
    d = 0.75
    for s in sizes:
        result = runTrain.run(d, s, 0.25, 0.25)
elif(version == "4"):
    print("Running discount 0.95 tests")
    d = 0.95
    for s in sizes:
        result = runTrain.run(d, s, 0.25, 0.25)

'''
discount = 0.95
sizes = [100,200,300]
balP = [0.1,0.2]
pecP = [0.4,0.3]

for s in sizes:
    for b in balP:
        for p in pecP:
            policy = runTrain.runForAcc(discount,s, b, p)
            runAccTest.acc("pectinate", policy, s, d , b ,p)
            runAccTest.acc("balanced", policy, s, d , b ,p)
            runAccTest.acc("random", policy, s, d , b ,p)


discounts = [0.05,0.5,0.75,0.95]
sizes = [100,200,300,400,500]
for s in sizes:
    for d in discounts:
        result = runTrain.run(d, s, 0.25, 0.25)


policy = runTrain.runForAcc(0.95,100, 0.25, 0.25)
runAccTest.acc("pectinate", policy, s, d , b ,p)
runAccTest.acc("balanced", policy, s, d , b ,p)
runAccTest.acc("random", policy, s, d , b ,p)
'''