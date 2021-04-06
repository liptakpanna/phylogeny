import runTrain
import runAccTest
import sys

discount = 0.95
sizes = [100, 300]
balP = [0.1,0.2]
pecP = [0.4,0.3]

version = (sys.argv[1])
if(version == "1"):
    print("Running bal 0.1, pec 0.4 tests")
    balP = 0.1
    pecP = 0.4
elif(version == "2"):
    print("Running bal 0.2, pec 0.4 tests")
    balP = 0.2
    pecP = 0.4
elif(version == "3"):
    print("Running bal 0.1, pec 0.3 tests")
    balP = 0.1
    pecP = 0.3
elif(version == "4"):
    print("Running bal 0.2, pec 0.3 tests")
    balP = 0.2
    pecP = 0.3
elif(version == "5"):
    print("Running bal 0.25, pec 0.25 tests")
    balP = 0.25
    pecP = 0.25

for s in sizes:
        policy = runTrain.runForAcc(discount,s, balP, pecP)
        runAccTest.acc("pectinate", policy)
        runAccTest.acc("balanced", policy)
        runAccTest.acc("random", policy)