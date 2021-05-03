import runTrain
import runAccTest
import time
import dendropy
'''
tns = dendropy.TaxonNamespace()
start_time = time.time()
for i in range(1,100):
  pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
        src=open("test_set/distances/random/dist"+str(i+1) +".csv"),
        delimiter=",")
  tree = pdm.nj_tree()

print("--- %s seconds ---" % (time.time() - start_time))


discount = 0.95
sizes = [100,300,500]
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

#discounts = [0.05,0.5,0.75,0.95]
discounts = [0.5,0.95]
sizes = [100,300,500]
for s in sizes:
    for d in discounts:
        result = runTrain.run(d, s, 0.25, 0.25)

'''
policy = runTrain.runForAcc(0.95,100, 0.25, 0.25)
runAccTest.acc("pectinate", policy)
runAccTest.acc("balanced", policy)
runAccTest.acc("random", policy)
'''