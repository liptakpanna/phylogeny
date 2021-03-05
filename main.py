import runDiscountTest
import runAccTest

discounts = [0.05,0.5,0.75,0.95]
sizes = [100,200,300,400,500]
for s in sizes:
    for d in discounts:
        result = runDiscountTest.run(d, s, 0.25, 0.25)

'''
policy = runDiscountTest.runForAcc(0.95,100, 0.25, 0.25)
runAccTest.acc("pectinate", policy)
runAccTest.acc("balanced", policy)
runAccTest.acc("random", policy)
'''