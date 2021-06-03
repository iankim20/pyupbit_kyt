import numpy as np

var1 = 2>1
var2 = 4>1
var3 = 5>1
var4 = 4<1

unlock = np.zeros(8)
unlock[3]=1

print(unlock)

if (6000>5000) & bool(unlock[3]):
    print("hi")