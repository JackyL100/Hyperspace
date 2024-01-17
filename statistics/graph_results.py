import numpy as np
import matplotlib.pyplot as plt

filePath = input()
file = open(filePath, "r")

results = []
eps = []
while True:
    line = file.readline()
    if not line:
        break
    parts = line.split()
    score = ""
    ep = ""
    if parts[0] == "Eval":
        score = parts[2]
    else:
        ep = parts[3]
        ep = ep[:-1]
        score = parts[4]
        results.append(float(score))
        eps.append(int(ep))

plt.plot(eps, results)
plt.ylabel('Reward')
plt.xlabel('Episodes')
plt.show()