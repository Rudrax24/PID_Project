import control as ctrl
import matplotlib.pyplot as plt
import numpy as np

# G(s) = 1/(s+1)

num = [1]
denum = [1, 1]

G = ctrl.TransferFunction(num, denum)

time, response = ctrl.step_response(G)

plt.plot(time, response)
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("G(s) = 1/(s+1)")
plt.grid()
plt.show()