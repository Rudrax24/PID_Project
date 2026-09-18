# Compare step response for different G(s)

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl



# Plot 1
G1 = ctrl.TransferFunction(
    [1],
    [1, 1]
)

# Plot 2
G2 = ctrl.TransferFunction(
    [1],
    [1, 2, 1]
)

# Plot 3
G3 = ctrl.TransferFunction(
    [1],
    [1, 0.4, 1]
)

# Plot 4
G4 = ctrl.TransferFunction(
    [1],
    [1, 3, 2]
)

# Time axis
t = np.linspace(0, 20, 1000)

# Responses
t1, y1 = ctrl.step_response(G1, T=t)
t2, y2 = ctrl.step_response(G2, T=t)
t3, y3 = ctrl.step_response(G3, T=t)
t4, y4 = ctrl.step_response(G4, T=t)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(t1, y1, label="G1 = 1/(s+1)")
plt.plot(t2, y2, label="G2 = 1/(s²+2s+1)")
plt.plot(t3, y3, label="G3 = 1/(s²+0.4s+1)")
plt.plot(t4, y4, label="G4 = 1/(s²+3s+2)")

plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("Comparison of Transfer Functions")
plt.legend()
plt.grid()

plt.show()