import numpy as np
import matplotlib.pyplot as plt

# Load data (adjust if header or delimiter differs)
data = np.genfromtxt("../CH_output0/MultiCH/anitpeakon/u.csv", delimiter=',', names=True)

x = data['arc_length']  # horizontal domain
# Grab all columns containing time-varying function values
u_cols = [col for col in data.dtype.names if 'function' in col]

plt.figure(figsize=(12, 6))
offset = 0.1  # change this if curves are too close or too far

for i, col in enumerate(u_cols):
    plt.plot(x, data[col] + i * offset, color='black', linewidth=0.6)

plt.axis('off')
plt.tight_layout()
plt.show()
