import matplotlib.pyplot as plt
import pandas as pd

# Manual transcription of the data from the image
data = {
    'Mic4': {'mix': 1.449982727, 'CCAMS': 1.595263085, 'IGCRN_parallel': 1.700001196,
             'ICCRN_parallel': 1.763082463, 'TFGridNet_parallel': 2.36565685},
    'Mic8': {'mix': 1.441767996, 'CCAMS': 1.585514086, 'IGCRN_parallel': 1.887017688,
             'ICCRN_parallel': 1.792652323, 'TFGridNet_parallel': 2.503386107},
    'Mic12': {'mix': 1.445470237, 'CCAMS': 1.580354066, 'IGCRN_parallel': 1.855842463,
              'ICCRN_parallel': 1.791901023, 'TFGridNet_parallel': 2.537425624},
    'Mic16': {'mix': 1.436155198, 'CCAMS': 1.575616778, 'IGCRN_parallel': 1.730683548,
              'ICCRN_parallel': 1.745698153, 'TFGridNet_parallel': 2.480484151}
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Create a plot
plt.figure(figsize=(10, 6))
for method in df.index:
    plt.plot(df.columns, df.loc[method], marker='o', label=method)

plt.title('PESQ Scores by Microphone Configuration')
plt.xlabel('Microphone Configuration')
plt.ylabel('PESQ Score')
plt.legend(title='Method')
plt.grid(True)
plt.show()
