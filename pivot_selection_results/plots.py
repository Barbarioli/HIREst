import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# load ratios_8_window and ratios_16_window
ratios_8_window = np.loadtxt('pivot_selection_results/ratios_8_window.csv',delimiter=',').astype(np.float32)
ratios_8_row = np.loadtxt('pivot_selection_results/ratios_8_row.csv',delimiter=',').astype(np.float32)
diff = ratios_8_window[:,0]-ratios_8_row[:, 0]
# seaborn histplot with density=True between ratios_8_window[:,0] and ratios_8_row[:,0] 
sns.histplot(diff, color='blue', kde=True)

plt.title("Window flattening vs. Row flattening (w=8)")
plt.xlabel('Window CR - row CR')

plt.savefig('pivot_selection_results/8_window_row_CR_difference.png')
plt.clf()
ratios_16_window = np.loadtxt('pivot_selection_results/ratios_16_window.csv',delimiter=',').astype(np.float32)

header = {0:'window first', 1:'window random', 2:'hist'}
counts = np.zeros(3)

# calcualte argmin over ratios_16_window columns
argmin = np.argmin(ratios_16_window, axis=1)

for mn in argmin:
    counts[mn] += 1

counts /= argmin.shape[0]

plt.bar(header.values(), counts)

plt.title("Comparison of Pivot Selection Methods (w=16)")
plt.ylabel('Frequency of CR Win (decimal)')
plt.xlabel('Pivot Selection Method fast->slow')

plt.savefig('pivot_selection_results/16_window_pivot_selection.png')


plt.clf()

# stacked barplot for ratios_8_window and  ratios_4_window 
ratios_2_window = np.loadtxt('pivot_selection_results/ratios_2_window.csv',delimiter=',').astype(np.float32)
ratios_4_window = np.loadtxt('pivot_selection_results/ratios_4_window.csv',delimiter=',').astype(np.float32)
ratios_8_window = np.loadtxt('pivot_selection_results/ratios_8_window.csv',delimiter=',').astype(np.float32)

header = {0:'window first', 1:'window random'}
counts = np.zeros((3,2))

argmin_2 = np.argmin(ratios_2_window, axis=1)
argmin_4 = np.argmin(ratios_4_window, axis=1)
argmin_8 = np.argmin(ratios_8_window, axis=1)

for mn in argmin_2:
    counts[0,mn] += 1

for mn in argmin_4:
    counts[1,mn] += 1

for mn in argmin_8:
    counts[2,mn] += 1

counts /= argmin_4.shape[0]

plt.bar(header.values(), counts[0], label='w=2', color='blue')
plt.bar(header.values(), counts[1], bottom=counts[0], label='w=4', color='green')
plt.bar(header.values(), counts[2], bottom=counts[0]+counts[1], label='w=8', color='orange')

plt.title("Comparison of Pivot Selection Methods (w=2,4,8)")
plt.ylabel('Frequency of CR Win (decimal)')
plt.xlabel('Pivot Selection Method fast->slow')
plt.legend()

plt.savefig('pivot_selection_results/4_8_window_pivot_selection.png')

