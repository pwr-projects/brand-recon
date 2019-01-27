#%%
from src import *
import seaborn as sns
from collections import Counter
#%%
annotations = get_annotations()
#%%
counts = Counter(annotations.values())

#%%
plot = sns.barplot(list(counts.keys()), list(counts.values()), color='r')
plot.set_xticklabels(list(counts.keys()), rotation=90)
plt.savefig('out.pdf')

#%%
