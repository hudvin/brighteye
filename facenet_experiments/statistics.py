import pandas as pd
import numpy as np

work_dir = "/tmp"
print("Loading embeddings.")
fname = "{}/embeddings.csv".format(work_dir)
data = pd.read_csv(fname, header=None).as_matrix()
labels = data[:, 0]
embeddings = data[:,1:]
embeddings = embeddings * 100000000000000

label_embedding = zip(labels, embeddings)

joined = np.column_stack((labels, embeddings))

df = pd.DataFrame(joined)
print df.describe

from scipy.spatial import distance_matrix

groups =  df.groupby(0)
for person_name,x in groups:
   # print person_name, x.as_matrix()
    print distance_matrix(x.values[:,1:], x.values[:,1:])


#print grouped

