import numpy as np
import gzip
import os

# specify the path to the zipped file.
os.chdir('/mnt/c/Users/amsj1/OneDrive - University of Tennessee/2nd_year/BZAN554_deep_learning/group_assignment_1')

def parse(path):
    """
    Function to read in the VERY LARGE dataset and yield it as a generator for
    memory efficiency. Takes one argument which is the path to the file being
    read in. This path is set abose using 'os.chdir'.
    """
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

# print the first 10 rows of the data; includes the title of product and
# categories it belongs to.
i = 0
df = {}
for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
    i += 1
    X = np.array(d['title'])
    print('X (title):\n')
    print(X)
    Y = np.array(d['category'])
    print('\nY (category):\n')
    print(Y)
    if i == 10:
        break