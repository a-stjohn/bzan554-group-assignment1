# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import gzip
import os
import tensorflow as tf

# specify the path to the zipped file.
os.chdir('/mnt/c/Users/amsj1/OneDrive - University of Tennessee/2nd_year/BZAN554_deep_learning/bzan554-group-assignment1')


# %%
def parse(path):
    """
    Function to read in the VERY LARGE dataset and yield it as a generator for
    memory efficiency. Takes one argument which is the path to the file being
    read in. This path is set abose using 'os.chdir'.
    """
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield eval(l)

def yield_columns():
    """
    Parse the large data set and return a generator of a tuple with X
    (product title) and Y (product categories). Each item in the generator is
    a numpy array. Column X is at postion 0 and column Y is at position 1.

    For example, to access X (product title), the code would be
    `next(yield_columns())[0]`.
    """
    for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
        X = np.array(d['title'])
        Y = np.array(d['category'])

        yield X, Y


# %%
# set up model
inputs = tf.keras.layers.Input(shape = (1,))
hidden1 = tf.keras.layers.Dense(
    units = 2,
    activation = 'sigmoid',
    name = 'hidden1'
)(inputs)
hidden2 = tf.keras.layers.Dense(
    units = 2,
    activation = 'sigmoid',
    name = 'hidden2'
)(hidden1)
output = tf.keras.layers.Dense(
    units = 2,
    activation = 'sigmoid',
    name = 'output'
)(hidden2)

# create & compile the model
model = tf.keras.Model(inputs = inputs, outputs = output)
model.compile(
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)
)


# %%
for i in range(1000): # number of epochs
    for j in yield_columns(): # number of instances
        model.fit(
            x=next(yield_columns())[0],
            y=next(yield_columns())[1],
            epochs=1,
            batch_size=1
        )


