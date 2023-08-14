# import libraries
import tensorflow as tf
import tensorflow_datasets as tfdata
from tensorflow_examples.models.pix2pix import pix2pix
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Dataset preparation
dataset, metadata = tfdata.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']


def preprocess(image):
  # resize
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # crop
  image = tf.image.random_crop(image)
  # mirror
  image = tf.image.random_flip_left_right(image)
  return image


# Training set and testing set
train_horses = train_horses.cache().map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(
    1000).batch(1)

train_zebras = train_zebras.cache().map(
    preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(
    1000).batch(1)

horse = next(iter(train_horses))
zebra = next(iter(train_zebras))

 # Import pretrained model
channels = 3

g_generator = pix2pix.unet_generator(channels, norm_type='instancenorm')
f_generator = pix2pix.unet_generator(channels, norm_type='instancenorm')

a_discriminator = pix2pix.discriminator(norm_type='instancenorm', target=False)
b_discriminator = pix2pix.discriminator(norm_type='instancenorm', target=False)

to_zebra = g_generator(horse)
to_horse = f_generator(zebra)
plt.figure(figsize=(8, 8))
contrast = 8

# Define loss functions
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator(real, generated):
  real = loss(tf.ones_like(real), real)

  generated = loss(tf.zeros_like(generated), generated)

  total_disc= real + generated

  return total_disc * 0.5

def generator(generated):
  return loss(tf.ones_like(generated), generated)

# Model training
def train(a_real, b_real):

  with tf.GradientTape(persistent=True) as tape:

    b_fake = g_generator(a_real, training=True)
    a_cycled = f_generator(b_fake, training=True)

    a_fake = f_generator(b_real, training=True)
    b_cycled = g_generator(a_fake, training=True)

    a = f_generator(a_real, training=True)
    b = g_generator(b_real, training=True)

    a_disc_real = a_discriminator(a_real, training=True)
    b_disc_real = b_discriminator(b_real, training=True)

    a_disc_fake = a_discriminator(a_fake, training=True)
    b_disc_fake = b_discriminator(b_fake, training=True)

    # loss calculation
    g_loss = generator(a_disc_fake)
    f_loss = generator(b_disc_fake)

# Model run
for epoch in range(10):
  start = time.time()

  n = 0
  for a_image, b_image in tf.data.Dataset.zip((train_horses, train_zebras)):
    train(a_image, b_image)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

generator(g_generator, horse)