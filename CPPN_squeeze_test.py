from squeeze import binary_filter_np, reduce_precision_np, median_filter_np
import keras
import os
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import numpy as np
import matplotlib.pyplot as plt

if keras.backend.image_dim_ordering() != 'tf':
    keras.backend.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

#Load Model (Trained on MNIST and two iterations of CPPN examples)
model = keras.models.load_model('./checkpoint.h5')
predictions = model(x)

#Load CPPN examples generated to trick model that was just loaded
print('')
print('Loading CPPN examples...')
number_of_examples = len(os.walk('./CPPN_data').next()[2])
CPPN_examples = np.zeros([number_of_examples, 28, 28, 1, 11])
files_in_dir = os.listdir('./CPPN_data')
k=0
for files in files_in_dir:
    data = np.load(os.path.join('./CPPN_data',files))
    array = data['features']
    array = array[0,0,:,:]
    CPPN_examples[k,:,:,0,0] = array
    target = data['targets'][0]
    one_hot = [0,0,0,0,0,0,0,0,0,0,0]
    one_hot[target] = 1
    CPPN_examples[k,0,0,0,:] = one_hot
    k=k+1
np.random.shuffle(CPPN_examples)


#Test model on CPPN examples
print('')
print('Testing model on CPPN examples...')
correct_count=0
wrong_count=0
bad_example_marker=789
for i in range(number_of_examples):
    example = CPPN_examples[i:i+1,:,:,:,0]
    target = np.where(CPPN_examples[i,0,0,0,:]==1)[0][0]
    pred = predictions.eval(session=sess, feed_dict={x: example})[0]
    if pred[target]>=0.85:
        correct_count = correct_count + 1
    if pred[target]<.85:
        CPPN_examples[i,0,0,:,0]=bad_example_marker
        wrong_count=wrong_count+1
trick_rate = float(correct_count)*100/number_of_examples
number_of_examples=number_of_examples-wrong_count

print('')
print('Model tricked by {:.2f}% of CPPN examples'.format(trick_rate))  
print('CPPN examples in the dataset which did not successfully trick the model were marked and wont be considered in subsequent tests (I cant figure out why any of them would fail, though)')

print('')
print("Testing model on binary filtered CPPN examples...")
same_classification = 0
new_classification_not_noise = 0
no_classification = 0
noise_classification = 0
for i in range(np.shape(CPPN_examples)[0]):
    example = CPPN_examples[i:i+1,:,:,:,0]
    if example[0,0,0,:]!=bad_example_marker:
        target = np.where(CPPN_examples[i,0,0,0,:]==1)[0][0]
        example = binary_filter_np(example)
        pred = predictions.eval(session=sess, feed_dict={x: example})[0]
        most_likely_class = np.argmax(pred)
        if pred[most_likely_class]>=0.85:
            if most_likely_class == target:
                same_classification=same_classification+1
            if most_likely_class!=target:
                if most_likely_class==10:
                    noise_classification=noise_classification+1
                if most_likely_class!=10:
                    new_classification_not_noise=new_classification_not_noise+1
        if pred[most_likely_class]<0.85:
            no_classification=no_classification+1

print('')
print('Same Classification: {:.2f}%').format(float(same_classification)*100/number_of_examples)
print('New Classification Other Than Noise: {:.2f}%').format(float(new_classification_not_noise)*100/number_of_examples)
print('No Classification: {:.2f}%').format(float(no_classification)*100/number_of_examples)
print('Noise Classification: {:.2f}%').format(float(noise_classification)*100/number_of_examples)

print('')
print("Testing model on median filtered CPPN examples...")
median_param = 10
same_classification = 0
new_classification_not_noise = 0
no_classification = 0
noise_classification = 0
for i in range(np.shape(CPPN_examples)[0]):
    example = CPPN_examples[i:i+1,:,:,:,0]
    if example[0,0,0,:]!=bad_example_marker:
        target = np.where(CPPN_examples[i,0,0,0,:]==1)[0][0]
        example = median_filter_np(example,median_param) # second arguement is width of sliding window
        pred = predictions.eval(session=sess, feed_dict={x: example})[0]
        most_likely_class = np.argmax(pred)
        if pred[most_likely_class]>=0.85:
            if most_likely_class == target:
                same_classification=same_classification+1
            if most_likely_class!=target:
                if most_likely_class==10:
                    noise_classification=noise_classification+1
                if most_likely_class!=10:
                    new_classification_not_noise=new_classification_not_noise+1
        if pred[most_likely_class]<0.85:
            no_classification=no_classification+1

print('')
print('Same Classification: {:.2f}%').format(float(same_classification)*100/number_of_examples)
print('New Classification Other Than Noise: {:.2f}%').format(float(new_classification_not_noise)*100/number_of_examples)
print('No Classification: {:.2f}%').format(float(no_classification)*100/number_of_examples)
print('Noise Classification: {:.2f}%').format(float(noise_classification)*100/number_of_examples)

print('')
print("Testing model on reduced precision CPPN examples...")
precision_param=100
same_classification = 0
new_classification_not_noise = 0
no_classification = 0
noise_classification = 0
for i in range(np.shape(CPPN_examples)[0]):
    example = CPPN_examples[i:i+1,:,:,:,0]
    if example[0,0,0,:]!=bad_example_marker:
        target = np.where(CPPN_examples[i,0,0,0,:]==1)[0][0]
        example = reduce_precision_np(binary_filter_np(example),precision_param)
        pred = predictions.eval(session=sess, feed_dict={x: example})[0]
        most_likely_class = np.argmax(pred)
        if pred[most_likely_class]>=0.85:
            if most_likely_class == target:
                same_classification=same_classification+1
            if most_likely_class!=target:
                if most_likely_class==10:
                    noise_classification=noise_classification+1
                if most_likely_class!=10:
                    new_classification_not_noise=new_classification_not_noise+1
        if pred[most_likely_class]<0.85:
            no_classification=no_classification+1

print('')
print('Same Classification: {:.2f}%').format(float(same_classification)*100/number_of_examples)
print('New Classification Other Than Noise: {:.2f}%').format(float(new_classification_not_noise)*100/number_of_examples)
print('No Classification: {:.2f}%').format(float(no_classification)*100/number_of_examples)
print('Noise Classification: {:.2f}%').format(float(noise_classification)*100/number_of_examples)

#Visualize
print('')
print('Visualizing examples...')
print('')

fig = plt.figure()
random_examples = np.random.random_integers(number_of_examples, size=(1,5))[0]
ax = [0]*4*np.shape(random_examples)[0]
for num, i in enumerate(random_examples):
    example = CPPN_examples[i:i+1,:,:,:,0]
    while example[0,0,0,:]==bad_example_marker or np.count_nonzero(random_examples==random_examples[num])!=1:
            random_examples[num]=np.random.random_integers(number_of_examples)
            example = CPPN_examples[random_examples[num]:random_examples[num]+1,:,:,:,0]
    binary = binary_filter_np(example)
    median = median_filter_np(example,median_param)
    reduced_precision = reduce_precision_np(binary,precision_param)
    ax[num*4] = fig.add_subplot(np.shape(random_examples)[0], 4, num*4+1, aspect='equal')
    ax[num*4].imshow(example[0,:,:,0], cmap='gray_r', interpolation='nearest')
    ax[num*4].axes.get_xaxis().set_visible(False)
    ax[num*4].axes.get_yaxis().set_visible(False)

    ax[num*4+1] = fig.add_subplot(np.shape(random_examples)[0], 4, num*4+2, aspect='equal')
    ax[num*4+1].imshow(binary[0,:,:,0], cmap='gray_r', interpolation='nearest')
    ax[num*4+1].axes.get_xaxis().set_visible(False)
    ax[num*4+1].axes.get_yaxis().set_visible(False)

    ax[num*4+2] = fig.add_subplot(np.shape(random_examples)[0], 4, num*4+3, aspect='equal')
    ax[num*4+2].imshow(median[0,:,:,0], cmap='gray_r', interpolation='nearest')
    ax[num*4+2].axes.get_xaxis().set_visible(False)
    ax[num*4+2].axes.get_yaxis().set_visible(False)

    ax[num*4+3] = fig.add_subplot(np.shape(random_examples)[0], 4, num*4+4, aspect='equal')
    ax[num*4+3].imshow(reduced_precision[0,:,:,0], cmap='gray_r', interpolation='nearest')
    ax[num*4+3].axes.get_xaxis().set_visible(False)
    ax[num*4+3].axes.get_yaxis().set_visible(False)

    if num==0:
       ax[num*4].set_title('CPPN')
       ax[num*4+1].set_title('Binary')
       ax[num*4+2].set_title('Median')
       ax[num*4+3].set_title('Reduced Precision')   

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

fig.tight_layout()
fig.subplots_adjust(top=0.93)

plt.show()
