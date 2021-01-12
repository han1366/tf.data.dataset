import tensorflow as tf
import pathlib
import random
import os
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W,strides,padding):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)


def max_pool(x,ksize,strides,padding):
    # return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    #                       strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.max_pool(x, ksize=ksize,
                           strides=strides, padding=padding)

def model(x):
    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([11,11,3,96])
        b_conv1 = bias_variable([96])
        x = tf.nn.relu(conv2d(x,w_conv1,[1,4,4,1],padding='VALID')+b_conv1)
    with tf.name_scope('maxpool1'):
        x = max_pool(x,[1,3,3,1],[1,2,2,1],'SAME')
    with tf.name_scope('conv2'):
        w_conv2 = weight_variable([5,5,96,256])
        b_conv2 = bias_variable([256])
        x = tf.nn.relu(conv2d(x,w_conv2,[1,1,1,1],'SAME') + b_conv2)
    with tf.name_scope('maxpool2'):
        x = max_pool(x,[1,3,3,1],[1,2,2,1],'VALID')
    with tf.name_scope('conv3'):
        w_conv3 = weight_variable([3,3,256,384])
        b_conv3 = bias_variable([384])
        x = tf.nn.relu(conv2d(x,w_conv3,[1,1,1,1],'SAME') + b_conv3)
    with tf.name_scope('conv4'):
        w_conv4 = weight_variable([3,3,384,384])
        b_conv4 = bias_variable([384])
        x = tf.nn.relu(conv2d(x,w_conv4,[1,1,1,1],'SAME') + b_conv4)
    with tf.name_scope('conv5'):
        w_conv5 = weight_variable([3,3,384,256])
        b_conv5 = bias_variable([256])
        x = tf.nn.relu(conv2d(x,w_conv5,[1,1,1,1],'SAME') + b_conv5)
    with tf.name_scope('maxpool3'):
        x = max_pool(x,[1,3,3,1],[1,2,2,1],'VALID')
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([6 * 6 * 256, 2048])
        b_fc1 = bias_variable([2048])
        h_pool2_flat = tf.reshape(x, [-1, 6 * 6 * 256])
        x = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([2048, 1024])
        b_fc2 = bias_variable([1024])
        x = tf.nn.relu(tf.matmul(x, W_fc2) + b_fc2)
    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([1024, 5])
        b_fc3 = bias_variable([5])
        x = tf.matmul(x, W_fc3) + b_fc3
    return x

def load_and_preprocess_from_path_label(path, label):
    image = tf.io.read_file(path)  # 读取图片
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])  # 原始图片大小为(266, 320, 3)，重设为(192, 192)
    image /= 255.0  # 归一化到[0,1]范围
    return image, label

if __name__ == '__main__':
    # x = tf.random.uniform([2,224,224,3])
    # output = model(x)
    # print("output.shape:",output.shape)

    with tf.name_scope('placeholder_x'):
        x = tf.placeholder(tf.float32,[None,224,224,3])
    with tf.name_scope('placehilder_y'):
        y = tf.placeholder(tf.int32,[None,])
    out_put = model(x)
    print('output:',out_put)
    with tf.name_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_put,labels=y)
    loss = tf.reduce_mean(loss)
    with tf.name_scope('adam'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    with tf.name_scope('accuracy'):
        accuracy = tf.equal(tf.cast(tf.argmax(out_put,1),tf.int32),y)
        accuracy = tf.cast(accuracy,tf.float32)
    with tf.name_scope('output'):
        accuracy = tf.reduce_mean(accuracy)

    #make datasets
    batchsize = 8
    epoch = 100
    train_path = pathlib.Path('flower_data/train')
    val_path = pathlib.Path('flower_data/val')
    train_image_paths = list(train_path.glob('*/*'))
    val_image_paths = list(train_path.glob('*/*'))
    train_length = len(train_image_paths)
    val_length = len(val_image_paths)
    random.shuffle(train_image_paths)
    random.shuffle(val_image_paths)
    train_image_paths = [str(path) for path in train_image_paths]  # 所有图片路径的列表
    val_image_paths = [str(path) for path in val_image_paths]
    print(len(train_image_paths))
    train_label_names = sorted(item.name for item in train_path.glob('*/') if item.is_dir())
    print(train_label_names)
    label_to_index = dict((name, index) for index, name in enumerate(train_label_names))
    print(label_to_index)
    train_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_image_paths]
    train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))

    val_label_names = sorted(item.name for item in val_path.glob('*/') if item.is_dir())
    print(val_label_names)
    label_to_index = dict((name, index) for index, name in enumerate(val_label_names))
    print(label_to_index)
    val_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in val_image_paths]
    val_ds = tf.data.Dataset.from_tensor_slices((val_image_paths, val_image_labels))

    train_ds = train_ds.map(load_and_preprocess_from_path_label).repeat(epoch).shuffle(buffer_size=80).batch(batchsize)
    print(train_ds)
    val_ds = val_ds.map(load_and_preprocess_from_path_label).batch(batchsize)
    print(val_ds)
    result = train_ds.make_one_shot_iterator().get_next()
    val_result = val_ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_dir = 'model/alexnet/'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        for epoch in range(train_length//batchsize*epoch):
            input = sess.run(result)
            if epoch % 100 == 0 :
                print("iter:", epoch)
                train_loss = loss.eval(feed_dict={x: input[0], y: input[1]})
                train_accuracy = accuracy.eval(feed_dict={x: input[0], y: input[1]})
                print("step %d, training accuracy %g,training loss %g" % (epoch, train_accuracy,train_loss))
            train_step.run(feed_dict={x: input[0], y: input[1]})

        saver = tf.train.Saver()
        saver.save(sess, 'model/alexnet/final')

        total_loss = 0
        total_acc = 0
        for epoch in range(val_length//batchsize):
            test_input = sess.run(val_result)
            batch_loss,batch_acc = sess.run([loss,accuracy],feed_dict={x: test_input[0], y: test_input[1]})
            total_loss += batch_loss
            total_acc += batch_acc

        print(" validation loss: %f" % (np.sum(total_loss) / (val_length//batchsize)))
        print(" validation acc: %f" % (np.sum(total_acc) / (val_length//batchsize)))
