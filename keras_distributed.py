import tensorflow as tf
import keras

# Define input flags to identify the job and task
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# Create a tensorflow cluster
# Replace localhost with the host names if you are running on multiple hosts
cluster = tf.train.ClusterSpec({"ps": ["192.168.1.165:2221"],
    "worker": ["192.168.1.10:2222", "192.168.1.88:2223"]})

# Start the server
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

# Configurations
batch_size = 128
learning_rate = 0.0005
training_iterations = 100
num_classes = 10
log_frequency = 10

# Load mnist data
def load_data():
    global mnist
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print("Data loaded")

# Create Keras model
def create_model():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    return model

# Create the optimizer
# We cannot use model.compile and model.fit
def create_optimizer(model, targets):
    predictions = model.output
    loss = tf.reduce_mean(
        keras.losses.categorical_crossentropy(targets, predictions))

    # Only if you have regularizers, not in this example
    total_loss = loss * 1.0  # Copy
    for regularizer_loss in model.losses:
        tf.assign_add(total_loss, regularizer_loss)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    # Barrier to compute gradients after updating moving avg of batch norm
    with tf.control_dependencies(model.updates):
        barrier = tf.no_op(name="update_barrier")

    with tf.control_dependencies([barrier]):
        grads = optimizer.compute_gradients(
            total_loss,
            model.trainable_weights)
        grad_updates = optimizer.apply_gradients(grads)

    with tf.control_dependencies([grad_updates]):
        train_op = tf.identity(total_loss, name="train")

    return (train_op, total_loss, predictions)

# Train the model (a single step)
def train(train_op, total_loss, global_step, step):
        import time
        start_time = time.time()
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # perform the operations we defined earlier on batch
        loss_value, step_value = sess.run(
            [train_op, global_step],
            feed_dict={
                model.inputs[0]: batch_x,
                targets: batch_y})

        if step % log_frequency == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            accuracy = sess.run(total_loss,
                                feed_dict={
                                    model.inputs[0]: mnist.test.images,
                                    targets: mnist.test.labels})
            print("Step: %d," % (step_value + 1),
                  " Iteration: %2d," % step,
                  " Cost: %.4f," % loss_value,
                  " Accuracy: %.4f" % accuracy,
                  " AvgTime: %3.2fms" % float(elapsed_time * 1000 / log_frequency))


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    load_data()

    # Assign operations to local server
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        keras.backend.set_learning_phase(1)
        keras.backend.manual_variable_initialization(True)
        model = create_model()
        targets = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
        train_op, total_loss, predictions = create_optimizer(model, targets)

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        init_op = tf.global_variables_initializer()

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             global_step=global_step,
                             logdir="./train_logs",
                             save_model_secs=600,
                             init_op=init_op)

    print("Waiting for other servers")
    with sv.managed_session(server.target) as sess:
        keras.backend.set_session(sess)
        step = 0
        while not sv.should_stop() and step < 100:
            train(train_op, total_loss, global_step, step)
            step += 1

    sv.stop()
    print("done")
