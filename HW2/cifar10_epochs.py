import cifar10_train
import cifar10_eval
import cifar10

import tensorflow as tf


EPOCHS = 100

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train-dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval-data', 'test',
                           """Either 'test' or 'train_eval'.""")

def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)


    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    for i in range(EPOCHS):
        print "Epoch: " + str(i+1) + "/" + str(EPOCHS)
        train_res = cifar10_train.train()
        test_res = cifar10_eval.simple_eval_once(saver, top_k_op)

        print train_res
        print test_res
        #print ('%s: precision @ 1 = %.3f' % (datetime.now(), precision))



if __name__ == '__main__':
  tf.app.run()
