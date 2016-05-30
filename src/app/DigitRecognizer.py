from matplotlib import offsetbox
from sklearn import manifold

import numpy as np
import sys
sys.path.append('../../../')
import matplotlib.pyplot as plt


from Util.util.data.DataPrep import *
from CNN.src.ConvolutionalNetwork import *


class DigitRecognizer(object):
    def __init__(self,datafile):
        self.batch_size = 500
        self.initialize_data(datafile)
        self.model = ConvolutionalNetwork(batch_size=self.batch_size,input_shape=(1,28,28))


    def initialize_data(self,datafile):
        [self.train_x,self.train_y], [self.dev_x,self.dev_y], [self.test_x,self.test_y] = DataPrep.load_mnist_data(datafile)

        self.number_of_batches_in_train = self.train_x.get_value(borrow=True).shape[0] // self.batch_size
        self.number_of_batches_in_dev = self.dev_x.get_value(borrow=True).shape[0] // self.batch_size
        self.number_of_batches_in_test = self.test_x.get_value(borrow=True).shape[0] // self.batch_size

    def train_model(self):
        self.model.build_model()
        epochs = 1

        for epoch in range(epochs):
            # For each training example...
            for i in np.random.permutation(int(self.number_of_batches_in_train)):
                self.model.sgd_step(self.train_x[i*self.batch_size:(i+1)*self.batch_size].eval(),self.train_y[i*self.batch_size:(i+1)*self.batch_size].eval())

            test_cost = 0.0
            for i in np.arange(int(self.number_of_batches_in_test)):
                test_cost += self.model.test_model(self.test_x[i*self.batch_size:(i+1)*self.batch_size].eval(),self.test_y[i*self.batch_size:(i+1)*self.batch_size].eval())
            train_cost = 0.0
            for i in np.arange(int(self.number_of_batches_in_train)):
                train_cost += self.model.test_model(self.train_x[i*self.batch_size:(i+1)*self.batch_size].eval(),self.train_y[i*self.batch_size:(i+1)*self.batch_size].eval())

            test_cost = test_cost / self.number_of_batches_in_test
            train_cost = train_cost / self.number_of_batches_in_train
            print("test cost: ")
            print(test_cost)
            print("train cost: ")
            print(train_cost)

            vis_features = []
            vis_predictions = []
            vis_classes = []
            for i in np.arange(int(self.number_of_batches_in_test)):
                features, predictions = self.model.get_visualization_data(self.test_x[i*self.batch_size:(i+1)*self.batch_size].eval())
                print(features.shape)
                print(predictions.shape)
                vis_features.extend(features)
                vis_predictions.extend(predictions)
                vis_classes.extend(self.test_y[i*self.batch_size:(i+1)*self.batch_size].eval())

            x = np.asarray(vis_features)
            y = np.asarray(vis_classes)
            n_samples, n_features = x.shape

            print("Computing t-SNE embedding")
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            X_tsne = tsne.fit_transform(x)

            DigitRecognizer.plot_embedding(X_tsne,y,np.asarray(vis_predictions),
                           "t-SNE embedding of the embedded digits")

            plt.show()

    # Scale and visualize the embedding vectors
    @staticmethod
    def plot_embedding(X,features,targets,classes, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(targets[i]),
                     color=plt.cm.Set1(classes[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(features.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                """imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)"""
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    def step_by_step(self):
        self.model.build_model()
        cost = self.model.sgd_step(self.train_x[0:self.batch_size],np.asanyarray(self.train_y,dtype=np.int32)[0:self.batch_size])
        accuracy = self.model.test_model(self.train_x[0:self.batch_size],np.asanyarray(self.train_y,dtype=np.int32)[0:self.batch_size] )
        print(accuracy)


    def not_mine_training(self):
        self.model.build_model()
        print('... training')
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                           # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
        validation_frequency = min(self.number_of_batches_in_train, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        n_epochs = 200
        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(int(self.number_of_batches_in_train)):

                iter = (epoch - 1) * self.number_of_batches_in_train + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = self.model.sgd_step(self.train_x[minibatch_index*self.batch_size:(minibatch_index+1)*self.batch_size].eval(),self.train_y[minibatch_index*self.batch_size:(minibatch_index+1)*self.batch_size].eval())

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.model.test_model(self.dev_x[i*self.batch_size:(i+1)*self.batch_size].eval(),self.dev_y[i*self.batch_size:(i+1)*self.batch_size].eval()) for i
                                         in range(int(self.number_of_batches_in_dev))]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, int(self.number_of_batches_in_train),
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter



                if patience <= iter:
                    done_looping = True
                    break

        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))


if __name__ == '__main__':
    dr = DigitRecognizer("../../data/mnist.pkl.gz")
    #out = dr.step_by_step()
    dr.train_model()
    #dr.not_mine_training()
    print("The End!")
