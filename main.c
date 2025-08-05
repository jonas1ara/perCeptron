#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utilities/img.h"
#include "neuron/activations.h"
#include "neuron/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"

int main() {
	srand(time(NULL));

	// TRAINING
	// Uncomment the following lines to train the neural network
	// and save it to a file. Make sure to adjust the number of images
	// and the file path as needed.
	// int number_imgs = 10000;
	// Img** imgs = csv_to_imgs("data/mnist_train.csv", number_imgs);
	// NeuralNetwork* net = network_create(784, 300, 10, 0.1);
	// network_train_batch_imgs(net, imgs, number_imgs);
	// network_save(net, "test");

	// // PREDICTION
	// Uncomment the following lines to load the trained network
	// and predict the labels for a set of images.
	int number_imgs = 3000;
	Img** imgs = csv_to_imgs("data/mnist_train.csv", number_imgs);
	NeuralNetwork* net = network_load("test");
	double score = network_predict_imgs(net, imgs, 1000);
	printf("Score: %1.5f \n", score);

	imgs_free(imgs, number_imgs);
	network_free(net);
	return 0;
}