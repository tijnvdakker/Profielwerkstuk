let AbstractLayer = require('./AbstractLayer');
let Matrix = require('./matrix');

class DenseLayer extends AbstractLayer {
    constructor(input_nodes, output_nodes) {
        super();

        this.learningRate = 0.15;

        this.weights = new Matrix(output_nodes, input_nodes).randomize();

        this.biases = new Matrix(output_nodes, 1).randomize();
    }

    setWeights(weights) {
        this.weights = weights;
    }

    setBiases(biases) {
        this.biases = biases;
    }

    setLearningRate(learningRate) {
        this.learningRate = learningRate;
    }

    forward(inputs) {
        this.inputs = inputs;
        return Matrix.multiply(this.weights, this.inputs).add(this.biases);
    }

    backward(outputGradient) {
        let weightGradient = Matrix.multiply(outputGradient, Matrix.transpose(this.inputs));
        let inputGradient = Matrix.multiply(Matrix.transpose(this.weights), outputGradient);
        this.weights = Matrix.subtract(this.weights, weightGradient.multiply(this.learningRate));
        this.biases = Matrix.subtract(this.biases, outputGradient.multiply(this.learningRate));
        return inputGradient;
    }
}

module.exports = DenseLayer;