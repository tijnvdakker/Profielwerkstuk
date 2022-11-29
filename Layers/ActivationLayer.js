let AbstractLayer = require('./AbstractLayer');

class ActivationLayer extends AbstractLayer {
    constructor(activationFunction, activationFunctionDerivative) {
        super();

        this.activationFunction = activationFunction;
        this.activationFunctionDerivative = activationFunctionDerivative;
    }

    forward(inputs) {
        this.inputs = inputs;
        return this.activationFunction(this.inputs);
    }

    backward(outputGradient) {
        return outputGradient.multiply(this.activationFunctionDerivative(this.inputs));
    }
}

module.exports = ActivationLayer;