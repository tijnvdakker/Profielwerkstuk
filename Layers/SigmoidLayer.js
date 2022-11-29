let ActivationLayer = require('./ActivationLayer');

class SigmoidLayer extends ActivationLayer {
    constructor() {
        function activation(matrix) {
            return matrix.map(x => 1 / (1 + Math.exp(-x)));
        }
        function activationPrime(matrix) {
            return matrix.map(x => x * (1 - x));
        }
        
        super(activation, activationPrime);
    }
}

module.exports = SigmoidLayer;