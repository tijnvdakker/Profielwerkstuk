let ActivationLayer = require('./ActivationLayer');

class TanhLayer extends ActivationLayer {
    constructor() {
        function activation(matrix) {
            return matrix.map(x => Math.tanh(x));
        }
        function activationPrime(matrix) {
            return matrix.map(x => 1 - Math.tanh(x) ** 2);
        }
        
        super(activation, activationPrime);
    }
}

module.exports = TanhLayer;