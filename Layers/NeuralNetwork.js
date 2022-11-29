let AbstractLayer = require('./AbstractLayer');
let meanSquaredError = require('./helpers').meanSquaredError;
let meanSquaredErrorDerivative = require('./helpers').meanSquaredErrorDerivative;
let Matrix = require('./matrix');
let fs = require('fs');

class NeuralNetwork {
    constructor() {
        this.layers = [];
        this.errors = [];
    }

    static fromJsonFile(fileName) {
        let decodedNeuralNetwork = JSON.parse(fs.readFileSync(fileName, 'utf8'));

        console.log(decodedNeuralNetwork);

        let newNeuralNetwork = new NeuralNetwork();

        newNeuralNetwork.setErrors(decodedNeuralNetwork.errors);
        
        decodedNeuralNetwork.layers.forEach(layer => {
            console.log(JSON.parse(JSON.stringify(layer)));
            let instantiatedLayer = Object.assign(new this[layer.constructorName](), layer.data);
            console.log(instantiatedLayer);
        });

        return new NeuralNetwork(decodedNeuralNetwork.layers, decodedNeuralNetwork.errors);
    }

    toJsonFile(fileName) {
        let test = {errors: this.errors, layers: []};
        this.layers.forEach(layer => {
            let constructorName = layer.constructor.name;
            test.layers.push({constructorName, layer});
        })

        fs.writeFileSync(fileName, JSON.stringify(test));
    }

    setErrors(errors) {
        this.errors = errors;
    }

    addLayer(layer) {
        if (!layer instanceof AbstractLayer) {
            throw new Error("Layer has to be extended from the AbstractLayer base class");
        }

        this.layers.push(layer);
    }

    train(inputs, answers, iterations) {
        for (let i = 0; i < iterations; i++) {
            for (let j = 0; j < inputs.length; j++) {
                let output = this.forward(Matrix.fromArray(inputs[j]));
    
                this.backward(answers[j], output);
            }
        }
    }

    testPerformance(testInputs, testAnswers) {
        let totalTestInputs = testInputs.length;
        let testsCorrect = 0;

        testInputs.forEach((testInput, testInputIndex) => {
            let output = this.forward(Matrix.fromArray(testInput));

            let maxPredicted = 0;
            let maxPredictedIndex = 0;

            output.data.forEach((item, index) => {
                if (item[0] > maxPredicted) {
                    maxPredicted = item[0];
                    maxPredictedIndex = index;
                }
            });

            let maxAnswerIndex = testAnswers[testInputIndex].indexOf(Math.max(...testAnswers[testInputIndex]));

            if (maxAnswerIndex === maxPredictedIndex) {
                testsCorrect += 1;
            }
        });

        return testsCorrect / totalTestInputs * 100;
    }

    forward(output) {
        this.layers.forEach(layer => {
            output = layer.forward(output);
        });

        return output;
    }

    backward(answers, predictions) {
        this.errors.push(meanSquaredError(Matrix.fromArray(answers), predictions));

        let gradient = meanSquaredErrorDerivative(Matrix.fromArray(answers), predictions);

        this.layers.slice().reverse().forEach(layer => {
            gradient = layer.backward(gradient);
        });
    }

    serialize() {
        return JSON.stringify(this);
    }
    
}

module.exports = NeuralNetwork;