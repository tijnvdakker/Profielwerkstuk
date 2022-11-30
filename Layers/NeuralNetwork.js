let AbstractLayer = require('./AbstractLayer');
let DenseLayer = require('./DenseLayer');
let TanhLayer = require('./TanhLayer');
let SigmoidLayer = require('./SigmoidLayer');
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

        let newNeuralNetwork = new NeuralNetwork();

        newNeuralNetwork.setErrors(decodedNeuralNetwork.errors);

        decodedNeuralNetwork.layers.forEach(layer => {
             let instantiatedLayer = eval("new " + layer.constructorName + "()");

             if (layer.constructorName == 'DenseLayer') {
                instantiatedLayer.setLearningRate(layer.learningRate);

                let newWeights = new Matrix(layer.weights.rows, layer.weights.cols);
                newWeights.setData(layer.weights.data);
                instantiatedLayer.setWeights(newWeights);

                let newBiases = new Matrix(layer.biases.rows, layer.biases.cols);
                newBiases.setData(layer.biases.data);
                instantiatedLayer.setBiases(newBiases);
             }
             
             newNeuralNetwork.addLayer(instantiatedLayer);
        });
        
        return newNeuralNetwork;
    }

    toJsonFile(fileName) {
        this.layers.forEach((layer, index) => {
            let constructorName = layer.constructor.name;
            layer.constructorName = constructorName;
            this.layers[index] = layer;
        })

        fs.writeFileSync(fileName, JSON.stringify(this));
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

            console.log(output);
            console.log(testAnswers[testInputIndex]);

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
                console.log("CORRECT");
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

    predict(input) {
        return this.forward(Matrix.fromArray(input));
    }
}

module.exports = NeuralNetwork;