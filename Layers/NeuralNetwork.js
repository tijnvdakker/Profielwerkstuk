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
        this.performances = [];
        this.wronglyGuessedNumberCounts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    }

    static fromJsonFile(fileName) {
        let decodedNeuralNetwork = JSON.parse(fs.readFileSync(fileName, 'utf8'));

        let newNeuralNetwork = new NeuralNetwork();

        newNeuralNetwork.setErrors(decodedNeuralNetwork.errors);
        newNeuralNetwork.setPerformances(decodedNeuralNetwork.performances);
        newNeuralNetwork.setWronglyGuessedNumberCounts(decodedNeuralNetwork.wronglyGuessedNumberCounts);

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

    setPerformances(performances) {
        this.performances = performances ?? [];
    }

    setWronglyGuessedNumberCounts(wronglyGuessedNumberCounts) {
        this.wronglyGuessedNumberCounts = wronglyGuessedNumberCounts ?? [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
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
                if (j % 1000 == 0) {
                    let performance = this.testPerformance(this.testInputs, this.testOutputs);
                    this.performances.push(performance);
                    this.wronglyGuessedNumberCounts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                    console.log(performance);
                }
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
            } else {
                this.wronglyGuessedNumberCounts[maxAnswerIndex] += 1;
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

    setTests(testInputs, testOutputs) {
        this.testInputs = testInputs;
        this.testOutputs = testOutputs;
    }
}

module.exports = NeuralNetwork;