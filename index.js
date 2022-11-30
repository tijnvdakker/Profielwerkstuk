let DenseLayer = require('./Layers/DenseLayer');
let Matrix = require('./Layers/matrix');
let TanhLayer = require('./Layers/TanhLayer');
let NeuralNetwork = require('./Layers/NeuralNetwork');
let SigmoidLayer = require('./Layers/SigmoidLayer');

// let inputs = [[1, 0], [0, 1], [1, 1], [0, 0]];
// let outputs = [[1], [1], [0], [0]];

let network = [
    new DenseLayer(784, 200),
    new TanhLayer(),
    new DenseLayer(200, 10),
    new SigmoidLayer()
];

let nn = new NeuralNetwork();
nn.addLayer(new DenseLayer(784, 1));
nn.addLayer(new SigmoidLayer());
nn.addLayer(new DenseLayer(1, 10));
nn.addLayer(new SigmoidLayer());

let mnist = require('mnist');
let set = mnist.set(100, 10);

let inputs = set.training.map(i => i.input);
let outputs = set.training.map(i => i.output);

nn.train(inputs, outputs, 1);

let testInputs = set.test.map(i => i.input);
let testOutputs = set.test.map(i => i.output);

let performance = nn.testPerformance(testInputs, testOutputs);

console.log(performance);

nn = new NeuralNetwork();
nn.addLayer(new DenseLayer(2, 2));
nn.addLayer(new TanhLayer());

nn.toJsonFile('test.txt');

console.log(NeuralNetwork.fromJsonFile('test.txt'));

