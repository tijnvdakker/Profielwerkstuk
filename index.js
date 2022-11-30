let DenseLayer = require('./Layers/DenseLayer');
let TanhLayer = require('./Layers/TanhLayer');
let NeuralNetwork = require('./Layers/NeuralNetwork');
let SigmoidLayer = require('./Layers/SigmoidLayer');

// let inputs = [[1, 0], [0, 1], [1, 1], [0, 0]];
// let outputs = [[1], [1], [0], [0]];

let nn = new NeuralNetwork();
nn.addLayer(new DenseLayer(784, 100));
nn.addLayer(new SigmoidLayer());
nn.addLayer(new DenseLayer(100, 10));
nn.addLayer(new SigmoidLayer());


let mnist = require('mnist');
let set = mnist.set(1000, 10);

let inputs = set.training.map(i => i.input);
let outputs = set.training.map(i => i.output);

nn.train(inputs, outputs, 1);

let testInputs = set.test.map(i => i.input);
let testOutputs = set.test.map(i => i.output);

console.log(nn.testPerformance(testInputs, testOutputs));

// let nn = new NeuralNetwork();
// nn.addLayer(new DenseLayer(2, 100));
// nn.addLayer(new TanhLayer());
// nn.addLayer(new DenseLayer(100, 1));
// nn.addLayer(new SigmoidLayer());

// nn.train(inputs, outputs, 10000);

// //let nn = NeuralNetwork.fromJsonFile('models/nn.txt');
// console.log(nn.predict(inputs[3]).data[0]);



//let testInputs = set.test.map(i => i.input);
//let testOutputs = set.test.map(i => i.output);

// nn.toJsonFile('models/nn.txt');