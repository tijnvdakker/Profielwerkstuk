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

const express = require('express');
const bodyParser = require('body-parser');
const mnist = require('mnist');
const app = express();
const port = 3000;

app.use(bodyParser.urlencoded({ extended: false, limit: '50mb', parameterLimit: 1000000}));
app.use(bodyParser.json());

app.get('/', (req, res) => {
  res.redirect('/home');
})

app.get('/home', (req, res) => {
    res.sendFile('Views/home.html', {root: __dirname });
})

app.get('/get_model/:name', (req, res) => {
    res.send(NeuralNetwork.fromJsonFile('models/' + req.params.name + '.txt'));
});

app.post('/train_and_test', (req, res) => {
    let trainingAmount = req.body.trainingAmount;
    let testAmount = req.body.testAmount;

    let nn = NeuralNetwork.fromJsonFile('Models/nn.txt');

    let set = mnist.set(trainingAmount, testAmount);

    let inputs = set.training.map(i => i.input);
    let outputs = set.training.map(i => i.output);

    nn.train(inputs, outputs, 1);

    let testInputs = set.test.map(i => i.input);
    let testOutputs = set.test.map(i => i.output);

    nn.toJsonFile('Models/nn.txt');

    let output = nn.testPerformance(testInputs, testOutputs);

    res.send(output.toString());
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
});

app.post('/predict', (req, res) => {
   let input = req.body;

   let nn = NeuralNetwork.fromJsonFile('Models/nn.txt');

   let output = nn.predict(input);

   console.log(output);

   res.json(output);
});

//nn = NeuralNetwork.fromJsonFile('Models/nn.txt');

let set = mnist.set(1000, 100);

let inputs = set.training.map(i => i.input);
let outputs = set.training.map(i => i.output);

nn.train(inputs, outputs, 1);

let testInputs = set.test.map(i => i.input);
let testOutputs = set.test.map(i => i.output);

let output = nn.testPerformance(testInputs, testOutputs);

nn.toJsonFile('Models/test.txt');

console.log(output);


// 
// 

// let testInputs = set.test.map(i => i.input);
// let testOutputs = set.test.map(i => i.output);

// console.log(nn.testPerformance(testInputs, testOutputs));



// nn.train(inputs, outputs, 10000);

// //let nn = NeuralNetwork.fromJsonFile('models/nn.txt');
// console.log(nn.predict(inputs[3]).data[0]);



//let testInputs = set.test.map(i => i.input);
//let testOutputs = set.test.map(i => i.output);

// nn.toJsonFile('models/nn.txt');