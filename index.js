let DenseLayer = require('./Layers/DenseLayer');
let TanhLayer = require('./Layers/TanhLayer');
let NeuralNetwork = require('./Layers/NeuralNetwork');
let SigmoidLayer = require('./Layers/SigmoidLayer');

// let inputs = [[1, 0], [0, 1], [1, 1], [0, 0]];
// let outputs = [[1], [1], [0], [0]];

const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;
const fs = require('fs');
const path = require('path');

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

app.get('/get_errors', (req, res) => {
   let nn = NeuralNetwork.fromJsonFile('Models/moreComplicatedNetwork.txt');

   let errors = nn.errors;

   res.json({errors})
});

app.get('/get_performances', (req, res) => {
  let nn = NeuralNetwork.fromJsonFile('Models/moreComplicatedNetwork.txt');

  let performances = nn.performances;

  res.json({performances})
});

app.get('/get_numbers', (req, res) => {
  let nn = NeuralNetwork.fromJsonFile('Models/moreComplicatedNetwork.txt');

  let numbers = nn.wronglyGuessedNumberCounts;

  res.json({numbers})
});

app.post('/predict', (req, res) => {
   let input = req.body;

   let nn = NeuralNetwork.fromJsonFile('Models/saveTest.txt');

   let output = nn.predict(input);

   console.log(output);

   res.json(output);
});

function toArrayBuffer(buf) {
  const ab = new ArrayBuffer(buf.length);
  const view = new Uint8Array(ab);
  for (let i = 0; i < buf.length; ++i) {
      view[i] = buf[i];
  }
  return ab;
}

function loadMNIST(callback) {
  let mnist = {};
  let files = {
    train_images: 'train-images-idx3-ubyte',
    train_labels: 'train-labels-idx1-ubyte',
    test_images: 't10k-images-idx3-ubyte',
    test_labels: 't10k-labels-idx1-ubyte',
  };
  return Promise.all(Object.keys(files).map(async file => {
    mnist[file] = await loadFile(files[file])
  })).then(() => callback(mnist));
}

async function loadFile(file) {
  let buffer = fs.readFileSync(path.join(__dirname, "./Data/" + file));
  buffer = toArrayBuffer(buffer);
  let headerCount = 4;
  let headerView = new DataView(buffer, 0, 4 * headerCount);
  let headers = new Array(headerCount).fill().map((_, i) => headerView.getUint32(4 * i, false));

  // Get file type from the magic number
  let type, dataLength;
  if(headers[0] == 2049) {
    type = 'label';
    dataLength = 1;
    headerCount = 2;
  } else if(headers[0] == 2051) {
    type = 'image';
    dataLength = headers[2] * headers[3];
  } else {
    throw new Error("Unknown file type " + headers[0])
  }

  let data = new Uint8Array(buffer, headerCount * 4);
  if(type == 'image') {
    dataArr = [];
    for(let i = 0; i < headers[1]; i++) {
      dataArr.push(data.subarray(dataLength * i, dataLength * (i + 1)));
    }
    return dataArr;
  }
  return data;
}

let mnist;

loadMNIST(function(data) {
  mnist = data;


  // let nn = new NeuralNetwork();
  // nn.addLayer(new DenseLayer(784, 100));
  // nn.addLayer(new TanhLayer());
  // nn.addLayer(new DenseLayer(100, 10));
  // nn.addLayer(new SigmoidLayer());

  // let nn = NeuralNetwork.fromJsonFile('Models/moreComplicatedNetwork.txt');
  // nn.layers[0].setLearningRate(0.05);
  // nn.layers[2].setLearningRate(0.05);

  // let inputs = mnist.train_images.slice(20000, 50000).map(i => Array.from(i).map(n => n / 255));
  // let targets = mnist.train_labels.slice(20000, 50000);
  // let outputs = [];

  // targets.forEach(target => {
  //   let dataPoints = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  //   dataPoints[target] = 1;
  //   outputs.push(dataPoints);
  // });

  // let testInputs = mnist.test_images.map(i => Array.from(i).map(n => n / 255));
  // let testTargets = mnist.test_labels;
  // let testOutputs = [];

  // testTargets.forEach(testTarget => {
  //   let dataPoints = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  //   dataPoints[testTarget] = 1;
  //   testOutputs.push(dataPoints);
  // });

  // nn.setTests(testInputs, testOutputs);

  // nn.train(inputs, outputs, 1);

  // let output = nn.testPerformance(testInputs, testOutputs);

  // nn.toJsonFile('Models/moreComplicatedNetwork.txt');

  // console.log(output);
});






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