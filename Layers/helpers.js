let Matrix = require('./matrix');

function meanSquaredError(answers, predictions) {
    let temp = Matrix.subtract(answers, predictions);
    temp.map(x => x ** 2);
    return Matrix.mean(temp);
}

function meanSquaredErrorDerivative(answers, predictions) {
    let temp = Matrix.subtract(predictions, answers);
    temp = temp.map(x => 2 * x);
    return temp.map(x => x / (temp.rows * temp.cols));
}

module.exports = {
    meanSquaredError,
    meanSquaredErrorDerivative
}