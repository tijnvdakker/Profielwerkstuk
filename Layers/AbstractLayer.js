class AbstractLayer {
    constructor() {
        if (this.constructor === AbstractLayer) {
            throw new Error("Abstract class 'AbstractLayer' cannot be instantiated.");
        }
        this.input = undefined;
        this.output = undefined;
    }

    forward(inputs) {
        throw new Error("Function 'forward' must be implemented.");
    }

    backward(outputGradient) {
        throw new Error("Function 'backward' must be implemented.");
    }
}

module.exports = AbstractLayer;
