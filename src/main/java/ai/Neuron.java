package ai;

public class Neuron {

    static {
        System.loadLibrary("neuron_logic");
    }

    private int layerSize;
    private double[] weights, inputs;
    private double bias;
    private double learningRate = 0.0;
    private Neuron[] peers;
    private double prediction;
    private double selfWeight;

    public Neuron(int inputSize, double[] inputs, double learningRate, int layerSize, Neuron[] peers) {
        this.learningRate = learningRate;
        this.layerSize = layerSize;
        this.weights = new double[inputSize * layerSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < layerSize; j++) {
                this.weights[i * layerSize + j] = (Math.random() * 2) - 1;;
            }
        }
        this.bias = 0.1;
        this.inputs = inputs;
        this.peers = peers;
        selfWeight = (Math.random() * 2) - 1;
    }

    public Neuron(int inputSize, double[] inputs, double learningRate, int layerSize) {
        this.learningRate = learningRate;
        this.layerSize = layerSize;
        this.weights = new double[inputSize * layerSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < layerSize; j++) {
                this.weights[i * layerSize + j] = (Math.random() * 2) - 1;;
            }
        }
        this.bias = 0.1;
        this.inputs = inputs;
        selfWeight = (Math.random() * 2) - 1;
    }

    private static native double activate(double x);

    private static native double derivative(double x);

    public native double predict();

    public native void train(double target);

    public native void motivate(double reward, double gamma, double state);

    public native void importWeights(String filename);

    public native void exportWeights(String filename);

    public int getLayerSize() {
        return layerSize;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    public double[] getInputs() {
        return inputs;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLayerSize(int layerSize) {
        this.layerSize = layerSize;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void load(String filename) {
        importWeights(filename);
    }

    public void save(String filename) {
        exportWeights(filename);
    }
}