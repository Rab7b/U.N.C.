package ai;

public class Neuron {
    private double[][] weights;
    private final double[] bias;
    private final double[] out;
    private final double learningRate;
    private double[] inputs;
    private double[] targets;

    static { System.loadLibrary("neuron_logic"); }

    public Neuron(int inputSize, double learningRate, double[] inputs, double[] targets) {
        this.weights = new double[inputSize + 3][7];
        this.bias = new double[7];
        this.out = new double[7];
        double scale = Math.sqrt(2.0 / inputSize);
        for (int i = 0; i < this.weights.length; i++) {
            for (int j = 0; j < 7; j++) this.weights[i][j] = (Math.random() * 2 - 1) * scale;
        }
        this.targets = targets;
        this.inputs = inputs;
        this.learningRate = learningRate;
    }

    private native static double activate(double sum);
    private native static double derivative(double output);
    public native double predict();
    public native void train(double target);
    public native void motivate(double reward, double[] nextState, double gamma);
    public native void exportWeights(String path, double[][] weightsToExport);
    public native void importWeights(String path);

    public void addInput(double[] input) { this.inputs = input; }
    public void addWeight(double[] weight) {
        double[][] nW = new double[this.weights.length + 1][7];
        System.arraycopy(this.weights, 0, nW, 0, this.weights.length);
        nW[this.weights.length] = weight;
        this.weights = nW;
    }
    public void save(String f) { exportWeights(f, this.weights); }
    public void load(String f) { importWeights(f); }
    
    public double[] getTarget() { return targets; }
    public void setTarget(double[] t) { this.targets = t; }
    public void setInputs(double[] i) { this.inputs = i; }
    public double[][] getWeights() { return weights; }
}