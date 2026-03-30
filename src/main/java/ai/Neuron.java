package ai;

public class Neuron {

    private double[][] weights;
    private final double[] bias;
    private final double[] out;
    private final double learningRate;
    private double[] inputs;
    private double[] targets;

    static {
        System.loadLibrary("neuron_logic");
    }

    public Neuron(int inputSize, double learningRate, double[] inputs, double[] targets, int outs) {
        this.weights = new double[inputSize+3][7];
        this.bias = new double[7];
        this.out = new double[outs];

        if(inputSize > inputs.length){
            inputSize = inputs.length;
        }
        if(outs > targets.length){
            outs = targets.length;
        }
        if(inputs.length > 0){
            System.arraycopy(inputs, 0, this.inputs, 0, inputSize);
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < 7; j++) {
                this.weights[i][j] = Math.random() * 2 - 1;
            }
        }
        for (int i = 0; i < 7; i++) {
            this.bias[i] = Math.random() * 2 - 1 * 0.1;
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

    public double[] getTarget() {
        return targets;
    }

    public void setTarget(double[] targets) {
        this.targets = targets;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public void addInput(double[] input){
        this.inputs = input;
    }

    public void addWeight(double[] weight){
        double[][] newWeights = new double[this.weights.length + 1][];
        System.arraycopy(this.weights, 0, newWeights, 0, this.weights.length);
        newWeights[this.weights.length] = weight;
        this.weights = newWeights;
    }

    public void setWeights(double[][] newWeights) {
        for (int i = 0; i < newWeights.length; i++) {
            for (int j = 0; j < newWeights[i].length; j++) {
                this.weights[i][j] = Math.max(-1.0, Math.min(1.0, newWeights[i][j]));
            }
        }
    }

    public double[][] getWeights() {
        return weights;
    }

    public native void exportWeights(String path, double[][] weightsToExport);

    public native void importWeights(String path);

    public void save(String fileName) {
        for(int i = 0; i < this.weights.length; i++){
            for(int j = 0; j < this.weights[i].length; j++){
                if(Double.isNaN(bias[j])){
                    this.bias[j] = 0.0;
                }
                if(Double.isNaN(weights[i][j]) || Double.isInfinite(weights[i][j])){
                    this.weights[i][j] = 0.0;
                }
                this.weights[i][j] += this.bias[j];
                this.weights[i][j] = Math.max(-1.0, Math.min(1.0, this.weights[i][j]));
            }
        }
        exportWeights(fileName, this.weights);
    }

    public void load(String fileName){
        importWeights(fileName);
    }

    public double[] parsePic(String path){
        return new double[0];
    }
}
