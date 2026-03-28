import ai.Neuron;

public class Main {

    static {
        System.loadLibrary("neuron_logic");
    }
    public static void main(String[] args) {
        Neuron neuron = new Neuron(1, 0.01, new double[]{0.1}, new double[]{1}, 1);
        for (int i = 0; i < 100000; i++) {
            neuron.train(1);
        }
        System.out.println(neuron.predict());
    }
}