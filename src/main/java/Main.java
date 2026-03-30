import ai.Neuron;

public class Main {

    static {
        System.loadLibrary("neuron_logic");
    }
    public static void main(String[] args) {
        Neuron neuron = new Neuron(2, 0.01, new double[]{0.5, 0.5}, new double[]{1}, 1);
        for (int i = 0; i < 100000; i++) {
            neuron.train(1);
            System.out.println(neuron.predict());
        }
        System.out.println("Final answer: " + neuron.predict());
    }
}