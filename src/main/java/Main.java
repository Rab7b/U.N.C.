import ai.Neuron;

public class Main {

    static {
        System.loadLibrary("neuron_logic");
    }
    public static void main(String[] args) {
        Neuron neuron = new Neuron(2, 0.01, new double[]{0.5, 0.5}, new double[]{1});
        neuron.load("w.txt");
        
        for (int i = 0; i < 100; i++) {
            neuron.train(1);
            neuron.motivate(i, new double[]{0.5, 0.5}, i);
        }
        System.out.println("Final answer: " + neuron.predict());
        neuron.save("w.txt");
    }
}