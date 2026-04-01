import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import ai.Neuron;

public class Main extends JPanel implements ActionListener {

    private int x1 = 100, y1 = 400;
    private int size = 50;
    private Timer timer;
    
    private int tries = load("tries.txt");
    private long startTime;
    private final int TIME_LIMIT_MS = 20000;
    private final int GOAL_X = 1500;

    Neuron neuron = new Neuron(1, new double[]{0.0}, 0.01, 1);

    public Main() {
        this.setFocusable(true);
        this.setBackground(Color.WHITE);
        neuron.load("w.txt");
        this.startTime = System.currentTimeMillis();

        timer = new Timer(20, this);
        timer.start();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.BOLD, 18));
        g2d.drawString("Tries: " + tries, 20, 30);
        
        long elapsed = System.currentTimeMillis() - startTime;
        double timeLeft = (TIME_LIMIT_MS - elapsed) / 1000.0;
        if (timeLeft < 0) timeLeft = 0;
        g2d.drawString("Time Left: " + String.format("%.2f", timeLeft) + "s", 20, 60);

        g2d.setColor(new Color(46, 204, 113));
        g2d.fillRect(GOAL_X, 0, 10, getHeight());

        g2d.setColor(new Color(52, 152, 219));
        g2d.fillRect(x1, y1, size, size);
        g2d.setColor(Color.BLACK);
        g2d.drawRect(x1, y1, size, size);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        double normalizedX = x1 / (double)GOAL_X; 
        neuron.setInputs(new double[]{ normalizedX });

        double rawPrediction = neuron.predict(); 
        double pixelPrediction = rawPrediction * GOAL_X;

        if (pixelPrediction > x1) {
             x1 += 3;
        } else if (pixelPrediction < x1) {
            x1 -= 3;
        }

        double reward = 0;
        long currentTime = System.currentTimeMillis();
        boolean reset = false;
        
        if (currentTime - startTime > TIME_LIMIT_MS) {
            reward = -2.0;
            System.out.println("TIMEOUT! Try: " + tries);
            reset = true;
        } else if (x1 <= 0) {
            reward = -1.0; 
            reset = true;
        } else if (x1 >= GOAL_X) {
            reward = 5.0;
            System.out.println("GOAL REACHED! Try: " + tries);
            reset = true;
        } else {
            reward = (x1 / (double)GOAL_X) * 0.1;
        }

        neuron.train(1.0);
        neuron.motivate(reward, 0.9, normalizedX);

        if (reset) {
            save("tries.txt", ++tries);
            x1 = 100;
            startTime = currentTime;
            if (tries % 10 == 0){
                 neuron.save("w.txt");
            }
        }

        repaint();
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("@AI_Playground_real");
        Main panel = new Main();
        frame.add(panel);
        frame.setSize(1600, 600);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent windowEvent) {
                panel.neuron.save("w.txt");
                save("tries.txt", panel.tries);
                System.out.println("Weights saved. Goodbye!");
                System.exit(0);
            }
        });

        frame.setVisible(true);
    }

    static void save(String filename, int tries) {
        try (java.io.PrintWriter out = new java.io.PrintWriter(filename)) {
            out.println(tries);
        } catch (java.io.IOException e) {
            e.printStackTrace();
        }
    }

    static int load(String filename) {
        java.io.File file = new java.io.File(filename);
        if (!file.exists()) return 0;
        try (java.util.Scanner scanner = new java.util.Scanner(file)) {
            if (scanner.hasNextInt()) {
                return scanner.nextInt();
            }
        } catch (java.io.IOException e) {
            e.printStackTrace();
        }
        return 0;
    }
}