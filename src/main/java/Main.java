import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import ai.Neuron;

public class Main extends JPanel implements ActionListener {

    private int x1 = 100, y1 = 300;
    private int size = 50;
    private Timer timer;

    private int tries;
    private long startTime;
    private final int TIME_LIMIT_MS = 20000;
    private final int GOAL_X = 1500;

    private Neuron[] neurons = {
            new Neuron(2, new double[] { 0.0, 0.0 }, 0.01, 1),
            new Neuron(2, new double[] { 0.0, 0.0 }, 0.01, 1)
    };

    public Main() {
        this.setFocusable(true);
        this.setBackground(Color.WHITE);
        if (load("tries.txt") == 0) {
            tries = 0;
        } else {
            tries = load("tries.txt");
        }
        neurons[0].load("w_x.txt");
        neurons[1].load("w_y.txt");
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
        if (timeLeft < 0)
            timeLeft = 0;
        g2d.drawString("Time Left: " + String.format("%.2f", timeLeft) + "s", 20, 60);

        g2d.setColor(new Color(46, 204, 113));
        g2d.fillRect(GOAL_X, 0, 15, getHeight());

        g2d.setColor(new Color(52, 152, 219));
        g2d.fillRect(x1, y1, size, size);
        g2d.setColor(Color.BLACK);
        g2d.drawRect(x1, y1, size, size);

        g2d.setColor(Color.BLACK);
        g2d.fillRect(950, 300, 15, 300);
        g2d.drawRect(950, 300, 15, 300);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        double normX = x1 / 1600.0;
        double normY = y1 / 600.0;
        double[] inputs = { normX, normY };

        neurons[0].setInputs(inputs);
        neurons[1].setInputs(inputs);

        double predX = neurons[0].predict() * 1600.0;
        double predY = neurons[1].predict() * 600.0;

        Rectangle player = new Rectangle(x1, y1, size, size);
        Rectangle wall = new Rectangle(950, 300, 15, 300);

        if (predX > x1) {
            x1 += 4;
        } else if (predX < x1) {
            x1 -= 4;
        }
        if (predY > y1) {
            y1 += 4;
        } else if (predY < y1) {
            y1 -= 4;
        }

        double reward = 0;
        long currentTime = System.currentTimeMillis();
        boolean reset = false;

        if (currentTime - startTime > TIME_LIMIT_MS) {
            reward = -2.0;
            reset = true;
        } else if (x1 <= 0 || x1 >= 1600 || y1 <= 0 || y1 >= 600) {
            reward = -1.0;
            reset = true;
        } else if (x1 >= GOAL_X) {
            reward = 10.0;
            System.out.println("GOAL REACHED! Try: " + tries);
            reset = true;
        } else {
            reward = (x1 / 1600.0) * 0.5;
        }

        if (player.intersects(wall)) {
            reward = -5.0;
            reset = true; 
        }

        neurons[0].train(1.0);
        neurons[1].train(0.5);

        neurons[0].motivate(reward, 0.9, normX);
        neurons[1].motivate(reward, 0.9, normY);

        if (reset) {
            save("tries.txt", ++tries);
            x1 = 100;
            y1 = 300;
            startTime = currentTime;
            if (tries % 10 == 0) {
                neurons[0].save("w_x.txt");
                neurons[1].save("w_y.txt");
            }
        }
        repaint();
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("@AI_Playground_XY_Line");
        Main panel = new Main();
        frame.add(panel);
        frame.setSize(1600, 600);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent windowEvent) {
                panel.neurons[0].save("w_x.txt");
                panel.neurons[1].save("w_y.txt");
                save("tries.txt", panel.tries);
                System.exit(0);
            }
        });
        frame.setVisible(true);
    }

    static void save(String filename, int tries) {
        try (java.io.PrintWriter out = new java.io.PrintWriter(filename)) {
            out.println(tries);
        } catch (java.io.IOException e) {
        }
    }

    static int load(String filename) {
        java.io.File file = new java.io.File(filename);
        if (!file.exists())
            return 0;
        try (java.util.Scanner sc = new java.util.Scanner(file)) {
            if (sc.hasNextInt())
                return sc.nextInt();
        } catch (java.io.IOException e) {
        }
        return 0;
    }
}