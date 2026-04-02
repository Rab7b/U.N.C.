import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import ai.Neuron;

public class Main extends JPanel implements ActionListener {

    private double x1 = 100, y1 = 300;
    private double angle = 0;
    private int size = 50;
    private Timer timer;

    private int tries;
    private long startTime;
    private final int TIME_LIMIT_MS = 30000;
    private final int GOAL_X = 1500;

    private Neuron[] neurons;

    public Main() {
        this.setFocusable(true);
        this.setBackground(Color.WHITE);
        this.tries = load("tries.txt");

        this.neurons = new Neuron[3];
        this.neurons[0] = new Neuron(3, new double[] { 0.0, 0.0, 0.0 }, 0.01, 10, neurons);
        this.neurons[1] = new Neuron(3, new double[] { 0.0, 0.0, 0.0 }, 0.01, 10, neurons);
        this.neurons[2] = new Neuron(3, new double[] { 0.0, 0.0, 0.0 }, 0.01, 10, neurons);

        this.neurons[0].load("w_x.txt");
        this.neurons[1].load("w_y.txt");
        this.neurons[2].load("w_speed.txt");
        this.startTime = System.currentTimeMillis();

        this.timer = new Timer(20, this);
        this.timer.start();
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
        double timeLeft = Math.max(0, (TIME_LIMIT_MS - elapsed) / 1000.0);
        g2d.drawString("Time Left: " + String.format("%.2f", timeLeft) + "s", 20, 60);

        g2d.setColor(new Color(46, 204, 113));
        g2d.fillRect(GOAL_X, 0, 15, getHeight());

        g2d.setColor(Color.BLACK);
        g2d.fillRect(950, 300, 15, 300);

        g2d.translate(x1 + size / 2.0, y1 + size / 2.0);
        g2d.rotate(angle);

        g2d.setColor(new Color(52, 152, 219));
        g2d.fillRect(-size / 2, -size / 2, size, size);
        g2d.setColor(Color.BLACK);
        g2d.drawRect(-size / 2, -size / 2, size, size);
        g2d.drawLine(0, 0, size / 2 + 10, 0);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        double sensorRange = 150.0;
        double rayX = x1 + Math.cos(angle) * sensorRange;
        double rayY = y1 + Math.sin(angle) * sensorRange;
        double wallSeen = (rayX >= 950 && rayX <= 965 && rayY >= 300 && rayY <= 600) ? 1.0 : 0.0;

        double normX = x1 / 1600.0;
        double normAngle = (angle % (2 * Math.PI)) / (2 * Math.PI);
        double[] inputs = { normX, normAngle, wallSeen };

        neurons[0].setInputs(inputs);
        neurons[1].setInputs(inputs);
        neurons[2].setInputs(inputs);

        double turnAction = (neurons[0].predict() - 0.5) * 0.15;
        angle += turnAction;

        double speed = neurons[2].predict() * 8.0;

        x1 += Math.cos(angle) * speed;
        y1 += Math.sin(angle) * speed;

        double reward = 0;
        boolean reset = false;
        long currentTime = System.currentTimeMillis();
        Rectangle player = new Rectangle((int) x1, (int) y1, size, size);
        Rectangle wall = new Rectangle(950, 300, 15, 300);

        if (currentTime - startTime > TIME_LIMIT_MS) {
            reward = -2.0;
            reset = true;
        } else if (x1 <= 0 || x1 >= 1600 || y1 <= 0 || y1 >= 600) {
            reward = -4.0;
            reset = true;
        } else if (player.intersects(wall)) {
            reward = -10.0;
            reset = true;
        } else if (x1 >= GOAL_X) {
            reward = 50.0;
            reset = true;
        } else {
            double progress = (x1 / 1600.0) * 2.0;
            double spinPenalty = Math.abs(turnAction) * 2.0;
            double straightBonus = (wallSeen > 0 && Math.abs(turnAction) < 0.05) ? 0.5 : 0.0;
            reward = progress - spinPenalty + straightBonus;
        }

        neurons[0].train(0.5);
        neurons[1].train(1.0);
        neurons[2].train(1.0);

        neurons[0].motivate(reward, 0.9, normX);
        neurons[1].motivate(reward, 0.9, normAngle);
        neurons[2].motivate(reward, 0.9, wallSeen);

        if (reset) {
            save("data/tries.txt", ++tries);
            x1 = 100;
            y1 = 300;
            angle = 0;
            startTime = currentTime;
            if (tries % 10 == 0) {
                neurons[0].save("w_x.txt");
                neurons[1].save("w_y.txt");
                neurons[2].save("w_speed.txt");
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
                panel.neurons[0].save("w_x.txt");
                panel.neurons[1].save("w_y.txt");
                panel.neurons[2].save("w_speed.txt");
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