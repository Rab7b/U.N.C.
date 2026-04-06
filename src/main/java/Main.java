import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import ai.Neuron;

public class Main extends JPanel implements ActionListener {

    private double x1 = 100, y1 = 540;
    private Neuron[] neurons;

    private int size = 60;
    private Timer timer;
    private int tries;
    private long startTime;
    private final int TIME_LIMIT_MS = 20000;
    private final int GOAL_X = 1800;

    private final int WALL_X = 900;
    private final int WALL_Y = 300;
    private final int WALL_W = 40;
    private final int WALL_H = 850;

    public Main() {
        this.setFocusable(true);
        this.setBackground(new Color(25, 25, 25));
        this.tries = load("data/dtries.txt");

        this.neurons = new Neuron[2];
        for (int i = 0; i < 2; i++) {

            this.neurons[i] = new Neuron(3, new double[] { 0.0, 0.0, 0.0 }, 0.01, 7, neurons);
            this.neurons[i].importWeights(i == 0 ? "data/w_x.txt" : "data/w_y.txt");
        }

        this.startTime = System.currentTimeMillis();
        this.timer = new Timer(16, this);
        this.timer.start();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g2d.setColor(new Color(200, 50, 50));
        g2d.fillRect(0, 0, getWidth(), 10);
        g2d.fillRect(0, getHeight() - 10, getWidth(), 10);

        g2d.setColor(new Color(255, 165, 0));
        g2d.fillRect(WALL_X, WALL_Y, WALL_W, WALL_H);
        g2d.setColor(Color.WHITE);
        g2d.drawRect(WALL_X, WALL_Y, WALL_W, WALL_H);

        g2d.setColor(Color.WHITE);
        g2d.setFont(new Font("Consolas", Font.BOLD, 26));
        g2d.drawString("TRIES: " + tries, 50, 60);
        long elapsed = System.currentTimeMillis() - startTime;
        double timeLeft = Math.max(0, (TIME_LIMIT_MS - elapsed) / 1000.0);
        g2d.drawString("TIME: " + String.format("%.2f", timeLeft) + "s", 50, 100);

        g2d.setColor(new Color(46, 204, 113));
        g2d.fillRect(GOAL_X, 0, 25, getHeight());

        g2d.setColor(new Color(52, 152, 219));
        g2d.fillRect((int) x1, (int) y1, size, size);
        g2d.setColor(Color.WHITE);
        g2d.drawRect((int) x1, (int) y1, size, size);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        double normX = x1 / 1920.0;
        double normY = y1 / 1080.0;
        double distToGoal = (GOAL_X - x1) / 1920.0;

        neurons[0].setInputs(new double[] { normX, normY, distToGoal });
        neurons[1].setInputs(new double[] { normX, normY, distToGoal });

        double moveX = neurons[0].predict() * 15.0; 
        double moveY = neurons[1].predict() * 15.0;

        x1 += moveX;
        y1 += moveY;

        Rectangle agent = new Rectangle((int) x1, (int) y1, size, size);
        Rectangle wall = new Rectangle(WALL_X, WALL_Y, WALL_W, WALL_H);

        double reward = (x1 / 1920.0) * 50.0;
        double yDiff = Math.abs(y1 - 150);
        reward -= (yDiff / 1080.0) * 20.0;

        boolean reset = false;

        if (agent.intersects(wall)) {
            reward = -5000.0; 
            reset = true;
        }

        if (y1 <= 10 || y1 >= getHeight() - size - 10 || x1 < 0) {
            reward = -1000.0;
            reset = true;
        }

        if (x1 >= GOAL_X) {
            reward = 10000.0;
            reset = true;
        }

        if (System.currentTimeMillis() - startTime > TIME_LIMIT_MS) {
            reset = true;
        }

        neurons[0].motivate(reward, 0.9, neurons[0].predict());
        neurons[1].motivate(reward, 0.9, neurons[1].predict());

        if (reset) {
            save("data/dtries.txt", ++tries);
            x1 = 100; y1 = 540;
            startTime = System.currentTimeMillis();
            if (tries % 5 == 0) {
                neurons[0].exportWeights("data/w_x.txt");
                neurons[1].exportWeights("data/w_y.txt");
            }
        }
        repaint();
    }

    public static void main(String[] args) {

        System.loadLibrary("neuron_logic"); 
        
        JFrame f = new JFrame("AI_Playground_JNI_Edition");
        Main p = new Main();
        f.add(p);
        f.setSize(1920, 1080);
        f.setExtendedState(JFrame.MAXIMIZED_BOTH);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setVisible(true);
    }

    static void save(String fn, int v) {
        try (java.io.PrintWriter o = new java.io.PrintWriter(fn)) {
            o.println(v);
        } catch (Exception e) {}
    }

    static int load(String fn) {
        java.io.File f = new java.io.File(fn);
        if (!f.exists()) return 0;
        try (java.util.Scanner s = new java.util.Scanner(f)) {
            return s.hasNextInt() ? s.nextInt() : 0;
        } catch (Exception e) { return 0; }
    }
}