import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import ai.Neuron;

public class Main extends JPanel implements ActionListener {

    private double x1 = 100, y1 = 200;
    private Neuron[] neuronsBlue;
    private double x2 = 100, y2 = 500;
    private Neuron[] neuronsRed;

    private int size = 50;
    private Timer timer;
    private int tries;
    private long startTime;
    private final int TIME_LIMIT_MS = 20000;
    private final int GOAL_X = 1500;
    private final Rectangle wall = new Rectangle(950, 450, 100, 300);

    public Main() {
        this.setFocusable(true);
        this.setBackground(Color.WHITE);
        this.tries = load("data/dtries.txt");

        // Исправлено: 2 нейрона (X и Y), 3 входа (соответствует updateAgent)
        this.neuronsBlue = new Neuron[2];
        for(int i = 0; i < 2; i++) {
            this.neuronsBlue[i] = new Neuron(3, new double[] { 0.0, 0.0, 0.0 }, 0.01, 7, neuronsBlue);
        }
        this.neuronsBlue[0].load("data/blue_w_x.txt");
        this.neuronsBlue[1].load("data/blue_w_y.txt");

        this.neuronsRed = new Neuron[2];
        this.neuronsRed[0] = new Neuron(3, new double[] { 0.0, 0.0, 0.0 }, 0.01, 1, neuronsRed);
        this.neuronsRed[1] = new Neuron(3, new double[] { 0.0, 0.0, 0.0 }, 0.01, 1, neuronsRed);
        this.neuronsRed[0].load("data/red_w_x.txt");
        this.neuronsRed[1].load("data/red_w_y.txt");

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
        g2d.fillRect(wall.x, wall.y, wall.width, wall.height);

        g2d.setColor(new Color(52, 152, 219));
        g2d.fillRect((int) x1, (int) y1, size, size);
        g2d.setColor(Color.BLACK);
        g2d.drawRect((int) x1, (int) y1, size, size);

        g2d.setColor(new Color(231, 76, 60));
        g2d.fillRect((int) x2, (int) y2, size, size);
        g2d.setColor(Color.BLACK);
        g2d.drawRect((int) x2, (int) y2, size, size);
    }

    private void updateAgent(int id) {
        double curX = (id == 1) ? x1 : x2;
        double curY = (id == 1) ? y1 : y2;
        Neuron[] curNeurons = (id == 1) ? neuronsBlue : neuronsRed;

        double normX = curX / 1600.0;
        double normY = curY / 600.0;
        Rectangle lookAhead = new Rectangle((int) curX + 60, (int) curY, size, size);
        double wallAhead = lookAhead.intersects(wall) ? 1.0 : 0.0;

        curNeurons[0].setInputs(new double[] { normX, normY, wallAhead });
        curNeurons[1].setInputs(new double[] { normX, normY, wallAhead });

        double moveX = Math.max(0, curNeurons[0].predict() * 12.0); // Чтобы не пятились назад
        double moveY = (curNeurons[1].predict() - 0.5) * 12.0;

        if (id == 1) {
            x1 += moveX;
            y1 += moveY;
        } else {
            x2 += moveX;
            y2 += moveY;
        }
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        updateAgent(1);
        updateAgent(2);

        Rectangle p1 = new Rectangle((int) x1, (int) y1, size, size);
        Rectangle p2 = new Rectangle((int) x2, (int) y2, size, size);

        double reward1 = calculateReward(x1, y1, p1, 200);
        double reward2 = calculateReward(x2, y2, p2, 500);

        boolean reset = false;
        long currentTime = System.currentTimeMillis();

        if (p1.intersects(wall)) { reward1 = -30.0; x1 = 100; y1 = 200; }
        if (p2.intersects(wall)) { reward2 = -30.0; x2 = 100; y2 = 500; }
        
        if (p1.intersects(p2)) {
            reward1 = -15.0; reward2 = -15.0;
            reset = true;
        }

        if (currentTime - startTime > TIME_LIMIT_MS || x1 >= GOAL_X || x2 >= GOAL_X) {
            reset = true;
        }

        neuronsBlue[0].train(0.5);
        neuronsBlue[1].train(1.0);
        neuronsBlue[0].motivate(reward1, 0.9, x1 / 1600.0);
        neuronsBlue[1].motivate(reward1, 0.9, y1 / 600.0);

        neuronsRed[0].train(0.5);
        neuronsRed[1].train(1.0);
        neuronsRed[0].motivate(reward2, 0.9, x2 / 1600.0);
        neuronsRed[1].motivate(reward2, 0.9, y2 / 600.0);

        if (reset) {
            save("data/dtries.txt", ++tries);
            if (x1 >= GOAL_X) {
                neuronsBlue[0].motivate(100, 0.9, x1 / 1600.0);
                neuronsBlue[1].motivate(100, 0.9, y1 / 600.0);
                neuronsRed[0].motivate(-100, 0.9, x2 / 1600.0);
                neuronsRed[1].motivate(-100, 0.9, y2 / 600.0);
            } else if (x2 >= GOAL_X) {
                neuronsBlue[0].motivate(-100, 0.9, x1 / 1600.0);
                neuronsBlue[1].motivate(-100, 0.9, y1 / 600.0);
                neuronsRed[0].motivate(100, 0.9, x2 / 1600.0);
                neuronsRed[1].motivate(100, 0.9, y2 / 600.0);
            }
            x1 = 100; y1 = 200;
            x2 = 100; y2 = 500;
            startTime = currentTime;
            if (tries % 5 == 0) {
                neuronsBlue[0].save("data/blue_w_x.txt");
                neuronsBlue[1].save("data/blue_w_y.txt");
                neuronsRed[0].save("data/red_w_x.txt");
                neuronsRed[1].save("data/red_w_y.txt");
            }
        }
        repaint();
    }

    private double calculateReward(double x, double y, Rectangle p, int targetY) {
        if (p.intersects(wall)) return -20.0;
        if (x >= GOAL_X) return 100.0;
        double progress = (x / 1600.0) * 5.0;
        double yPenalty = Math.abs(y - targetY) / 300.0;
        return progress - yPenalty;
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("@AI_Playground_Battle");
        Main panel = new Main();
        frame.add(panel);
        frame.setSize(1600, 600);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    static void save(String filename, int val) {
        try (java.io.PrintWriter out = new java.io.PrintWriter(filename)) {
            out.println(val);
        } catch (java.io.IOException e) {}
    }

    static int load(String filename) {
        java.io.File file = new java.io.File(filename);
        if (!file.exists()) return 0;
        try (java.util.Scanner sc = new java.util.Scanner(file)) {
            if (sc.hasNextInt()) return sc.nextInt();
        } catch (java.io.IOException e) {}
        return 0;
    }
}