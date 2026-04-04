import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import ai.Neuron;

public class Main extends JPanel implements ActionListener {

    private double x1 = 100, y1 = 300;
    private Neuron[] neuronsBlue;
    private double x2 = 100, y2 = 700;
    private Neuron[] neuronsRed;

    private int size = 60;
    private Timer timer;
    private int tries;
    private long startTime;
    private final int TIME_LIMIT_MS = 25000;
    private final int GOAL_X = 1800;
    private final Rectangle wall = new Rectangle(1000, 350, 150, 400);

    public Main() {
        this.setFocusable(true);
        this.setBackground(new Color(25, 25, 25));
        this.tries = load("data/dtries.txt");

        this.neuronsBlue = new Neuron[2];
        for (int i = 0; i < 2; i++) {
            this.neuronsBlue[i] = new Neuron(3, new double[]{0.0, 0.0, 0.0}, 0.01, 7, neuronsBlue);
            this.neuronsBlue[i].load(i == 0 ? "data/blue_w_x.txt" : "data/blue_w_y.txt");
        }

        this.neuronsRed = new Neuron[2];
        for (int i = 0; i < 2; i++) {
            this.neuronsRed[i] = new Neuron(3, new double[]{0.0, 0.0, 0.0}, 0.01, 1, neuronsRed);
            this.neuronsRed[i].load(i == 0 ? "data/red_w_x.txt" : "data/red_w_y.txt");
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

        g2d.setColor(Color.WHITE);
        g2d.setFont(new Font("Consolas", Font.BOLD, 26));
        g2d.drawString("TRIES: " + tries, 50, 60);
        long elapsed = System.currentTimeMillis() - startTime;
        double timeLeft = Math.max(0, (TIME_LIMIT_MS - elapsed) / 1000.0);
        g2d.drawString("TIME: " + String.format("%.2f", timeLeft) + "s", 50, 100);

        g2d.setColor(new Color(46, 204, 113));
        g2d.fillRect(GOAL_X, 0, 25, getHeight());

        g2d.setColor(new Color(80, 80, 80));
        g2d.fillRect(wall.x, wall.y, wall.width, wall.height);
        g2d.setColor(Color.WHITE);
        g2d.setStroke(new BasicStroke(3));
        g2d.drawRect(wall.x, wall.y, wall.width, wall.height);

        g2d.setColor(new Color(52, 152, 219));
        g2d.fillRect((int) x1, (int) y1, size, size);
        g2d.setColor(Color.WHITE);
        g2d.drawRect((int) x1, (int) y1, size, size);

        g2d.setColor(new Color(231, 76, 60));
        g2d.fillRect((int) x2, (int) y2, size, size);
        g2d.setColor(Color.WHITE);
        g2d.drawRect((int) x2, (int) y2, size, size);
    }

    private void updateAgent(int id) {
        double curX = (id == 1) ? x1 : x2;
        double curY = (id == 1) ? y1 : y2;
        Neuron[] curNeurons = (id == 1) ? neuronsBlue : neuronsRed;

        double normX = curX / 1920.0;
        double normY = curY / 1080.0;
        Rectangle lookAhead = new Rectangle((int) curX + 70, (int) curY, size, size);
        double wallAhead = lookAhead.intersects(wall) ? 1.0 : 0.0;

        curNeurons[0].setInputs(new double[]{normX, normY, wallAhead});
        curNeurons[1].setInputs(new double[]{normX, normY, wallAhead});

        double moveX = Math.max(0, curNeurons[0].predict() * 14.0);
        double moveY = (curNeurons[1].predict() - 0.5) * 14.0;

        if (id == 1) { x1 += moveX; y1 += moveY; } 
        else { x2 += moveX; y2 += moveY; }
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        updateAgent(1);
        updateAgent(2);

        Rectangle p1 = new Rectangle((int) x1, (int) y1, size, size);
        Rectangle p2 = new Rectangle((int) x2, (int) y2, size, size);

        double r1 = calculateReward(x1, y1, p1, 300);
        double r2 = calculateReward(x2, y2, p2, 700);

        boolean reset = false;
        long now = System.currentTimeMillis();

        if (p1.intersects(wall)) { r1 = -40.0; x1 = 100; y1 = 300; }
        if (p2.intersects(wall)) { r2 = -40.0; x2 = 100; y2 = 700; }

        if (y1 <= 10 || y1 >= getHeight() - size - 10) { r1 = -25.0; x1 = 100; y1 = 300; }
        if (y2 <= 10 || y2 >= getHeight() - size - 10) { r2 = -25.0; x2 = 100; y2 = 700; }

        if (p1.intersects(p2)) { r1 = -15.0; r2 = -15.0; reset = true; }
        if (now - startTime > TIME_LIMIT_MS || x1 >= GOAL_X || x2 >= GOAL_X) reset = true;

        neuronsBlue[0].train(0.5);
        neuronsBlue[1].train(1.0);
        neuronsBlue[0].motivate(r1, 0.9, x1 / 1920.0);
        neuronsBlue[1].motivate(r1, 0.9, y1 / 1080.0);

        neuronsRed[0].train(0.5);
        neuronsRed[1].train(1.0);
        neuronsRed[0].motivate(r2, 0.9, x2 / 1920.0);
        neuronsRed[1].motivate(r2, 0.9, y2 / 1080.0);

        if (reset) {
            save("data/dtries.txt", ++tries);
            if (x1 >= GOAL_X) {
                applyFinal(neuronsBlue, neuronsRed, x1, y1, x2, y2);
            } else if (x2 >= GOAL_X) {
                applyFinal(neuronsRed, neuronsBlue, x2, y2, x1, y1);
            }
            x1 = 100; y1 = 300;
            x2 = 100; y2 = 700;
            startTime = now;
            if (tries % 5 == 0) saveWeights();
        }
        repaint();
    }

    private void applyFinal(Neuron[] w, Neuron[] l, double wx, double wy, double lx, double ly) {
        w[0].motivate(200, 0.95, wx / 1920.0);
        w[1].motivate(200, 0.95, wy / 1080.0);
        l[0].motivate(-150, 0.95, lx / 1920.0);
        l[1].motivate(-150, 0.95, ly / 1080.0);
    }

    private void saveWeights() {
        neuronsBlue[0].save("data/blue_w_x.txt");
        neuronsBlue[1].save("data/blue_w_y.txt");
        neuronsRed[0].save("data/red_w_x.txt");
        neuronsRed[1].save("data/red_w_y.txt");
    }

    private double calculateReward(double x, double y, Rectangle p, int ty) {
        if (p.intersects(wall)) return -35.0;
        if (x >= GOAL_X) return 250.0;
        return ((x / 1920.0) * 10.0) - (Math.abs(y - ty) / 540.0);
    }

    public static void main(String[] args) {
        JFrame f = new JFrame("AI Warehouse Style Simulator");
        Main p = new Main();
        f.add(p);
        f.setSize(1920, 1080);
        f.setExtendedState(JFrame.MAXIMIZED_BOTH);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setVisible(true);
    }

    static void save(String fn, int v) {
        try (java.io.PrintWriter o = new java.io.PrintWriter(fn)) { o.println(v); } catch (Exception e) {}
    }

    static int load(String fn) {
        java.io.File f = new java.io.File(fn);
        if (!f.exists()) return 0;
        try (java.util.Scanner s = new java.util.Scanner(f)) { return s.hasNextInt() ? s.nextInt() : 0; } catch (Exception e) { return 0; }
    }
}