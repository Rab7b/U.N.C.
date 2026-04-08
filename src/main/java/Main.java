import ai.Neuron;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.geom.AffineTransform;

public class Main extends JPanel {
    private BufferedImage track;
    private Agent[] agents;
    private final int AGENT_COUNT = 3;
    private final int WIDTH = 1920;
    private final int HEIGHT = 1080;
    private final Color ASphaltColor = new Color(50, 50, 50);

    public Main() {
        track = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = track.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g2.setColor(new Color(34, 139, 14));
        g2.fillRect(0, 0, WIDTH, HEIGHT);

        g2.setStroke(new BasicStroke(160, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        g2.setColor(ASphaltColor);
        g2.drawOval(200, 150, WIDTH - 400, HEIGHT - 350);

        g2.setStroke(new BasicStroke(5, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[] { 30 }, 0));
        g2.setColor(Color.WHITE);
        g2.drawOval(200, 150, WIDTH - 400, HEIGHT - 350);
        g2.dispose();

        agents = new Agent[AGENT_COUNT];
        Color[] colors = { Color.CYAN, Color.MAGENTA, Color.YELLOW };
        for (int i = 0; i < AGENT_COUNT; i++) {
            agents[i] = new Agent(WIDTH / 2, 150, colors[i]);
        }

        new Timer(16, e -> {
            for (Agent a : agents)
                a.update(track);
            repaint();
        }).start();

        new Timer(15000, e -> {
            for (Agent a : agents) {
                if (a.distanceCovered < 200)
                    a.reset();
                a.distanceCovered = 0;
            }
        }).start();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(track, 0, 0, getWidth(), getHeight(), null);
        for (Agent a : agents)
            a.draw(g);
    }

    class Agent {
        double x, y, startX, startY, angle = Math.PI;
        double distanceCovered = 0;
        Color color;
        Neuron[] brain;
        int id;
        private static int agentCounter = 0;

        Agent(double x, double y, Color c) {
            this.id = agentCounter++;
            this.x = this.startX = x;
            this.y = this.startY = y;
            this.color = c;
            brain = new Neuron[2];
            brain[0] = new Neuron(5, new double[] { 0.5, 0.5, 0.5, 0.5, 0.5 }, 0.5, 1, brain);
            brain[1] = new Neuron(5, new double[] { 0.5, 0.5, 0.5, 0.5, 0.5 }, 0.5, 1, brain);

            brain[0].load("data/speed_" + id + ".txt");
            brain[1].load("data/steer_" + id + ".txt");
        }

        void update(BufferedImage track) {
            double[] inputs = new double[5];
            for (int i = 0; i < 5; i++)
                inputs[i] = getRay(track, (i - 2) * (Math.PI / 8));

            brain[0].setInputs(inputs);
            brain[1].setInputs(inputs);

            double speed = Math.max(0.8, brain[0].predict() * 14);
            double steer = (brain[1].predict() - 0.5) * 0.18;

            angle += steer;
            double oldX = x, oldY = y;

            x += Math.cos(angle) * speed;
            y += Math.sin(angle) * speed;

            for (Agent other : agents) {
                if (other == this)
                    continue;
                double dx = other.x - x;
                double dy = other.y - y;
                double dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 70) {
                    double dot = (dx * Math.cos(angle) + dy * Math.sin(angle)) / dist;
                    if (dot > 0.7) {
                        brain[0].motivate(-5.0, 0.9, 0.5);
                        speed *= 0.4;
                    }
                }
            }

            if (isOffTrack(track, x, y)) {
                x = oldX;
                y = oldY;
                brain[0].motivate(-6.0, 0.9, 0.5);
                brain[1].motivate(-2.0, 0.9, 0.5);
                angle += (Math.random() - 0.5) * 0.4;
            } else {
                double directionBonus = -Math.cos(angle) * 2.5;
                if (speed < 4.0)
                    brain[0].motivate(-1.0, 0.9, 0.5);
                brain[0].motivate(directionBonus + (speed * 0.15), 0.9, 0.5);
                distanceCovered += speed;
            }

            brain[0].save("data/speed_" + id + ".txt");
            brain[1].save("data/steer_" + id + ".txt");
        }

        void reset() {
            x = startX;
            y = startY;
            angle = Math.PI;
            distanceCovered = 0;
        }

        double getRay(BufferedImage track, double offset) {
            double rayRange = 350;
            for (int d = 0; d < rayRange; d += 15) {
                int cx = (int) (x + Math.cos(angle + offset) * d);
                int cy = (int) (y + Math.sin(angle + offset) * d);
                if (isOffTrack(track, cx, cy))
                    return d / rayRange;
            }
            return 1.0;
        }

        boolean isOffTrack(BufferedImage track, double px, double py) {
            if (px < 0 || px >= WIDTH || py < 0 || py >= HEIGHT)
                return true;
            return track.getRGB((int) px, (int) py) != ASphaltColor.getRGB();
        }

        void draw(Graphics g) {
            Graphics2D g2 = (Graphics2D) g;
            double scaleX = (double) getWidth() / WIDTH;
            double scaleY = (double) getHeight() / HEIGHT;
            AffineTransform old = g2.getTransform();
            g2.translate(x * scaleX, y * scaleY);
            g2.rotate(angle);
            g2.setColor(color);
            g2.fillRect(-20, -10, 40, 20);
            g2.setColor(Color.BLACK);
            g2.fillRect(15, -10, 5, 20);
            g2.setTransform(old);
        }
    }

    public static void main(String[] args) {
        System.loadLibrary("neuron_logic");
        JFrame f = new JFrame("@AI_Playground_UNC");
        f.setExtendedState(JFrame.MAXIMIZED_BOTH);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.add(new Main());
        f.setVisible(true);
    }
}