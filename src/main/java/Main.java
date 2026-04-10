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
    private boolean raceFinished = false;

    public Main() {
        track = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = track.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g2.setColor(new Color(30, 120, 30));
        g2.fillRect(0, 0, WIDTH, HEIGHT);

        g2.setColor(ASphaltColor);
        g2.fillRect(0, 400, WIDTH, 280);

        g2.setStroke(new BasicStroke(5, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{40}, 0));
        g2.setColor(Color.WHITE);
        g2.drawLine(0, 540, WIDTH, 540);
        
        g2.setStroke(new BasicStroke(20));
        g2.setColor(Color.RED);
        g2.drawLine(WIDTH - 100, 400, WIDTH - 100, 680); 
        g2.dispose();

        agents = new Agent[AGENT_COUNT];
        Color[] colors = {Color.CYAN, Color.MAGENTA, Color.YELLOW};
        for (int i = 0; i < AGENT_COUNT; i++) {
            agents[i] = new Agent(50, 450 + (i * 80), colors[i]);
        }

        new Timer(16, e -> {
            for (Agent a : agents) a.update(track);
            repaint();
        }).start();

        new Timer(8000, e -> {
            if (raceFinished) {
                for (Agent a : agents) a.reset();
                raceFinished = false;
            }
        }).start();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(track, 0, 0, getWidth(), getHeight(), null);
        for (Agent a : agents) a.draw(g);
    }

    class Agent {
        double x, y, startX, startY, angle = 0;
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
            brain[0] = new Neuron(5, new double[]{0.5, 0.5, 0.5, 0.5, 0.5}, 0.5, 1, brain);
            brain[1] = new Neuron(5, new double[]{0.5, 0.5, 0.5, 0.5, 0.5}, 0.5, 1, brain);
            
            brain[0].load("data/speed_drag_" + id + ".txt");
            brain[1].load("data/steer_drag_" + id + ".txt");
        }

        void update(BufferedImage track) {
            if (raceFinished) return;

            double[] inputs = new double[5];
            for (int i = 0; i < 5; i++)
                inputs[i] = getRay(track, (i - 2) * (Math.PI / 10));

            brain[0].setInputs(inputs);
            brain[1].setInputs(inputs);

            double speed = Math.max(0.0, brain[0].predict() * 20);
            double steer = (brain[1].predict() - 0.5) * 0.1;

            angle += steer;
            double oldX = x, oldY = y;

            x += Math.cos(angle) * speed;
            y += Math.sin(angle) * speed;

            if (x > WIDTH - 120 && !raceFinished) {
                raceFinished = true;
                brain[0].motivate(50.0, 0.95, 0.5); 
                System.out.println("Winner: Agent " + id);
            }

            if (isOffTrack(track, x, y)) {
                x = oldX;
                y = oldY;
                brain[0].motivate(-10.0, 0.9, 0.5);
                angle = 0; 
            } else {
                double forwardBonus = Math.cos(angle) * speed * 0.5;
                brain[0].motivate(forwardBonus, 0.9, 0.5);
                distanceCovered += speed;
            }
            
            brain[0].save("data/speed_drag_" + id + ".txt");
            brain[1].save("data/steer_drag_" + id + ".txt");
        }

        void reset() {
            x = startX;
            y = startY;
            angle = 0;
            distanceCovered = 0;
        }

        double getRay(BufferedImage track, double offset) {
            double rayRange = 400;
            for (int d = 0; d < rayRange; d += 20) {
                int cx = (int) (x + Math.cos(angle + offset) * d);
                int cy = (int) (y + Math.sin(angle + offset) * d);
                if (isOffTrack(track, cx, cy)) return d / rayRange;
            }
            return 1.0;
        }

        boolean isOffTrack(BufferedImage track, double px, double py) {
            if (px < 0 || px >= WIDTH || py < 400 || py >= 680) return true;
            return false;
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
        JFrame f = new JFrame("UNC");
        f.setExtendedState(JFrame.MAXIMIZED_BOTH);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.add(new Main());
        f.setVisible(true);
    }
}