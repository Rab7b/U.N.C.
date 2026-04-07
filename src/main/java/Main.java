import ai.Neuron;
import javax.swing.*;
import java.awt.*;

public class Main extends JPanel {

    private double x = 50; 
    private double y = 200;
    private final double finishX = 550;

    private final int wallX = 300;
    private final int wallY = 100;
    private final int wallWidth = 20;
    private final int wallHeight = 200;

    private Neuron[] brain;

    public Main() {
        brain = new Neuron[2];

        brain[0] = new Neuron(2, new double[]{0.5, 0.1}, 0.1, 1, brain);
        brain[1] = new Neuron(2, new double[]{0.5, 0.1}, 0.1, 1, brain);

        new Timer(10, e -> {

            double distToWall = (wallX - x) / 600.0;

            brain[0].setInputs(new double[]{ x / 600.0, distToWall });
            brain[1].setInputs(new double[]{ y / 400.0, distToWall });
            
            double move = brain[0].predict(); 
            double move2 = brain[1].predict();

            double oldX = x;
            double oldY = y;
            double oldDist = Math.abs(finishX - x);

            x += move * 8; 
            y += move2 * 8;

            if (x + 40 > wallX && x < wallX + wallWidth && y + 40 > wallY && y < wallY + wallHeight) {

                x = oldX;
                y = oldY;
                brain[0].motivate(-5.0, 0.5, x / 600.0); 
                brain[1].motivate(-5.0, 0.5, y / 400.0);
            }

            double newDist = Math.abs(finishX - x);
            
            double reward = (newDist < oldDist) ? 1.0 : -1.0;
            
            brain[0].motivate(reward, 0.5, x / 600.0);
            brain[1].motivate(reward, 0.5, y / 400.0);

            if (x >= finishX) {
                System.out.println("WINNER! Wall bypassed.");
                x = 50; 
                y = 200; 
            }
            
            if (x < 0) x = 0; 
            if (y < 0) y = 0;
            if (y > 360) y = 360;

            repaint();
        }).start();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        
        g.setColor(Color.GREEN); 
        g.fillRect((int)finishX, 0, 20, 400);
        
        g.setColor(Color.RED);
        g.fillRect(wallX, wallY, wallWidth, wallHeight);
        
        g.setColor(Color.BLUE); 
        g.fillRect((int)x, (int)y, 40, 40);
    }

    public static void main(String[] args) {
        System.loadLibrary("neuron_logic");
        JFrame f = new JFrame("AI vs Wall");
        f.add(new Main());
        f.setSize(600, 400);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setVisible(true);
    }
}
