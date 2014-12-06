package com.flowyk.neuron.transferfunction;

/**
 * Created by Lukas on 6. 12. 2014.
 */
public class BipolarnaFunkcia implements TransferFunction {

    private double threshold;

    public BipolarnaFunkcia(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public double transfer(Number sum) {
        double sumDouble = sum.doubleValue();
        if (sumDouble > threshold) {
            return 1;
        } else if (sumDouble < -threshold) {
            return -1;
        } else {
            return 0;
        }
    }
}
