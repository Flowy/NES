package com.flowyk.neuron.transferfunction;

/**
 * Created by Lukas on 6. 12. 2014.
 */
public class SaturatedLinearFunction implements TransferFunction {
    @Override
    public double transfer(Number sum) {
        double sumDouble = sum.doubleValue();
        if (sumDouble >= 1) {
            return 1;
        } else if (sumDouble < 0) {
            return 0;
        } else {
            return sumDouble;
        }
    }
}
