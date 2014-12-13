package com.flowyk.neuralnetworks.messenger;

import com.flowyk.neuralnetworks.transferfunction.TransferFunction;
import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 11. 12. 2014.
 */
public class NeuronOutput {
    private List<BigDecimal> sensorOutputs;
    private TransferFunction transferFunction;

    public NeuronOutput(@NotNull List<BigDecimal> sensorOutputs, @NotNull TransferFunction transferFunction) {
        this.sensorOutputs = Collections.unmodifiableList(sensorOutputs);
        this.transferFunction = transferFunction;
    }

    public List<BigDecimal> getSensorOutputs() {
        return sensorOutputs;
    }

    public BigDecimal getOutput() {
        BigDecimal sum = BigDecimal.ZERO;
        for (BigDecimal output: sensorOutputs) {
            sum = sum.add(output);
        }
        return transferFunction.transfer(sum);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NeuronOutput)) return false;

        NeuronOutput that = (NeuronOutput) o;

        if (!sensorOutputs.equals(that.sensorOutputs)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return sensorOutputs.hashCode();
    }

    @Override
    public String toString() {
        return "NeuronOutput{" +
                "sensorOutputs=" + sensorOutputs +
                '}';
    }
}
