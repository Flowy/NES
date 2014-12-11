package com.flowyk.neuron;

import com.flowyk.neuron.messenger.ActivationInput;
import com.flowyk.neuron.messenger.NeuronOutput;

import java.math.BigDecimal;
import java.util.List;

/**
 * Created by Lukas on 11. 12. 2014.
 */
public interface Neuron  {
    void train(ActivationInput input, BigDecimal error);
    NeuronOutput activate(ActivationInput input);
    List<BigDecimal> getWeights();
}