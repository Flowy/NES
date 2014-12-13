package com.flowyk.neuron;

import com.flowyk.neuron.messenger.NeuronOutput;
import com.flowyk.neuron.messenger.TrainingInput;
import com.sun.istack.internal.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 11. 12. 2014.
 */
public abstract class NeuronTrainer {
    private static final Logger LOG = LoggerFactory.getLogger(NeuronTrainer.class);

    protected abstract BigDecimal calculateError(NeuronOutput output, BigDecimal desiredOutput);

    private Neuron neuron;

    public NeuronTrainer(@NotNull Neuron neuron) {
        this.neuron = neuron;
    }

    public void trainAll(List<TrainingInput> inputs) {
        List<NeuronOutput> lastOutputs;
        List<NeuronOutput> actualOutputs = Collections.emptyList();
        int iterations = 0;
        do {
            lastOutputs = actualOutputs;
            actualOutputs = trainOnce(inputs);
            LOG.debug("Iteration {}, output: {}", iterations++, actualOutputs);
        } while (!lastOutputs.equals(actualOutputs));
    }

    private List<NeuronOutput> trainOnce(List<TrainingInput> inputs) {
        List<NeuronOutput> results = new ArrayList<>(inputs.size());
        for (TrainingInput input: inputs) {
            NeuronOutput output = neuron.activate(input);
            results.add(neuron.activate(input));
            if (output.getOutput().compareTo(input.getDesiredOutput()) != 0) {
                neuron.train(input, calculateError(output, input.getDesiredOutput()));
            }
        }
        return results;
    }
}
