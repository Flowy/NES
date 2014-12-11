package com.flowyk.neuron;

import com.flowyk.neuron.messenger.NeuronOutput;
import com.flowyk.neuron.messenger.TrainingInput;
import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 11. 12. 2014.
 */
public abstract class NeuronTrainer {

    protected abstract BigDecimal calculateError(NeuronOutput output, BigDecimal desiredOutput);

    private Neuron neuron;

    public NeuronTrainer(@NotNull Neuron neuron) {
        this.neuron = neuron;
    }

    public void trainAll(List<TrainingInput> inputs) {
        List<NeuronOutput> lastOutputs;
        List<NeuronOutput> actualOutputs = Collections.emptyList();
        do {
            lastOutputs = actualOutputs;
            actualOutputs = trainOnce(inputs);
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
