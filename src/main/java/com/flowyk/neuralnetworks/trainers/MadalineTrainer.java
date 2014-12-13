package com.flowyk.neuralnetworks.trainers;

import com.flowyk.neuralnetworks.messenger.TrainingInput;
import com.flowyk.neuralnetworks.neuron.McCullochPittsNeuron;
import com.flowyk.neuralnetworks.neuron.Neuron;
import com.flowyk.neuralnetworks.transferfunction.SignFunction;
import com.flowyk.neuralnetworks.transferfunction.TransferFunction;

import java.math.BigDecimal;
import java.util.List;

/**
 * Created by Lukas on 13. 12. 2014.
 */
public class MadalineTrainer {

    private TransferFunction transferFunction = new SignFunction();
    private Neuron Z1 = new McCullochPittsNeuron(2, transferFunction, BigDecimal.valueOf(0.1d));
    private Neuron Z2 = new McCullochPittsNeuron(2, transferFunction, BigDecimal.valueOf(0.1d));
    private Neuron Y = new McCullochPittsNeuron(2, transferFunction, BigDecimal.valueOf(0.1d));


    public void trainAll(List<TrainingInput> inputs) {

    }
}
