/*

By Tim Ehrensberger, 2017

*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace PatternRecognition
{
        public class NeuralNetwork
        {            
            public List<Neuron> InputLayer { get; set; }
            public List<Neuron> HiddenLayer { get; set; }
            public List<Neuron> OutputLayer { get; set; }
            public double alpha { get; set; }            
            static Random random = new Random();

            public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, TextBox txtLearn)
            {
                alpha = float.Parse(txtLearn.Text);                
                InputLayer = new List<Neuron>();
                HiddenLayer = new List<Neuron>();
                OutputLayer = new List<Neuron>();

                for (int i = 0; i < inputSize; i++)
                    InputLayer.Add(new Neuron());

                for (int i = 0; i < hiddenSize; i++)
                    HiddenLayer.Add(new Neuron(InputLayer));

                for (int i = 0; i < outputSize; i++)
                    OutputLayer.Add(new Neuron(HiddenLayer));
            }

            //Sends the inputs once through the network
            public void Train(params double[] inputs)
            {
                int i = 0;
                InputLayer.ForEach(a => a.Value = inputs[i++]); //Assign input data to input-neurons
                HiddenLayer.ForEach(a => a.Calc_Value());       //Hidden Calc
                OutputLayer.ForEach(a => a.Calc_Value());       //Outuput Calc 
            }

            //Sends the inputs once through the network and returns the output
            public double[] Compute(params double[] inputs)
            {
                Train(inputs);
                return OutputLayer.Select(a => a.Value).ToArray();
            }

            //Sends the outputs once backwards through the network to adjust the weights
            public void Backpropagate(params double[] targets)
            {
                int i = 0;
                OutputLayer.ForEach(a => a.Calc_ConstantPartOfGradient(targets[i++]));    //Gradient of output layer
                HiddenLayer.ForEach(a => a.Calc_ConstantPartOfGradient());                //Gradient of hidden layer
                           
                OutputLayer.ForEach(a => a.UpdateWeights(alpha)); //Adjust weights in hidden layer
                HiddenLayer.ForEach(a => a.UpdateWeights(alpha)); //Adjust weights in hidden layer
            }

            //Returns a random number between -1 and 1
            public static double NextRandom()
            {
                return 2 * random.NextDouble() - 1;
            }

            public static double SigmoidFunction(double x)
            {
                //The first part is to round the values at a reasonable point
                if (x < -50.0)
                    return 0.0;
                else if (x > 50.0)
                    return 1.0;

                //actual function
                return 1.0 / (1.0 + Math.Exp(-x));
            }
        }

        public class Neuron
        {
            public List<Synapse> IncomingSynapses { get; set; }
            public List<Synapse> OutgoingSynapses { get; set; }
            public double Bias { get; set; }
            public double BiasDelta { get; set; }
            public double PartOfGradient { get; set; }
            public double Value { get; set; }

            //Constructor for input neurons
            public Neuron()
            {
                IncomingSynapses = new List<Synapse>();
                OutgoingSynapses = new List<Synapse>();
                Bias = NeuralNetwork.NextRandom();
            }

            //Constructor for other neurons        
            //First the other constructor is executed thanks to :this() "self-inheritance"
            public Neuron(List<Neuron> previousNeurons) : this()
            {
                foreach (var previousNeuron in previousNeurons)
                {
                    var synapse = new Synapse(previousNeuron, this);
                    previousNeuron.OutgoingSynapses.Add(synapse);
                    IncomingSynapses.Add(synapse);
                }
            }

            public virtual double Calc_Value()
            {
                return Value = NeuralNetwork.SigmoidFunction(IncomingSynapses.Sum(a => a.Weight * a.FromNeuron.Value) + Bias);
            }

            //Gradient in output neurons
            public double Calc_ConstantPartOfGradient(double target)
            {
                return PartOfGradient = (target - Value) *  Value * (1 - Value);
            }

            //Gradient in hidden neurons
            public double Calc_ConstantPartOfGradient()
            {
                //First part is a sum because there are more than 1 ouput neurons
                return PartOfGradient = OutgoingSynapses.Sum(a => a.ToNeuron.PartOfGradient * a.Weight) * Value * (1 - Value); 
            }

            public void UpdateWeights(double alpha)
            {
                foreach (var s in IncomingSynapses)
                {                    
                    s.WeightDelta = alpha * PartOfGradient * s.FromNeuron.Value;
                    //Info: Should be identical to the theory in the paper 
                    //except that the first part of the factors in PartOfGradient is a sum here, because there are more than one ouput neurons
                    s.Weight += s.WeightDelta;
                }
            }
        }
       
        public class Synapse
        {
            public Neuron FromNeuron { get; set; }
            public Neuron ToNeuron { get; set; }
            public double Weight { get; set; }
            public double WeightDelta { get; set; }

            public Synapse(Neuron fromNeuron, Neuron toNeuron)
            {
                FromNeuron = fromNeuron;
                ToNeuron = toNeuron;
                Weight = NeuralNetwork.NextRandom();
            }
        }
    
}
