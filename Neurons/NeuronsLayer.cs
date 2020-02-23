using System;
using System.Collections.Generic;

namespace Neurons
{
    public class NeuronsLayer
    {
        private List<Neuron[]> _neurons = new List<Neuron[]>();
        private List<double[]> _weights = new List<double[]>();

        private List<double[]> _inputParams;
        private List<double[]> _outputParams;

        private Random _random = new Random();

        private int _hiddenLayerInsertIndex = 1;

        public NeuronsLayer(List<double[]> inputParams, List<double[]> outputParams)
        {
            var inputs = GenerateNeurons(inputParams[0].Length);
            var outputs = GenerateNeurons(outputParams[0].Length);

            _inputParams = inputParams;
            _outputParams = outputParams;

            _neurons.Add(inputs);
            _neurons.Add(outputs);
        }

        public void GenerateHiddenLayer(int neuronsCount)
        {
            var neurons = GenerateNeurons(neuronsCount);

            _neurons.Insert(_hiddenLayerInsertIndex, neurons);
            _hiddenLayerInsertIndex++;
        }

        public void GenerateRandomWeights()
        {
            for(int i = 0; i < _neurons.Count - 1; i++)
            {
                double[] weights = GenerateRandomWeights(_neurons[i].Length * _neurons[i + 1].Length, _random);
                _weights.Add(weights);
            }
        }

        public Neuron[] GenerateNeurons(int neuronsCount)
        {
            Neuron[] neurons = new Neuron[neuronsCount];
            for (int i = 0; i < neuronsCount; i++)
            {
                neurons[i] = new Neuron();
            }

            return neurons;
        }

        public double[] GenerateRandomWeights(int weightsCount, Random random)
        {
            double[] weights = new double[weightsCount];
            for(int i = 0; i < weightsCount; i++)
            {
                weights[i] = random.NextDouble();
            }

            return weights;
        }

        public void Learn(double desireAccuracy)
        {
            for (int j = 0; true; j++)
            {
                foreach (var neurons in _neurons)
                {
                    ClearNeurons(neurons);
                }

                SetInputNeurons(_inputParams[j % _inputParams.Count]);
                SetOutputNeurons(_outputParams[j % _outputParams.Count]);

                for (int i = 0; i < _neurons.Count - 1; i++)
                {
                    Forwards(_neurons[i], _neurons[i + 1], _weights[i]);
                }

                FindOutputError(j % _outputParams.Count);

                for (int i = 0; i < _neurons.Count - 1; i++)
                {
                    FindError(_neurons[i], _neurons[i + 1], _weights[i]);
                }

                double error = 0;
                for (int i = 0; i < _neurons[_neurons.Count - 1].Length; i++)
                {
                    error += _neurons[_neurons.Count - 1][i].Error;
                }
                error /= _neurons[_neurons.Count - 1].Length;

                if(Math.Abs(error) <= (1 - desireAccuracy))
                {
                    break;
                }

                for (int i = 0; i < _neurons.Count - 1; i++)
                {
                    CorrectingWeights(_neurons[i], _neurons[i + 1], _weights[i], 0.3f);
                }
            }
        }

        public double[] GetResult(double[] inputs)
        {
            SetInputNeurons(inputs);

            for (int i = 0; i < _neurons.Count - 1; i++)
            {
                Forwards(_neurons[i], _neurons[i + 1], _weights[i]);
            }

            double[] results = new double[_neurons[_neurons.Count - 1].Length];
            for(int i = 0; i < results.Length; i++)
            {
                results[i] = _neurons[_neurons.Count - 1][i].Value;
            }

            return results;
        }

        public void Forwards(Neuron[] inputNeurons, Neuron[] outputNeurons, double[] weights)
        {
            for(int i = 0; i < outputNeurons.Length; i++)
            {
                for(int j = 0; j < inputNeurons.Length; j++)
                {
                    outputNeurons[i].Value += inputNeurons[j].Value * weights[outputNeurons.Length*j + i];
                }
                outputNeurons[i].Value = SigmoidFunction(outputNeurons[i].Value);
            }
        }

        public void FindError(Neuron[] inputNeurons, Neuron[] outputNeurons, double[] weights)
        {
            for(int i = 0; i < inputNeurons.Length; i++)
            {
                for(int j = 0; j < outputNeurons.Length; j++)
                {
                    inputNeurons[i].Error += outputNeurons[j].Error * weights[outputNeurons.Length*i + j];
                }
            }
        }

        public void CorrectingWeights(Neuron[] inputNeurons, Neuron[] outputNeurons, double[] weights, double learnCoefficient)
        {
            for(int i = 0; i < inputNeurons.Length; i++)
            {
                for(int j = 0; j < outputNeurons.Length; j++)
                {
                    weights[outputNeurons.Length * i + j] += learnCoefficient * outputNeurons[j].Error * (outputNeurons[j].Value * (1 - outputNeurons[j].Value)) * inputNeurons[i].Value; ;
                }
            }
        }

        public void ClearNeurons(Neuron[] neurons)
        {
            for(int i = 0; i < neurons.Length; i++)
            {
                neurons[i].Error = 0;
                neurons[i].Value = 0;
            }
        }

        public void SetInputNeurons(double[] inputs)
        {
            if (inputs.Length == _neurons[0].Length)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    _neurons[0][i].Value = inputs[i];
                }
            }
        }

        public void SetOutputNeurons(double[] outputs)
        {
            if(outputs.Length == _neurons[_neurons.Count - 1].Length)
            {
                for(int i = 0; i < outputs.Length; i++)
                {
                    _neurons[_neurons.Count - 1][i].Value = outputs[i];
                }
            }
        }

        public void FindOutputError(int outputIndex)
        {
            if(_outputParams[outputIndex].Length == _neurons[_neurons.Count - 1].Length)
            {
                for(int i = 0; i < _outputParams[outputIndex].Length; i++)
                {
                    _neurons[_neurons.Count - 1][i].Error = _neurons[_neurons.Count - 1][i].Value - _outputParams[outputIndex][i];
                }
            }
        }

        public double SigmoidFunction(double value)
        {
            return 1 / (1 + Math.Pow(Math.E, -value));
        }
    }
}
