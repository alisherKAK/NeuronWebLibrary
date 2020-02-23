using System;
using System.Collections.Generic;

namespace Neurons
{
    class Program
    {
        static void Main(string[] args)
        {
            List<double[]> inputs = new List<double[]>()
            {
                new double[3]{0, 0, 0},
                new double[3]{0, 1, 1},
                new double[3]{1, 0, 1},
                new double[3]{1, 1, 0}
            };

            List<double[]> outputs = new List<double[]>()
            {
                new double[1]{0},
                new double[1]{0},
                new double[1]{1},
                new double[1]{1}
            };

            NeuronsLayer layer = new NeuronsLayer(inputs, outputs);
            layer.GenerateHiddenLayer(2);
            layer.GenerateRandomWeights();

            layer.Learn(0.9);

            var result = layer.GetResult(new double[3] { 1, 1, 1 });

            for(int i = 0; i < result.Length; i++)
            {
                Console.WriteLine($"Result {i}: {result[i]}");
            }
            Console.ReadLine();
        }
    }
}
