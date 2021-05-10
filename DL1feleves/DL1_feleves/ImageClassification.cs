using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;

namespace DL1_feleves
{
    public class ImageClassification
    {
        const int batchSize = 250;
        const int epochCount = 30;

        readonly Variable x;
        readonly Function y;

        readonly int hiddenLayerCount;
        readonly int hiddenNeuronCount;


        public ImageClassification(int hiddenLayerCount, int hiddenNeuronCount)
        {
            this.hiddenLayerCount = hiddenLayerCount;
            this.hiddenNeuronCount = hiddenNeuronCount;

            List<int> layers = new List<int>();
            layers.Add(ImageProcessor.InputSize);
            for (int i = 0; i < hiddenLayerCount; i++)
            {
                layers.Add(hiddenNeuronCount);
            }
            layers.Add(ImageProcessor.OutputSize);

            x = Variable.InputVariable(new int[] { layers[0] }, DataType.Float);

            Function lastLayer = x;
            for (int i = 0; i < layers.Count - 1; i++)
            {
                Parameter weight = new Parameter(new int[] { layers[i + 1], layers[i] }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                Parameter bias = new Parameter(new int[] { layers[i + 1] }, DataType.Float, CNTKLib.GlorotNormalInitializer());

                Function times = CNTKLib.Times(weight, lastLayer);
                Function plus = CNTKLib.Plus(times, bias);

                if (i != layers.Count - 2)
                    lastLayer = CNTKLib.Sigmoid(plus);
                else
                    lastLayer = CNTKLib.Softmax(plus);
            }
            y = lastLayer;
        }

        public ImageClassification(string filepath)
        {

            y = Function.Load(filepath, DeviceDescriptor.CPUDevice);
            x = y.Arguments.First(x => x.Name == "x");
        }

        public void Train(ImageProcessor trainDS)
        {
            Variable yt = Variable.InputVariable(new int[] { ImageProcessor.OutputSize }, DataType.Float);
            Function loss = CNTKLib.CrossEntropyWithSoftmax(y, yt);
            Function err = CNTKLib.ClassificationError(y, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(1.0, batchSize));
            Trainer trainer = Trainer.CreateTrainer(y, loss, err, new List<Learner>() { learner });

            for (int epochI = 0; epochI <= epochCount; epochI++)
            {
                double sumLoss = 0;
                double sumError = 0;

                for (int batchI = 0; batchI < trainDS.ImgCount / batchSize ; batchI++)
                {
                    Value x_value = Value.CreateBatch(x.Shape, trainDS.ImageInput.GetRange(batchI * batchSize * ImageProcessor.InputSize, batchSize * ImageProcessor.InputSize), DeviceDescriptor.CPUDevice);
                    Value yt_value = Value.CreateBatch(yt.Shape, trainDS.ImageOutput.GetRange(batchI * batchSize * ImageProcessor.OutputSize, batchSize * ImageProcessor.OutputSize), DeviceDescriptor.CPUDevice);
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                    sumError += trainer.PreviousMinibatchEvaluationAverage() * trainer.PreviousMinibatchSampleCount();
                }

                Console.WriteLine(String.Format("{0}\t{1:0.0000}\t{2:0.0000}", epochI, sumLoss / trainDS.ImgCount, 1.0 - sumError / trainDS.ImgCount));
                Console.WriteLine("Sumloss: " + sumLoss.ToString() +"\tSumError: " + sumError.ToString());
            }

            trainDS.Shuffle();
        }

        public void Evaluate(ImageProcessor DS, out double lossValue, out double accValue)
        {
            Variable yt = Variable.InputVariable(new int[] { ImageProcessor.OutputSize }, DataType.Float);
            Function loss = CNTKLib.CrossEntropyWithSoftmax(y, yt);
            Function err = CNTKLib.ClassificationError(y, yt);

            Evaluator evaluator_loss = CNTKLib.CreateEvaluator(loss);
            Evaluator evaluator_err = CNTKLib.CreateEvaluator(err);

            double sumEval = 0;
            double sumLoss = 0;

            for (int batchI = 0; batchI < DS.ImgCount / batchSize; batchI++)
            {
                Value x_value = Value.CreateBatch(x.Shape, DS.ImageInput.GetRange(batchI * batchSize * ImageProcessor.InputSize, batchSize * ImageProcessor.InputSize), DeviceDescriptor.CPUDevice);
                Value yt_value = Value.CreateBatch(yt.Shape, DS.ImageOutput.GetRange(batchI * batchSize * ImageProcessor.OutputSize, batchSize * ImageProcessor.OutputSize), DeviceDescriptor.CPUDevice);
                var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                sumLoss += evaluator_loss.TestMinibatch(inputDataMap) * batchSize;
                sumEval += evaluator_err.TestMinibatch(inputDataMap) * batchSize;

            }
            lossValue = sumLoss / DS.ImgCount;
            accValue = 1 - sumEval / DS.ImgCount;
        }


        void SaveNetwork(string filepath)
        {
            y.Save("Models/" + filepath + ".model");
        }  

        public String VisibleTest(ImageProcessor ds,int countPerOutput , int count)
        {
            StringBuilder sb = new StringBuilder();
            int IstartRange = 0;
            int OstartRange = 0;

            for (int i = 0; i < ImageProcessor.OutputSize; i++)
            {
                Value x_value = Value.CreateBatch(x.Shape, ds.ImageInput.GetRange(IstartRange, count * ImageProcessor.InputSize), DeviceDescriptor.CPUDevice);
                var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, x_value }
                    };
                var outputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { y, null }
                    };

                y.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
                IList<IList<float>> resultValue = outputDataMap[y].GetDenseData<float>(y);


                for (int j = 0; j < count; j++)
                {
                    sb.Append(ds.DataToString(IstartRange, OstartRange , j));
                    sb.Append("\nResult:[");
                    int max = 0;
                    for (int d = 0; d <= 9; d++)
                    {
                        if (resultValue[j][d] > resultValue[j][max])
                            max = d;
                        if (d != 0) sb.Append(", ");
                        sb.Append(String.Format("{0:0.00}", resultValue[j][d]));
                    }
                    sb.Append("]\nPrediction:").Append(max).Append("\n\n");
                }

                IstartRange += countPerOutput * ImageProcessor.InputSize;
                OstartRange += countPerOutput * ImageProcessor.OutputSize;

            }
            return sb.ToString();
        }

    static void Main(string[] args)
        {
            ImageProcessor trainDS = new ImageProcessor("train", 1000);
            ImageProcessor testDS = new ImageProcessor("test", 100);
            ImageClassification ImgClass = new ImageClassification(3, 10);
            ImgClass.Train(trainDS);
            ImgClass.Evaluate(trainDS, out double trainLoss, out double trainAcc);
            ImgClass.Evaluate(testDS, out double testLoss, out double testAcc);

            Console.WriteLine(ImgClass.VisibleTest(testDS, testDS.ImgCountPerDS, 2));

            ImgClass.SaveNetwork("Model1");

            Console.WriteLine(String.Format("Final evaluation:\t{0:0.0000}\t{1:0.0000}\t{2:0.0000}\t{3:0.0000}", trainLoss, trainAcc, testLoss, testAcc));
            Console.ReadLine();
        }
    }
}
