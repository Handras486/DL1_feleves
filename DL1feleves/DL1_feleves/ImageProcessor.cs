using CNTK;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;

namespace DL1_feleves
{
    public class ImageProcessor
    {

        public int ImgCountPerDS { get; set; }
        public int ImgCount { get; set; }

        public static System.Drawing.Size ImgSize { get; set; } = new System.Drawing.Size(96, 96);
        public List<float> ImageInput { get; set; } = new List<float>();
        public List<float> ImageOutput { get; set; } = new List<float>();

        public static int OutputSize { get; set; } = Enum.GetNames(typeof(AnimalTypes)).Length;
        public static int InputSize { get; set; } = ImgSize.Width * ImgSize.Height;

        enum AnimalTypes
        {
            butterfly,
            cat,
            chicken,
            cow,
            dog,
            elephant,
            horse,
            sheep,
            spider,
            squirrel,
        }



        public ImageProcessor(string dbtype, int ImgCountPerDS)
        {
            this.ImgCountPerDS = ImgCountPerDS;
            ImgCount = ImgCountPerDS * OutputSize;

            foreach (AnimalTypes type in Enum.GetValues(typeof(AnimalTypes)))
            {
                LoadImages(dbtype, type);
            }

            Shuffle();
        }


        void LoadImages(string dbtype, AnimalTypes animaltype)
        {
            string[] filepaths = Directory.GetFiles("../../../../DL1_DB/"+ dbtype +"-img/" + animaltype);

            for (int i = 0; i < ImgCountPerDS; i++)
            {

                Mat img = CvInvoke.Imread(filepaths[i], Emgu.CV.CvEnum.ImreadModes.Grayscale);
                CvInvoke.Resize(img, img, ImgSize);
                CvInvoke.Canny(img, img, 255/3, 255);

                foreach (var item in img.GetRawData())
                {
                    ImageInput.Add(item);
                }

                for (int j = 0; j < OutputSize; j++)
                {
                    ImageOutput.Add(j == (int)animaltype ? 1.0f : 0.0f);
                }

            }

        }

        public String DataToString(int IstartRange,int OstartRange, int index)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < InputSize; i++)
            {
                if (i % ImgSize.Width == 0)
                {
                    sb.Append("\n");
                }
                float data = ImageInput[IstartRange + index * InputSize + i];
                if (data > 200)
                    sb.Append("\u2593");
                else
                if (data > 150)
                    sb.Append("\u2592");
                else
                if (data > 100)
                    sb.Append("\u2591");
                else
                    sb.Append(" ");
            }

            sb.Append("\nAnswer:[");
            for (int d = 0; d <= 9; d++)
            {
                if (d != 0) sb.Append(", ");
                sb.Append(String.Format("{0:0.00}", ImageOutput[OstartRange + index * OutputSize + d]));
            }
            sb.Append("]\n");

            return sb.ToString();
        }

        public void Shuffle()
        {
            Random rnd = new Random();
            for (int swapI = 0; swapI < ImgCount; swapI++)
            {
                var a = rnd.Next(ImgCount);
                var b = rnd.Next(ImgCount);
                if (a != b)
                {
                    float T;
                    for (int i = 0; i < InputSize; i++)
                    {
                        T = ImageInput[a * InputSize + i];
                        ImageInput[a * InputSize + i] = ImageInput[b * InputSize + i];
                        ImageInput[b * InputSize + i] = T;
                    }

                    for (int j = 0; j < OutputSize; j++)
                    {
                        T = ImageOutput[a * OutputSize + j];
                        ImageOutput[a * OutputSize + j] = ImageOutput[b * OutputSize + j];
                        ImageOutput[b * OutputSize + j] = T;
                    }
                }
            }
        }

    }
}
