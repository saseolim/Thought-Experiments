using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;

namespace IMG
{
    public static class LamdaFilter
    {
        //option.R += 255 - 249; //기본 필터에서 흰색을 흰색으로 유지하는 필터
        //option.G += 255 - 246;
        //option.B += 255 - 218;
        public static readonly LamdaFilterOption DefaultFilter = new LamdaFilterOption();
        public static readonly LamdaFilterOption DefaultFilterWhiteSet = new LamdaFilterOption(
            new LamdaFilterOption().gamma,
            5 + 255 - 249,
            3 + 255 - 246,
            -10 + 255 - 218
            );
    }
    public struct LamdaFilterOption
    {
        public LamdaFilterOption(float gamma = 2.8f, Int32 R = 5, Int32 G = 3, Int32 B = -10,
            float Usefull = 0.03f, int Gap = 5, int Length = 100, float StartLamda = 380, float EndLamda = 776, bool UsingAll = true,
            bool UsingFlat = false,
            float RLamda = 700, float RLamdaStd = 30 * 2.35f,
            float GLamda = 560, float GLamdaStd = 25 * 2.5f,
            float BLamda = 420, float BLamdaStd = 25 * 2.6f,
            bool OneTen = true, bool GetGrayTone = false, float UsefullDown = 1000000f)
        {
            this.gamma = gamma;
            this.R = R;
            this.G = G;
            this.B = B;
            this.Usefull = Usefull;
            this.Gap = Gap;
            this.Length = Length;
            this.StartLamda = StartLamda;
            this.EndLamda = EndLamda;
            this.UsingAll = UsingAll;

            this.UsingFlat = UsingFlat;

            this.RLamda = RLamda;
            this.RLamdaStd = RLamdaStd;
            this.GLamda = GLamda;
            this.GLamdaStd = GLamdaStd;
            this.BLamda = BLamda;
            this.BLamdaStd = BLamdaStd;

            this.OneTen = OneTen;
            this.GetGrayTone = GetGrayTone;
            this.UsefullDown = UsefullDown;
        }
        private void Set(float gamma = 2.8f, Int32 R = 5, Int32 G = 3, Int32 B = -10,
            float Usefull = 0.03f, int Gap = 5, int Length = 100, float StartLamda = 380, float EndLamda = 776, bool UsingAll = true,
            bool UsingFlat = false,
            float RLamda = 700, float RLamdaStd = 30 * 2.35f,
            float GLamda = 560, float GLamdaStd = 25 * 2.5f,
            float BLamda = 420, float BLamdaStd = 25 * 2.6f,
            bool OneTen = true, bool GetGrayTone = false, float UsefullDown = 1000000f)
        {
            this.gamma = gamma;
            this.R = R;
            this.G = G;
            this.B = B;
            this.Usefull = Usefull;
            this.Gap = Gap;
            this.Length = Length;
            this.StartLamda = StartLamda;
            this.EndLamda = EndLamda;
            this.UsingAll = UsingAll;

            this.UsingFlat = UsingFlat;

            this.RLamda = RLamda;
            this.RLamdaStd = RLamdaStd;
            this.GLamda = GLamda;
            this.GLamdaStd = GLamdaStd;
            this.BLamda = BLamda;
            this.BLamdaStd = BLamdaStd;

            this.OneTen = OneTen;
            this.GetGrayTone = GetGrayTone;
            this.UsefullDown = UsefullDown;
        }
        public LamdaFilterOption()
        {
            Set(); //2.8f, 5, 3, -10, 0.03f, 5, 100, 380, 776, true
        }

        public float gamma { get; set; }
        public Int32 R { get; set; }
        public Int32 G { get;  set; }
        public Int32 B { get; set; }
        public float Usefull { get;  set; }
        public int Gap { get;  set; }
        public int Length { get;  set; }
        public float StartLamda { get;  set; }
        public float EndLamda { get;  set; }
        public bool UsingAll { get;  set; }
        public bool UsingFlat { get; set; }
        public float RLamda { get; set; }
        public float RLamdaStd { get; set; }
        public float GLamda { get; set; }
        public float GLamdaStd { get; set; }
        public float BLamda { get; set; }
        public float BLamdaStd { get; set; }
        public bool OneTen { get; set; }
        public bool GetGrayTone { get; set; }
        public float UsefullDown { get; set; }

    }
    public partial class LamdaImgFilterBlockControl
    {
        public Bitmap[,] Block;
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int WidthBlock { get; private set; }
        public int HeightBlock { get; private set; }
        public int BlockSize { get; private set; }
        public int MaxThread { get; private set; }
        public bool GPU { get; private set; }
        public LamdaFilterOption Option;
        public LamdaImgFilterBlockControl(Bitmap bitmap, LamdaFilterOption Option, int BlockSize = 256, int MaxThread = 4, bool GPU = false)
        {
            this.BlockSize = BlockSize;
            this.Width = bitmap.Width;
            this.Height = bitmap.Height;
            this.Option = Option;
            this.MaxThread = MaxThread;
            this.GPU = GPU;

            Block = MakeBlocks(bitmap);
        }

        private Bitmap[,] MakeBlocks(Bitmap bitmap)
        {
            this.WidthBlock = (int)Math.Ceiling((decimal)Width / (decimal)BlockSize);
            this.HeightBlock = (int)Math.Ceiling((decimal)Height / (decimal)BlockSize);
            Bitmap[,] Out = new Bitmap[WidthBlock, HeightBlock];

            for(int x = 0; x < WidthBlock; x++)
            {
                for(int y = 0; y < HeightBlock; y++)
                {
                    Out[x, y] = MakeBlock(bitmap, x, y);
                }
            }

            return Out;
        }
        private Bitmap MakeBlock(Bitmap bitmap, int Wx, int Hy)
        {
            int xStart = Wx * BlockSize, yStart = Hy * BlockSize;
            int xEnd, yEnd;
            if(Wx == WidthBlock - 1)
            {
                xEnd = bitmap.Width;
            }
            else
            {
                xEnd = (Wx + 1) * BlockSize;
            }

            if(Hy == HeightBlock - 1)
            {
                yEnd = bitmap.Height;
            }
            else
            {
                yEnd = (Hy + 1) * BlockSize;
            }

            int xLength = xEnd - xStart, yLength = yEnd - yStart;
            Bitmap Out = new Bitmap(xLength, yLength);

            int i = 0,ii = 0;
            for (int x = xStart; x < xEnd; x++)
            {
                for (int y = yStart; y < yEnd; y++)
                {
                    Out.SetPixel(i,ii,bitmap.GetPixel(x,y));

                    ii++;
                }
                ii = 0;
                i++;
            }

            return Out;
        }

        public Bitmap GetFiltering()
        {
            Bitmap Out = new Bitmap(Width, Height);

            SetFB[,] BF = new SetFB[WidthBlock, HeightBlock];

            List<Thread> threads = new List<Thread>();

            //for (int x = 0; x < WidthBlock; x++)
            //{
            //    for (int y = 0; y < HeightBlock; y++)
            //    {
            //                Bitmap bitmap = (Bitmap)Block[x, y].Clone();
            //                BF[x, y] = new SetFB(bitmap, Option);
            //    }
            //}

            int xf = 0, yf = 0;
            foreach (Bitmap bitmapF in Block)
            {
                BF[xf, yf] = new SetFB((Bitmap)bitmapF.Clone(),Option, GPU);
                threads.Add(new Thread(new ThreadStart(BF[xf,yf].Set)));

                lock(BF)
                {
                    lock(typeof(Tensorflow.Binding)) //이상해지면 삭제
                    {
                        threads[threads.Count - 1].Start();
                    }
                }

                //if (threads.Count % MaxThread == 0)
                //if (threads.Count(t => t.IsAlive) >= MaxThread)
                //{
                //    foreach (Thread thread in threads)
                //        thread.Join();
                //}
                while(threads.Count(t => t.IsAlive) >= MaxThread)
                {
                    Thread.Sleep(10);
                }

                if (yf + 1 < HeightBlock)
                    yf++;
                else
                {
                    yf = 0;
                    xf++;
                }

                if (!(xf < WidthBlock))
                    break;
            }

            foreach (Thread thread in threads)
            {
                thread.Join();
            }

            for (int x = 0; x < WidthBlock; x++)
            {
                for (int y = 0; y < HeightBlock; y++)
                {
                    Con(Out, BF[x, y].BF, x, y);
                }
            }

            return Out;
        }
        
        private void Con(Bitmap bitmap, Bitmap BF, int Wx, int Hy)
        {
            int xStart = Wx * BlockSize, yStart = Hy * BlockSize;
            int xEnd, yEnd;
            if (Wx == WidthBlock - 1)
            {
                xEnd = Width;
            }
            else
            {
                xEnd = (Wx + 1) * BlockSize;
            }

            if (Hy == HeightBlock - 1)
            {
                yEnd = Height;
            }
            else
            {
                yEnd = (Hy + 1) * BlockSize;
            }

            int xLength = xEnd - xStart, yLength = yEnd - yStart;

            int i = 0, ii = 0;
            for (int x = xStart; x < xEnd; x++)
            {
                for (int y = yStart; y < yEnd; y++)
                {
                    Color color = BF.GetPixel(i, ii);
                    bitmap.SetPixel(x, y, color);

                    ii++;
                }
                ii = 0;
                i++;
            }
        }

        public class SetFB
        {
            public Bitmap BF;
            Bitmap Block;
            LamdaFilterOption Option;
            bool GPU;
            public void Set()
            {
                if (GPU)
                    tf.enable_eager_execution();
                else
                {
                    LamdaImgFilter imgFilter = new LamdaImgFilter(Block, Option);
                    Bitmap Tmp = imgFilter.GetBitmap();

                    BF = Tmp;
                }
            }
            public SetFB(Bitmap Block, LamdaFilterOption Option,bool GPU)
            {
                this.Block = Block;
                this.Option = Option;
                this.BF = new Bitmap(1, 1);
                this.GPU = GPU;
            }
        }

    }
}
