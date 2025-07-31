using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IMG
{
    public class LamdaImgFilter
    {
        public LamdaSetting LamdaSet;
        public Px[,] Img;
        public int Width { get; private set; }
        public int Height { get; private set; }
        public float gamma { get; private set; }

        public float PxWhiteAvg { get; private set; }
        public float PxWhiteMax { get; private set; }
        public float PxWhiteAU { get; private set; }
        public int Gap { get; private set; }
        public float Usefull { get; private set; }
        public Int32 LuR { get; private set; }
        public Int32 LuG { get; private set; }
        public Int32 LuB { get; private set; }
        public float RLamda { get; set; }
        public float RLamdaStd { get; set; }
        public float GLamda { get; set; }
        public float GLamdaStd { get; set; }
        public float BLamda { get; set; }
        public float BLamdaStd { get; set; }
        public bool OneTen { get; set; }
        public bool GetGrayTone { get; set; }
        public float UsefullDown { get; set; }

        public class LamdaSetting
        {
            public int Length { get; private set; }
            public float Start { get; private set; }
            public float End { get; private set; }
            public float[] Lamda;
            public LamdaSetting(int Length = 100, float StartLamda = 380, float EndLamda = 776)
            {
                this.Start = StartLamda;
                this.End = EndLamda;
                this.Length = Length;
                this.Lamda = new float[this.Length];

                float floatOne = (End - Start) / (this.Length - 1);

                for (int i = 0; i < Lamda.Length; i++)
                {
                    Lamda[i] = Start + floatOne * i;
                }
            }
        }
        public partial class Px
        {
            public bool IsFlat = false;
            public bool[] IsUsefull;
            public float[] Lamda;
            public float OneLamda = 0;
            public int IsUsefullCount = 0;

            public Px(int Length)
            {
                IsUsefull = new bool[Length];
                Lamda = new float[Length];
            }
        }
        public LamdaImgFilter(Bitmap bitmap, LamdaFilterOption option)
        {
            LamdaSet = new(option.Length, option.StartLamda, option.EndLamda);

            this.Img = new Px[bitmap.Width, bitmap.Height];
            this.Width = bitmap.Width;
            this.Height = bitmap.Height;
            this.gamma = option.gamma;
            this.Gap = option.Gap;
            this.Usefull = option.Usefull;
            this.LuR = option.R;
            this.LuG = option.G;
            this.LuB = option.B;

            this.RLamda = option.RLamda;
            this.RLamdaStd = option.RLamdaStd;
            this.GLamda = option.GLamda;
            this.GLamdaStd = option.GLamdaStd;
            this.BLamda = option.BLamda;
            this.BLamdaStd = option.BLamdaStd;

            this.OneTen = option.OneTen;
            this.GetGrayTone = option.GetGrayTone;
            this.UsefullDown = option.UsefullDown;

            float[] floats = GetReferencePxWhite();
            this.PxWhiteAvg = (float)((decimal)floats.Average() * 1.0000744317892037537379201598529M);

            for (int i = 0; i < floats.Length; i++)
            {
                floats[i] /= PxWhiteAvg;
            }
            this.PxWhiteMax = floats.Max();
            this.PxWhiteAU = GetReferencePxWhiteAU(floats,option.UsingAll);

            for (int x = 0; x < Width; x++)
            {
                for (int y = 0; y < Height; y++)
                {
                    Img[x, y] = GetPx(bitmap.GetPixel(x, y),option.UsingFlat,option.UsingAll);
                }
            }
        }
        public float[] GetReferencePxWhite()
        {
            Color color = Color.FromArgb(255, 255, 255, 255);

            color = INsRGBgotoLinearColor(color);

            float[] bytes = new float[this.LamdaSet.Length];

            for (int i = 0; i < bytes.Length; i++)
            {
                bytes[i] = RGB(LamdaSet.Lamda[i], color);
            }

            return bytes;
        }
        public float GetReferencePxWhiteAU(float[] floats, bool BeforePerAvg = true)
        {
            if (!BeforePerAvg)
            {
                for (int i = 0; i < floats.Length; i++)
                {
                    floats[i] /= PxWhiteAvg;
                }
            }

            for (int i = 0; i < floats.Length; i++)
            {
                floats[i] /= PxWhiteMax;
                floats[i] *= 100;
            }

            return (float)(100M * 1.055467109345181834580297936324M / (decimal)floats.Average());
        }

        public Px GetPx(Color Onecolor, bool IsFlat = false, bool UsingAll = true)
        {
            Px px = new(this.LamdaSet.Length);
            Color color = Color.FromArgb(Onecolor.A, Onecolor.R, Onecolor.G, Onecolor.B);

            color = INsRGBgotoLinearColor(color,gamma);

            float[] floats = new float[this.LamdaSet.Length];

            for (int i = 0; i < floats.Length; i++)
            {
                floats[i] = RGB(LamdaSet.Lamda[i], color);
            }

            if (floats.Average() > 0)
            {
                for (int i = 0; i < floats.Length; i++)
                {
                    floats[i] = (floats[i] * 100f) / (PxWhiteAvg * PxWhiteMax);
                }
            }

            if (IsFlat)
            {
                if ((GetSTD(floats) < 0.3f && floats.Average() > 80f) || (Math.Abs(color.R - color.G) < Gap && Math.Abs(color.G - color.B) < Gap && Math.Abs(color.B - color.R) < Gap)) // || 디버깅 또는 압축 용으로는 (bool1) || (bool2)로 사용함
                {
                    for (int i = 0; i < floats.Length; i++)
                    {
                        if (floats[i] > 102f)
                            floats[i] = 102f;
                    }

                    px.IsFlat = true;
                    px.OneLamda = floats.Average() * PxWhiteAU;
                }

                floats = null;
                return px;
            }
            else
            {
                for (int i = 0; i < floats.Length; i++)
                {
                    if (floats[i] > 102f)
                        floats[i] = 102f;

                    px.IsUsefull[i] = IsUsefull(floats, i);
                    if (px.IsUsefull[i])
                    {
                        px.IsUsefullCount++;
                        px.Lamda[i] = floats[i] * PxWhiteAU;
                    }
                }
                floats = null;
                return px;
            }
        }

        public Bitmap GetBitmap(bool ZeroControl = false)
        {
            Bitmap bitmap = new(Width, Height);

            Color White = Color.FromArgb(ClampInt32(255 + LuR), ClampInt32(255 + LuG), ClampInt32(255 + LuB));

            for (int i = 0; i < Width; i++)
            {
                for (int ii = 0; ii < Height; ii++)
                {
                    Color color = Phi(Img[i, ii], ZeroControl);

                    color = Color.FromArgb(Clamp(color.R + LuR), Clamp(color.G + LuG), Clamp(color.B + LuB));

                    color = WhiteBalance(color, White);

                    bitmap.SetPixel(i, ii, color);
                }
            }

            if (GetGrayTone)
                bitmap = GetGrayToneImg(bitmap);

            return bitmap;
        }

        private Color Phi(Px px, bool Zero)
        {
            float R = 0, G = 0, B = 0;

            if(OneTen)
            {
                for (int i = 0; i < LamdaSet.Length; i++)
                {
                    if (px.IsFlat)
                    {
                        float value;
                        value = px.OneLamda;

                        value = value / LamdaSet.Length;

                        R = value;
                        G = value;
                        B = value;
                        break;
                    }
                    else if (px.IsUsefull[i])
                    {
                        float value;
                        value = px.Lamda[i];

                        value = value / (LamdaSet.Length * px.IsUsefullCount / 3);

                        R += value * (float)this.R(LamdaSet.Lamda[i]);
                        G += value * (float)this.G(LamdaSet.Lamda[i]);
                        B += value * (float)this.B(LamdaSet.Lamda[i]);
                    }
                }
            }
            else
            {
                for (int i = 0; i < LamdaSet.Length; i++)
                {
                    if (px.IsFlat)
                    {
                        float value;
                        value = px.OneLamda;

                        value = value / 100;

                        R = value;
                        G = value;
                        B = value;
                        break;
                    }
                    else if (px.IsUsefull[i])
                    {
                        float value;
                        value = px.Lamda[i];

                        value = value / (100 * px.IsUsefullCount / 3);

                        R += value * (float)this.R(LamdaSet.Lamda[i]);
                        G += value * (float)this.G(LamdaSet.Lamda[i]);
                        B += value * (float)this.B(LamdaSet.Lamda[i]);
                    }
                }
            }

                float total = R + G + B;
            if (total < 1e-4)
            {
                return Color.FromArgb(0, 0, 0);
            }

            float Max = Math.Max(R, Math.Max(G, B));
            if (Max > 1f)
            {
                R /= Max;
                G /= Max;
                B /= Max;
            }

            float scale;
            if (Max <= 1f && Zero)
                scale = 255f;

            if (px.IsFlat)
                scale = 1.0f;
            else
                scale = 0.95f;

            R *= scale;
            G *= scale;
            B *= scale;

            if (float.IsNaN(R) || float.IsInfinity(R))
                R = 0;
            if (float.IsNaN(G) || float.IsInfinity(G))
                G = 0;
            if (float.IsNaN(B) || float.IsInfinity(B))
                B = 0;

            Color color = InLinearRGBgotoSRGBColor(R, G, B, 255,gamma);

            Color DefWhite = Color.FromArgb(254, 254, 254);

            color = WhiteBalance(color, DefWhite);

            return color;
        }
        //

        float sRGBtoLinear(float x, float gamma = 2.8f) //sRGB -> LinearRGB gamma = 2.4f defineZero || gamma = 3.0f defineOne
        {
            if (x <= 0.04045f)
                return x / 12.92f;
            else
                return (float)Math.Pow((x + 0.055f) / 1.055f, gamma);
        }

        float LinearTosRGB(float x, float gamma = 2.8f) //LinearRGB -> sRGB
        {
            if (x <= 0.0031308f)
                return 12.92f * x;
            else
                return 1.055f * (float)Math.Pow(x, 1.0 / gamma) - 0.055f;
        }

        Color INsRGBgotoLinearColor(Color color, float gamma = 2.8f) //sRGB -> LinearRGB
        {
            float R = sRGBtoLinear(color.R / 255f);
            float G = sRGBtoLinear(color.G / 255f);
            float B = sRGBtoLinear(color.B / 255f);

            return Color.FromArgb(color.A, (int)(R * 255f), (int)(G * 255f), (int)(B * 255f));
        }

        Color InLinearRGBgotoSRGBColor(float R, float G, float B, int A = 255, float gamma = 2.8f)
        {
            R = LinearTosRGB(R, gamma);
            G = LinearTosRGB(G, gamma);
            B = LinearTosRGB(B, gamma);

            return Color.FromArgb(A, Math.Clamp((int)(R * 255f), 0, 255), Math.Clamp((int)(G * 255f), 0, 255), Math.Clamp((int)(B * 255f), 0, 255));
        }

        private static int Clamp(int value)
        {
            return Math.Max(0, Math.Min(255, value));
        }

        private static Int32 ClampInt32(Int32 value)
        {
            return Int32.Max(0, Int32.Min(255, value));
        }

        public static Color WhiteBalance(Color input, Color whitePoint)
        {
            // 화이트 포인트 RGB를 0~1 범위로 정규화
            double wR = whitePoint.R / 255.0;
            double wG = whitePoint.G / 255.0;
            double wB = whitePoint.B / 255.0;

            // 입력 RGB를 정규화 후 화이트 포인트로 나눔
            double r = input.R / 255.0 / wR;
            double g = input.G / 255.0 / wG;
            double b = input.B / 255.0 / wB;

            // 다시 0~255로 변환 + 클램핑
            int newR = Clamp((int)(r * 255));
            int newG = Clamp((int)(g * 255));
            int newB = Clamp((int)(b * 255));

            return Color.FromArgb(input.A, newR, newG, newB);
        }

        private float GetSTD(float[] floats)
        {
            float Arg = (floats).Average();
            float sdSum = floats.Select(val => (val - Arg) * (val - Arg)).Sum();
            return (float)Math.Sqrt(sdSum / (floats.Length - 1));
        }

        private float SmoothBell(float Lamda, float Center, float Width)
        {
            float t = (Lamda - Center) / Width;
            return MathF.Exp(-t * t);
        }

        private float R(float Lamda)
        {
            float R = SmoothBell(Lamda, RLamda, RLamdaStd); //700, 30 * 2.35f
            if (float.IsPositiveInfinity(R))
            {
                return float.MaxValue;
            }
            else if (float.IsNegativeInfinity(R))
            {
                return float.MinValue;
            }
            else if (float.IsNaN(R))
            {
                return 0;
            }
            else
            {
                return R;
            }
        }
        private float G(float Lamda)
        {
            float G = SmoothBell(Lamda, GLamda, GLamdaStd);//560, 25 * 2.5f
            if (float.IsPositiveInfinity(G))
            {
                return float.MaxValue;
            }
            else if (float.IsNegativeInfinity(G))
            {
                return float.MinValue;
            }
            else if (float.IsNaN(G))
            {
                return 0;
            }
            else
            {
                return G;
            }
        }
        private float B(float Lamda)
        {
            float B = SmoothBell(Lamda, BLamda, BLamdaStd);//420, 25 * 2.6f
            if (float.IsPositiveInfinity(B))
            {
                return float.MaxValue;
            }
            else if (float.IsNegativeInfinity(B))
            {
                return float.MinValue;
            }
            else if (float.IsNaN(B))
            {
                return 0;
            }
            else
            {
                return B;
            }
        }

        private float RGB(float Lamda, Color color)
        {
            float RGB = color.R * this.R(Lamda) + color.G * this.G(Lamda) + color.B * B(Lamda);

            if (float.IsPositiveInfinity(RGB))
            {
                return float.MaxValue;
            }
            else if (float.IsNegativeInfinity(RGB))
            {
                return float.MinValue;
            }
            else if (float.IsNaN(RGB))
            {
                return 0;
            }
            else
            {
                return RGB;
            }
        }

        private bool IsUsefull(float[] Lamda, int i)
        {
            if (Lamda[i] > (float)1.02f * Usefull && Lamda[i] < (float)1.02f * UsefullDown)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        //

        public Bitmap GetGrayToneImg(Bitmap bitmap)
        {
            Bitmap Out = new(bitmap.Width, bitmap.Height);
            for(int x = 0; x < bitmap.Width; x++)
            {
                for(int y = 0; y < bitmap.Height; y++)
                {
                    Color color = bitmap.GetPixel(x,y);
                    float[] colors = new float[3];
                    colors[0] = color.R;
                    colors[1] = color.G;
                    colors[2] = color.B;

                    byte NewColorOne = (byte)colors.Average();
                    Color NewColor = Color.FromArgb(255, NewColorOne, NewColorOne, NewColorOne);


                    Out.SetPixel(x, y, NewColor);
                }
            }
            return Out;
        }
    }
}
