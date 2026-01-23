using System;
using System.Collections.Generic;
using System.Text;

namespace ThreadStudy
{
    public struct ThreadHeader
    {
        public int mode { get; private set; }
        public int code { get; private set; }

        public ThreadHeader(int mode, int code)
        {
            this.mode = mode;
            this.code = code;
        }
    }
    public partial class ThreadStudy
    {
        public ThreadHeader header;
        public volatile bool IsRun = false;
        public volatile bool isEnd = false;
        public bool IsEnd { get => isEnd; private set => isEnd = value; }

        public ThreadStudy(ThreadHeader header)
        {
            this.header = header;
        }
        public void Run()
        {
            while (!IsRun)
            {
                Thread.Sleep(new Random().Next(1, 100));
            }

            int mode = header.mode;
            int code = header.code;
            Console.WriteLine("ThreadHeader Run Method : " + code);
            switch (mode)
            {
                case 0:
                    Console.WriteLine("Mode 0 selected : " + code);
                    break;
                case 1:
                    Console.WriteLine("Mode 1 selected : " + code);
                    break;
                case 2:
                    Console.WriteLine("Mode 2 selected : " + code);
                    break;
                default:
                    Console.WriteLine("Unknown mode : " + code);
                    break;
            }
            Console.WriteLine();
            while (!IsRun)
            {
                Thread.Sleep(new Random().Next(10, 100));
            }
            IsEnd = true;
        }
    }
}
