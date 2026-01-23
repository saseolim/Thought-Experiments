namespace ThreadStudy
{
    internal class ThreadStudyProgram
    {
        static void Main(string[] args)
        {
            Random mode = new Random();
            List<Thread> threads = new List<Thread>();

            ThreadStudy[] TF = new ThreadStudy[100];
            for (int i = 0; i < 100; i++)
            {
                int this_mode = mode.Next(-1, 3);
                TF[i] = new ThreadStudy(new ThreadHeader(this_mode, i));
                threads.Add(new Thread(new ThreadStart(TF[i].Run)));
                threads[threads.Count - 1].Start();
            }

            List<int> zero = new List<int>();
            List<int> one = new List<int>();
            List<int> two = new List<int>();
            List<int> def = new List<int>();

            for (int i = 0; i < 100; i++)
            {
                switch (TF[i].header.mode)
                {
                    case 0:
                        zero.Add(i);
                        break;
                    case 1:
                        one.Add(i);
                        break;
                    case 2:
                        two.Add(i);
                        break;
                    default:
                        def.Add(i);
                        break;
                }
            }
            int count = 0;
            Console.WriteLine(zero.Count + one.Count + two.Count + def.Count);

            for (int i = 0; i < zero.Count; i++)
            {
                TF[zero[i]].IsRun = true;
                threads[zero[i]].Join();
                count++;
            }

            for (int i = 0; i < one.Count; i++)
            {
                TF[one[i]].IsRun = true;
                threads[one[i]].Join();
                count++;
            }

            for (int i = 0; i < two.Count; i++)
            {
                TF[two[i]].IsRun = true;
                threads[two[i]].Join();
                count++;
            }

            for (int i = 0; i < def.Count; i++)
            {
                TF[def[i]].IsRun = true;
                threads[def[i]].Join();
                count++;
            }

            Console.WriteLine("Program End : " + count);

        }
    }
}
