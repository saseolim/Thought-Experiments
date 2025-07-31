using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;

namespace IMG
{
    public partial class SubIMGsControl
    {
        public SubIMGsControl(string FolderPath, string OutFolderName,LamdaFilterOption option, int BlockSize, int SubMaxThread, int OneMaxThread, bool GPU = false)
        {
            DirectoryInfo info = new DirectoryInfo(FolderPath);

            List<string[]> IMGs = new();

            foreach (FileInfo file in info.GetFiles())
            {
                if (Path.GetExtension(file.FullName).ToLower() == ".jpeg" ||
                    Path.GetExtension(file.FullName).ToLower() == ".jpg" ||
                    Path.GetExtension(file.FullName).ToLower() == ".png")
                {
                    IMGs.Add(new string[2]);
                    IMGs[IMGs.Count - 1][0] = file.FullName;
                    IMGs[IMGs.Count - 1][1] = Path.GetFileNameWithoutExtension(file.FullName);
                 }
            }

            info = new DirectoryInfo(OutFolderName);

            if (info.Exists == false)
                info.Create();

            List<Thread> threads = new List<Thread>();

            foreach (string[] PathAndName in IMGs)
            {
                threads.Add(new Thread(() =>
                {
                    Bitmap bitmap = new(PathAndName[0]);

                    LamdaImgFilterBlockControl LFB = new(bitmap, option, BlockSize,OneMaxThread,GPU);

                    Bitmap Tmp = LFB.GetFiltering();

                    Tmp.Save($"{OutFolderName}\\{PathAndName[1]}.png", ImageFormat.Png);
                }
                    ));
                threads[threads.Count - 1].Start();

                //if(threads.Count % SubMaxThread == 0)
                //if (threads.Count(t => t.IsAlive) >= SubMaxThread)
                //{
                //    foreach (Thread thread in threads)
                //        thread.Join();
                //}

                while (threads.Count(t => t.IsAlive) >= SubMaxThread)
                {
                    Thread.Sleep(10);
                }
            }

            foreach (Thread thread in threads)
            {
                thread.Join();
            }

        }
    }
}
