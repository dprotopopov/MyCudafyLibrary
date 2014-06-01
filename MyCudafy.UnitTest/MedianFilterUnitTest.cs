using System;
using System.Drawing;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MyCudafy.UnitTest
{
    [TestClass]
    public class MedianFilterUnitTest
    {
        private readonly Random _random = new Random();

        [TestMethod]
        public void TestMedianFilter()
        {
            int parity = 1;
            int itemsCount = 10;
            int ceilingItemSize = 5;
            int i = 2;
            int itemSize = 25;
            for (
                int tid = 0;
                tid < itemsCount*((1 << (ceilingItemSize - i - 1)) + parity);
                tid += 1)
            {
                int itemId = tid/((1 << (ceilingItemSize - i - 1)) + parity);
                int pairId = tid%((1 << (ceilingItemSize - i - 1)) + parity);
                int offset = itemId*itemSize + (pairId << (i + 1)) - (parity << i);
                int index0 =
                    Math.Min(Math.Max(itemId*itemSize, offset + (0 << i)),
                        (itemId + 1)*itemSize);
                int index1 =
                    Math.Min(Math.Max(itemId*itemSize, offset + (1 << i)),
                        (itemId + 1)*itemSize);
                int index2 =
                    Math.Min(Math.Max(itemId*itemSize, offset + (2 << i)),
                        (itemId + 1)*itemSize);
                Console.WriteLine(itemId + "->" + index0 + "," + index1 + "," + index2 + ",");
            }
            Console.WriteLine();
            var oldbmp = new Bitmap(33, 33);
            for (int y = 0; y < oldbmp.Height; y++)
                for (int x = 0; x < oldbmp.Width; x++)
                {
                    var bytes = new byte[3];
                    _random.NextBytes(bytes);
                    oldbmp.SetPixel(x, y, Color.FromArgb(bytes[0], bytes[1], bytes[2]));
                    oldbmp.SetPixel(x, y, Color.FromArgb(x, y, x ^ y));
                }
            for (int y = 0; y < oldbmp.Height; y++)
            {
                for (int x = 0; x < oldbmp.Width; x++)
                {
                    Color c = oldbmp.GetPixel(x, y);
                    Console.Write("(" + c.R + "," + c.G + "," + c.B + "),");
                }
                Console.WriteLine();
            }
            Console.WriteLine();

            Bitmap newbmp;
            lock (CudafyFilter.Semaphore)
            {
                CudafyFilter.SetBitmap(oldbmp, 5, 1 << 12);
                CudafyFilter.MedianFilter();
                newbmp = CudafyFilter.GetBitmap();
            }
            for (int y = 0; y < oldbmp.Height; y++)
            {
                for (int x = 0; x < oldbmp.Width; x++)
                {
                    Color c = newbmp.GetPixel(x, y);
                    Console.Write("(" + c.R + "," + c.G + "," + c.B + "),");
                }
                Console.WriteLine();
            }
            int Nh = 2;
            int N = 2*Nh + 1;
            int n = N*N/2;
            var R0 = new Byte[newbmp.Height, newbmp.Width];
            var G0 = new Byte[newbmp.Height, newbmp.Width];
            var B0 = new Byte[newbmp.Height, newbmp.Width];
            for (int y = 0; y < oldbmp.Height; y++)
                for (int x = 0; x < oldbmp.Width; x++)
                {
                    Color c = oldbmp.GetPixel(x, y);
                    R0[y, x] = c.R;
                    G0[y, x] = c.G;
                    B0[y, x] = c.B;
                }

            for (int y1 = Nh; y1 < newbmp.Height - Nh; y1++)
            {
                for (int x1 = Nh; x1 < newbmp.Width - Nh; x1++)
                {
                    i = 0;
                    Byte[] r = new Byte[N*N], g = new Byte[N*N], b = new Byte[N*N];
                    for (int y2 = -Nh; y2 <= Nh; y2++)
                    {
                        int y3 = y1 + y2;
                        for (int x2 = -Nh; x2 <= Nh; x2++)
                        {
                            int x3 = x1 + x2;
                            r[i] = R0[y3, x3];
                            g[i] = G0[y3, x3];
                            b[i] = B0[y3, x3];
                            i++;
                        }
                    }
                    Array.Sort(r);
                    Array.Sort(g);
                    Array.Sort(b);
                    Color c = newbmp.GetPixel(x1, y1);
                    Assert.AreEqual(r[n], c.R);
                    Assert.AreEqual(g[n], c.G);
                    Assert.AreEqual(b[n], c.B);
                }
            }
        }
    }
}