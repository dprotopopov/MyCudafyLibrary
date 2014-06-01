using System;
using System.Globalization;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MyCudafy.UnitTest
{
    [TestClass]
    public class ArrayUnitTest
    {
        private readonly Random _random = new Random();

        [TestMethod]
        public void TestMergeSort()
        {
            for (int t = 0; t < 10; t++)
            {
                var arr = new int[_random.Next(100, 200)];
                for (int i = 0; i < arr.Length; i++) arr[i] = _random.Next();
                var arr1 = new int[arr.Length];
                var arr2 = new int[arr.Length];
                for (int i = 0; i < arr.Length; i++) arr1[i] = arr[i];
                for (int i = 0; i < arr.Length; i++) arr2[i] = arr[i];
                Array.Sort(arr1);
                lock (CudafyArray.Semaphore)
                {
                    CudafyArray.SetArray(arr2);
                    CudafyArray.MergeSort();
                    arr2 = CudafyArray.GetArray();
                }
                Console.WriteLine(string.Join(",", arr1.Select(i => i.ToString(CultureInfo.InvariantCulture))));
                Console.WriteLine(string.Join(",", arr2.Select(i => i.ToString(CultureInfo.InvariantCulture))));
                Assert.IsTrue(arr1.SequenceEqual(arr2));
            }
        }

        [TestMethod]
        public void TestBitonicSort()
        {
            for (int t = 0; t < 10; t++)
            {
                var arr = new int[_random.Next(100, 200)];
                for (int i = 0; i < arr.Length; i++) arr[i] = _random.Next();
                var arr1 = new int[arr.Length];
                var arr2 = new int[arr.Length];
                for (int i = 0; i < arr.Length; i++) arr1[i] = arr[i];
                for (int i = 0; i < arr.Length; i++) arr2[i] = arr[i];
                Array.Sort(arr1);
                lock (CudafyArray.Semaphore)
                {
                    CudafyArray.SetArray(arr2);
                    CudafyArray.BitonicSort();
                    arr2 = CudafyArray.GetArray();
                }
                Console.WriteLine(string.Join(",", arr1.Select(i => i.ToString(CultureInfo.InvariantCulture))));
                Console.WriteLine(string.Join(",", arr2.Select(i => i.ToString(CultureInfo.InvariantCulture))));
                Assert.IsTrue(arr1.SequenceEqual(arr2));
            }
        }

        [TestMethod]
        public void TestOddEvenSort()
        {
            for (int t = 0; t < 10; t++)
            {
                var arr = new int[_random.Next(100, 200)];
                for (int i = 0; i < arr.Length; i++) arr[i] = _random.Next();
                var arr1 = new int[arr.Length];
                var arr2 = new int[arr.Length];
                for (int i = 0; i < arr.Length; i++) arr1[i] = arr[i];
                for (int i = 0; i < arr.Length; i++) arr2[i] = arr[i];
                Array.Sort(arr1);
                lock (CudafyArray.Semaphore)
                {
                    CudafyArray.SetArray(arr2);
                    CudafyArray.OddEvenSort();
                    arr2 = CudafyArray.GetArray();
                }
                Console.WriteLine(string.Join(",", arr1.Select(i => i.ToString(CultureInfo.InvariantCulture))));
                Console.WriteLine(string.Join(",", arr2.Select(i => i.ToString(CultureInfo.InvariantCulture))));
                Assert.IsTrue(arr1.SequenceEqual(arr2));
            }
        }
    }
}