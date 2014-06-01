using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MyCudafy.Collections
{
    public class StackListQueue<T> : MyLibrary.Collections.StackListQueue<T>
    {
        public new bool Contains(IEnumerable<T> collection)
        {
            try
            {
                int[,] matrix;
                int first;
                lock (CudafySequencies.Semaphore)
                {
                    CudafySequencies.SetSequencies(
                        collection.Select(GetInts).Select(item => item.ToArray()).ToArray(),
                        this.Select(GetInts).Select(item => item.ToArray()).ToArray()
                        );
                    CudafySequencies.Execute("Compare");
                    matrix = CudafySequencies.GetMatrix();
                }
                lock (CudafyMatrix.Semaphore)
                {
                    CudafyMatrix.SetMatrix(matrix);
                    CudafyMatrix.ExecuteRepeatZeroIndexOfZeroFirstIndexOfNonPositive();
                    first = CudafyMatrix.GetFirst();
                }
                return first < 0;
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
                return collection.All(Contains);
            }
        }

        public new int IndexOf(T item)
        {
            return base.IndexOf(item);
        }

        public IEnumerable<T> Distinct()
        {
            if (Count == 0) return new StackListQueue<T>();
            IEnumerable<IEnumerable<int>> list = this.Select(GetInts);
            int[,] matrix;
            int[] indexes;
            lock (CudafySequencies.Semaphore)
            {
                int[][] arr = this.Select(GetInts).Select(item => item.ToArray()).ToArray();
                CudafySequencies.SetSequencies(arr, arr);
                CudafySequencies.Execute("Compare");
                matrix = CudafySequencies.GetMatrix();
            }
            lock (CudafyMatrix.Semaphore)
            {
                CudafyMatrix.SetMatrix(matrix);
                CudafyMatrix.ExecuteRepeatZeroIndexOfZero();
                indexes = CudafyMatrix.GetIndexes();
            }
            return indexes.Where((value, index) => value == index)
                .Select(index => this[index]);
        }

        public override bool Equals(object obj)
        {
            var collection = obj as MyLibrary.Collections.SortedStackListQueue<T>;
            return collection != null && this.SequenceEqual(collection);
        }

        public new virtual StackListQueue<int> GetInts(T values)
        {
            throw new NotImplementedException();
        }

        public override int GetHashCode()
        {
            try
            {
                lock (CudafyArray.Semaphore)
                {
                    CudafyArray.SetArray(this.Select(item => item.GetHashCode()).ToArray());
                    CudafyArray.Execute("Hash");
                    return CudafyArray.GetHash();
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
                return this.Aggregate(0,
                    (current, item) => (current << 1) ^ (current >> (8*sizeof (int) - 1)) ^ item.GetHashCode());
            }
        }

        public override string ToString()
        {
            return string.Join(",", this.Select(item => item.ToString()));
        }

        #region

        public StackListQueue(IEnumerable<T> value) : base(value)
        {
        }

        public StackListQueue(T value) : base(value)
        {
        }

        public StackListQueue()
        {
        }

        #endregion
    }
}