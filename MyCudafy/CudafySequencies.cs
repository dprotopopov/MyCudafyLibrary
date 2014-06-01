using System;
using System.Diagnostics;
using System.Linq;
using MyLibrary.Collections;
using Cudafy;
using Cudafy.Translator;
using Cudafy.Host;

namespace MyCudafy
{
    /// <summary>
    ///     Класс работы с двумя множествами последовательностей
    ///     Данный класс реализует модель специализированного вычислительного устройства
    ///     с фиксированным набором элементарных операций и использует параллельные вычисления CUDA
    ///     для реализации этой модели
    /// </summary>
    public static class CudafySequencies
    {
        /// <summary>
        ///     Семафор для блокирования одновременного доступа к данному статичному классу
        ///     из разных параллельных процессов
        ///     Надеюсь CUDAfy тоже заботится о блокировании одновременного доступа к видеокарточке ;)
        /// </summary>
        public static readonly object Semaphore = new Object();

        #region Регистры класса

        [Cudafy] private static int[] _indexes1;
        [Cudafy] private static int[] _indexes2;
        [Cudafy] private static int[] _sequencies1;
        [Cudafy] private static int[] _sequencies2;
        [Cudafy] private static int[,] _matrix;

        #endregion

        public static void SetSequencies(int[][] value1, int[][] value2)
        {
            int counts1 = value1.Length;
            int counts2 = value2.Length;
            var list1 = new StackListQueue<int> {0};
            foreach (var value in value1) list1.Add(list1.Last() + value.Length);
            _indexes1 = list1.ToArray();
            var list2 = new StackListQueue<int> {0};
            foreach (var value in value2) list2.Add(list2.Last() + value.Length);
            _indexes2 = list2.ToArray();
            _sequencies1 = value1.SelectMany(seq => seq).ToArray();
            _sequencies2 = value2.SelectMany(seq => seq).ToArray();
            _matrix = new int[counts1, counts2];
        }

        public static int[,] GetMatrix()
        {
            return _matrix;
        }

        /// <summary>
        ///     Вызов и исполнение одной элементарной функции по имени функции
        /// </summary>
        /// <param name="function"></param>
        public static void Execute(string function)
        {
            Debug.Assert(_indexes1.Last() == _sequencies1.Length);
            Debug.Assert(_indexes2.Last() == _sequencies2.Length);

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            // copy the arrays 'a' and 'b' to the GPU
            int[] devIndexes1 = gpu.CopyToDevice(_indexes1);
            int[] devIndexes2 = gpu.CopyToDevice(_indexes2);
            int[] devSequencies1 = gpu.CopyToDevice(_sequencies1);
            int[] devSequencies2 = gpu.CopyToDevice(_sequencies2);
            int[,] devMatrix = gpu.Allocate(_matrix);

            int rows = _matrix.GetLength(0);
            int columns = _matrix.GetLength(1);

            dim3 gridSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));
            dim3 blockSize = Math.Min(15, (int) Math.Pow((double) rows*columns, 0.33333333333));

            gpu.Launch(gridSize, blockSize, function,
                devSequencies1, devIndexes1,
                devSequencies2, devIndexes2,
                devMatrix);

            // copy the array 'c' back from the GPU to the CPU
            gpu.CopyFromDevice(devMatrix, _matrix);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Вычисление значения операции сравнения двух последовательностей
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="sequencies1"></param>
        /// <param name="indexes1"></param>
        /// <param name="sequencies2"></param>
        /// <param name="indexes2"></param>
        /// <param name="matrix"></param>
        [Cudafy]
        public static void Compare(GThread thread,
            int[] sequencies1, int[] indexes1,
            int[] sequencies2, int[] indexes2,
            int[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows*columns;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid/columns;
                int column = tid%columns;
                matrix[row, column] = (indexes1[row + 1] - indexes1[row]) - (indexes2[column + 1] - indexes2[column]);
                for (int i = indexes1[row], j = indexes2[column];
                    i < indexes1[row + 1] && j < indexes2[column + 1] && matrix[row, column] == 0;
                    i++,j++)
                    matrix[row, column] = (sequencies1[i] - sequencies2[j]);
            }
        }

        /// <summary>
        ///     Подсчёт количества пар совпадающих элементов из двух последовательностей
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="sequencies1"></param>
        /// <param name="indexes1"></param>
        /// <param name="sequencies2"></param>
        /// <param name="indexes2"></param>
        /// <param name="matrix"></param>
        [Cudafy]
        public static void CountIntersections(GThread thread,
            int[] sequencies1, int[] indexes1,
            int[] sequencies2, int[] indexes2,
            int[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows*columns;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid/columns;
                int column = tid%columns;
                matrix[row, column] = 0;
                for (int i = indexes1[row]; i < indexes1[row + 1]; i++)
                    for (int j = indexes2[column]; j < indexes2[column + 1]; j++)
                        if (sequencies1[i] == sequencies2[j]) matrix[row, column]++;
            }
        }

        /// <summary>
        ///     Проверка, что начальная и конечная точка первой последовательности
        ///     находятся в множестве точек второй последовательности
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="sequencies1"></param>
        /// <param name="indexes1"></param>
        /// <param name="sequencies2"></param>
        /// <param name="indexes2"></param>
        /// <param name="matrix"></param>
        [Cudafy]
        public static void IsFromTo(GThread thread,
            int[] sequencies1, int[] indexes1,
            int[] sequencies2, int[] indexes2,
            int[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows*columns;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid/columns;
                int column = tid%columns;
                matrix[row, column] = 0;
                int b = 0;
                for (int i = indexes1[row]; i < indexes1[row] + 1 && b == 0; i++)
                    for (int j = indexes2[column]; j < indexes2[column + 1] && b == 0; j++)
                        if (sequencies1[i] == sequencies2[j])
                            b = 1;
                for (int i = indexes1[row + 1] - 1; i < indexes1[row + 1] && b != 0 && matrix[row, column] == 0; i++)
                    for (int j = indexes2[column]; j < indexes2[column + 1] && b != 0 && matrix[row, column] == 0; j++)
                        matrix[row, column] = (sequencies1[i] == sequencies2[j]) ? 1 : 0;
            }
        }
    }
}