using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using MyLibrary.Collections;
using MyLibrary.Trace;
using Double = MyLibrary.Types.Double;
using Int32 = MyLibrary.Types.Int32;

namespace MyCudafy
{
    public struct CudafyMulti
    {
        /// <summary>
        ///     Семафор для блокирования одновременного доступа к данному статичному классу
        ///     из разных параллельных процессов
        ///     Надеюсь CUDAfy тоже заботится о блокировании одновременного доступа к видеокарточке ;)
        /// </summary>
        public static readonly object Semaphore = new Object();

        private static int _gridSize1;
        private static int _blockSize1;
        private static int _gridSize2;
        private static int _blockSize2;
        private static int _gridSize3;
        private static int _blockSize3;

        #region Регистры класса

        [Cudafy] private static double[] _a;
        [Cudafy] private static double[] _b;
        [Cudafy] private static int[] _sizes; // Размеры куба (количество точек сетки) по соответствующим осям
        [Cudafy] private static double[] _lengths; // Шаг сетки по соответствующим осям
        [Cudafy] private static int[] _intV; // Координатная база внутренних точек
        [Cudafy] private static int[] _extV; // Координатная база точек
        [Cudafy] private static double[] _w; // Весовые коэффициенты слагаемых

        #endregion

        public static void SetArray(int[] sizes, double[] lengths, double[] array)
        {
            _sizes = sizes;
            _lengths = lengths;
            _a = array;
            _b = new double[1];

            _gridSize3 = Math.Min(15, (int) Math.Pow(array.Length, 0.333333333333));
            _blockSize3 = Math.Min(15, (int) Math.Pow(array.Length, 0.333333333333));
            _gridSize2 = Math.Min(15, (int) Math.Pow(array.Length, 0.22222222222));
            _blockSize2 = Math.Min(15, (int) Math.Pow(array.Length, 0.22222222222));
            _gridSize1 = Math.Min(15, (int) Math.Pow(array.Length, 0.1111111111111));
            _blockSize1 = Math.Min(15, (int) Math.Pow(array.Length, 0.1111111111111));
        }

        public static double[] GetArray()
        {
            return _a;
        }

        public static IEnumerable<double> ExecuteLaplaceSolver(double epsilon, double a, int gridSize = 0,
            int blockSize = 0,
            AppendLineCallback AppendLineCallback = null)
        {
            if (gridSize > 0)
            {
                _gridSize3 = Math.Min(255, Math.Max(1, gridSize));
                _gridSize2 = Math.Min(255, Math.Max(1, blockSize*2/3));
                _gridSize1 = Math.Min(255, Math.Max(1, blockSize*1/3));
            }
            if (blockSize > 0)
            {
                _blockSize3 = Math.Min(255, Math.Max(1, blockSize));
                _blockSize2 = Math.Min(255, Math.Max(1, blockSize*2/3));
                _blockSize1 = Math.Min(255, Math.Max(1, blockSize*1/3));
            }

            Debug.Assert(_sizes.Length == _lengths.Length);
            Debug.Assert(_sizes.Aggregate(Int32.Mul) > 0);
            Debug.Assert(_lengths.Aggregate(Double.Mul) > 0.0);

            if (AppendLineCallback != null) AppendLineCallback("Размер массива:");
            for (int i = 0; i < _sizes.Length; i++)
                if (AppendLineCallback != null)
                    AppendLineCallback(string.Format("Размер массива по оси № {0}:\t{1}", i, _sizes[i]));

            // Степень дифференциального оператора
            // Реализовано только для оператора Лапласа
            // Для больших степеней надо использовать соответствующие полиномы большей степени
            // Для дифференциального оператора степени 2 (оператора Лапласа) полином имеет степень 1
            // Для дифференциального оператора степени rank полином имеет степень rank-1
            const int rank = 2;

            _extV = new int[_sizes.Length + 1];
            _intV = new int[_sizes.Length + 1];
            _intV[0] = _extV[0] = 1;

            for (int i = 1; i <= _sizes.Length; i++)
            {
                _extV[i] = _extV[i - 1]*_sizes[i - 1];
                _intV[i] = _intV[i - 1]*(_sizes[i - 1] - rank);
            }

            // Расчёт коэффициентов слагаемых
            if (AppendLineCallback != null) AppendLineCallback("Расчёт коэффициентов слагаемых");
            _w = new double[_sizes.Length + 1];
            double sum2 = 0;
            for (int i = 0; i < _sizes.Length; i++) sum2 += (_sizes[i] - 1)*(_sizes[i] - 1)/(_lengths[i]*_lengths[i]);
            for (int i = 0; i < _sizes.Length; i++)
                _w[i] = (_sizes[i] - 1)*(_sizes[i] - 1)/(_lengths[i]*_lengths[i])/sum2/(1.0 + a);
            _w[_sizes.Length] = (a - 1.0)/(1.0 + a);

            if (AppendLineCallback != null) AppendLineCallback("Коэффициенты:");
            for (int i = 0; i < _sizes.Length; i++)
                if (AppendLineCallback != null)
                    AppendLineCallback(string.Format("Коэффициенты по оси № {0} (у двух точек):\t{1}", i, _w[i]));
            if (AppendLineCallback != null)
                AppendLineCallback(string.Format("Коэффициент у средней точки:\t{0}", _w[_sizes.Length]));

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            if (AppendLineCallback != null) AppendLineCallback("Выделяем видеоресурсы");
            double[][] devA =
            {
                gpu.Allocate<double>(_a.Length),
                gpu.Allocate<double>(_a.Length)
            };
            double[][] devB =
            {
                gpu.Allocate<double>(_a.Length),
                gpu.Allocate<double>((int) Math.Pow(_a.Length, 0.6666666666)),
                gpu.Allocate<double>((int) Math.Pow(_a.Length, 0.3333333333)),
                gpu.Allocate(_b)
            };
            int[] devSizes = gpu.Allocate(_sizes);
            int[][] devV =
            {
                gpu.Allocate(_extV),
                gpu.Allocate(_intV)
            };
            double[] devW = gpu.Allocate(_w);

            if (AppendLineCallback != null) AppendLineCallback("Копируем данные в видеопамять");
            gpu.CopyToDevice(_a, devA[0]);
            gpu.CopyToDevice(_sizes, devSizes);
            gpu.CopyToDevice(_extV, devV[0]);
            gpu.CopyToDevice(_intV, devV[1]);
            gpu.CopyToDevice(_w, devW);

            if (AppendLineCallback != null) AppendLineCallback("Дублируем массив в видеопамяти");
            gpu.Launch(_gridSize3, _blockSize3, "Copy", devA[0], devA[1]);
            var queue = new StackListQueue<double>();
            for (int step = 0;; step++)
            {
                //if (AppendLineCallback != null) AppendLineCallback(string.Format("Шаг итерации № {0}", step));

                // Вычисляем среднее взвешенное соседних точек
                //if (AppendLineCallback != null) AppendLineCallback("Вычисляем среднее взвешенное соседних точек");
                gpu.Launch(_gridSize3, _blockSize3, "LaplaceSolver", devA[step & 1], devA[1 - (step & 1)],
                    devSizes,
                    devV[0], devV[1],
                    devW);

                // Рассчитываем амплитуду изменений
                //if (AppendLineCallback != null) AppendLineCallback("Рассчитываем амплитуду изменений");
                gpu.Launch(_gridSize3, _blockSize3, "Delta", devA[step & 1], devA[1 - (step & 1)], devB[0]);

                // Трёх-ступенчатое  вычисление максимального значения в массиве
                //if (AppendLineCallback != null)
                //    AppendLineCallback("Трёх-ступенчатое  вычисление суммы значений в массиве");
                //if (AppendLineCallback != null)
                //    AppendLineCallback(string.Format("Массив размера {0} -> массив размера ...", _a.Length));
                gpu.Launch(_gridSize2, _blockSize2, "Sum", devB[0], devB[1]);
                //if (AppendLineCallback != null) AppendLineCallback("Массив размера ... -> массив размера ...");
                gpu.Launch(_gridSize1, _blockSize1, "Sum", devB[1], devB[2]);
                //if (AppendLineCallback != null) AppendLineCallback("Массив размера ... -> массив размера 1");
                gpu.Launch(1, 1, "Sum", devB[2], devB[3]);
                gpu.CopyFromDevice(devB[3], _b);

                double deltaSum = _b[0];

                gpu.Launch(_gridSize3, _blockSize3, "Square", devA[step & 1], devA[1 - (step & 1)], devB[0]);
                // Трёх-ступенчатое  вычисление максимального значения в массиве
                //if (AppendLineCallback != null)
                //    AppendLineCallback("Трёх-ступенчатое  вычисление суммы значений в массиве");
                //if (AppendLineCallback != null)
                //    AppendLineCallback(string.Format("Массив размера {0} -> массив размера ...", _a.Length));
                gpu.Launch(_gridSize2, _blockSize2, "Sum", devB[0], devB[1]);
                //if (AppendLineCallback != null) AppendLineCallback("Массив размера ... -> массив размера ...");
                gpu.Launch(_gridSize1, _blockSize1, "Sum", devB[1], devB[2]);
                //if (AppendLineCallback != null) AppendLineCallback("Массив размера ... -> массив размера 1");
                gpu.Launch(1, 1, "Sum", devB[2], devB[3]);
                gpu.CopyFromDevice(devB[3], _b);

                double squareSum = _b[0];

                //if (AppendLineCallback != null)
                //    AppendLineCallback(string.Format("Амплитуда изменений = {0}/{1}", deltaSum, squareSum));

                queue.Enqueue(deltaSum/squareSum);

                if (deltaSum > epsilon*squareSum) continue;

                if (AppendLineCallback != null)
                    AppendLineCallback(string.Format("Потребовалось {0} итераций", step + 1));

                // Если изменения меньше заданной величины, то возвращаем вычисленные значения
                if (AppendLineCallback != null)
                    AppendLineCallback("Копируем массив из видеопамяти в массив на компьютере");
                gpu.CopyFromDevice(devA[1 - (step & 1)], _a);
                break;
            }
            // free the memory allocated on the GPU
            if (AppendLineCallback != null) AppendLineCallback("Освобождаем видеоресурсы");
            gpu.FreeAll();
            return queue;
        }

        /// <summary>
        ///     Вычисление среднего арифметического соседних по осям точек
        ///     для внутренних точек куба
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="prev"></param>
        /// <param name="next"></param>
        /// <param name="sizes"></param>
        /// <param name="extV"></param>
        /// <param name="intV"></param>
        /// <param name="w"></param>
        [Cudafy]
        public static void LaplaceSolver(GThread thread,
            double[] prev, double[] next,
            int[] sizes,
            int[] extV, int[] intV,
            double[] w)
        {
            // Степень дифференциального оператора
            // Реализовано только для оператора Лапласа
            // Для больших степеней надо использовать соответствующие полиномы большей степени
            // Для дифференциального оператора степени 2 (оператора Лапласа) полином имеет степень 1
            // Для дифференциального оператора степени rank полином имеет степень rank-1
            const int rank = 2;

            // Перебор по индексам внутренних точек
            for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                tid < intV[sizes.Length];
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                // Преобразуем индекс внутренней точки в координаты
                // Преобразуем координаты в индекс точки
                int id = 0;
                for (int i = 0, v = tid; i < sizes.Length; i++)
                {
                    id += ((rank >> 1) + (v%(sizes[i] - rank)))*extV[i];
                    v = v/(sizes[i] - rank);
                }
                // Вычисляем среднее арифметическое соседних точек
                // для всех внутренних точек куба
                double s = prev[id]*w[sizes.Length];
                for (int i = 0; i < sizes.Length; i++)
                    s += (prev[id - extV[i]] + prev[id + extV[i]])*w[i];
                next[id] = s;
            }
        }

        /// <summary>
        ///     Копирование массива в видеопамяти
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="prev"></param>
        /// <param name="next"></param>
        [Cudafy]
        public static void Copy(GThread thread,
            double[] prev, double[] next)
        {
            for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                tid < prev.Length;
                tid += thread.blockDim.x*thread.gridDim.x)
                next[tid] = prev[tid];
        }

        [Cudafy]
        public static void Square(GThread thread,
            double[] prev, double[] next, double[] delta)
        {
            for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                tid < prev.Length;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                double x = next[tid];
                x = x*x;
                delta[tid] = x;
            }
        }

        [Cudafy]
        public static void Delta(GThread thread,
            double[] prev, double[] next, double[] delta)
        {
            for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                tid < prev.Length;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                double x = next[tid]*(prev[tid] - next[tid]);
                x = x*x;
                delta[tid] = x;
            }
        }

        /// <summary>
        ///     Шаг вычисления максимума в массиве
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="prev"></param>
        /// <param name="next"></param>
        [Cudafy]
        public static void Max(GThread thread,
            double[] prev, double[] next)
        {
            for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                tid < next.Length;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                next[tid] = 0;
                for (int i = 0; i*next.Length + tid < prev.Length; i++)
                {
                    int index = i*next.Length + tid;
                    if (prev[index] > next[tid]) next[tid] = prev[index];
                }
            }
        }

        /// <summary>
        ///     Шаг вычисления суммы элементов массива
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="prev"></param>
        /// <param name="next"></param>
        [Cudafy]
        public static void Sum(GThread thread,
            double[] prev, double[] next)
        {
            for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                tid < next.Length;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                next[tid] = 0;
                for (int i = 0; i*next.Length + tid < prev.Length; i++)
                {
                    int index = i*next.Length + tid;
                    next[tid] = next[tid] + prev[index];
                }
            }
        }
    }
}