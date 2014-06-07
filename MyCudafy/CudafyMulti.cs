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
        [Cudafy] private static double[] _c;
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
            _c = new double[1];

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

        /// <param name="a">Параметр алгоритма для вычисления весовых коэффициентов</param>
        /// <param name="relax">
        ///     При использовании метода релаксации задействовано в два раза меньше памяти и вычисления производятся
        ///     на-месте. Для устанения коллизий с совместным доступом производится раскраска точек красное-чёрное для обработки
        ///     их по-очереди
        /// </param>
        /// <param name="epsilon">Точность вычислений</param>
        /// <param name="gridSize"></param>
        /// <param name="blockSize"></param>
        /// <param name="trace"></param>
        public static IEnumerable<double> ExecuteLaplaceSolver(double epsilon, double a, bool relax, int gridSize = 0,
            int blockSize = 0,
            ITrace trace = null)
        {
            AppendLineCallback appendLineCallback = trace != null ? trace.AppendLineCallback : null;
            ProgressCallback progressCallback = trace != null ? trace.ProgressCallback : null;
            CompliteCallback compliteCallback = trace != null ? trace.CompliteCallback : null;

            if (gridSize > 0)
            {
                _gridSize3 = gridSize;
                _gridSize2 = blockSize;
                _gridSize1 = blockSize;
            }
            if (blockSize > 0)
            {
                _blockSize3 = blockSize;
                _blockSize2 = blockSize;
                _blockSize1 = blockSize;
            }

            Debug.Assert(_sizes.Length == _lengths.Length);
            Debug.Assert(_sizes.Aggregate(Int32.Mul) > 0);
            Debug.Assert(_lengths.Aggregate(Double.Mul) > 0.0);

            if (appendLineCallback != null) appendLineCallback("Размер массива:");
            for (int i = 0; i < _sizes.Length; i++)
                if (appendLineCallback != null)
                    appendLineCallback(string.Format("Размер массива по оси № {0}:\t{1}", i, _sizes[i]));

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
            if (appendLineCallback != null) appendLineCallback("Расчёт коэффициентов слагаемых");
            _w = new double[_sizes.Length + 1];
            double sum2 = 0;
            for (int i = 0; i < _sizes.Length; i++) sum2 += (_sizes[i] - 1)*(_sizes[i] - 1)/(_lengths[i]*_lengths[i]);
            for (int i = 0; i < _sizes.Length; i++)
                _w[i] = (_sizes[i] - 1)*(_sizes[i] - 1)/(_lengths[i]*_lengths[i])/sum2/(1.0 + a);
            _w[_sizes.Length] = (a - 1.0)/(1.0 + a);

            if (appendLineCallback != null) appendLineCallback("Коэффициенты:");
            for (int i = 0; i < _sizes.Length; i++)
                if (appendLineCallback != null)
                    appendLineCallback(string.Format("Коэффициенты по оси № {0} (у двух точек):\t{1}", i, _w[i]));
            if (appendLineCallback != null)
                appendLineCallback(string.Format("Коэффициент у средней точки:\t{0}", _w[_sizes.Length]));

            if (appendLineCallback != null && relax)
                appendLineCallback("Используется релаксация");

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            if (appendLineCallback != null) appendLineCallback("Выделяем видеоресурсы");

            //  При использовании метода релаксации задействовано в два раза меньше памяти и вычисления производятся
            //  на-месте. Для устанения коллизий с совместным доступом производится раскраска точек красное-чёрное для обработки
            //  их по-очереди
            double[][] devA = relax
                ? new[] {gpu.Allocate<double>(_a.Length)}
                : new[] {gpu.Allocate<double>(_a.Length), gpu.Allocate<double>(_a.Length)};

            double[][] devB =
            {
                gpu.Allocate<double>(_gridSize3*_blockSize3),
                gpu.Allocate(_b)
            };
            double[][] devC =
            {
                gpu.Allocate<double>(_gridSize3*_blockSize3),
                gpu.Allocate(_c)
            };
            int[] devSizes = gpu.Allocate(_sizes);
            int[][] devV =
            {
                gpu.Allocate(_extV),
                gpu.Allocate(_intV)
            };
            double[] devW = gpu.Allocate(_w);

            if (appendLineCallback != null) appendLineCallback("Копируем данные в видеопамять");
            gpu.CopyToDevice(_a, devA[0]);
            gpu.CopyToDevice(_sizes, devSizes);
            gpu.CopyToDevice(_extV, devV[0]);
            gpu.CopyToDevice(_intV, devV[1]);
            gpu.CopyToDevice(_w, devW);

            if (!relax)
            {
                if (appendLineCallback != null) appendLineCallback("Дублируем массив в видеопамяти");
                gpu.Launch(_gridSize3, _blockSize3, "Copy", devA[0], devA[1]);
            }
            var queue = new StackListQueue<double>();
            for (int step = 0;; step++)
            {
                //if (AppendLineCallback != null) AppendLineCallback(string.Format("Шаг итерации № {0}", step));

                // Вычисляем среднее взвешенное соседних точек
                //if (AppendLineCallback != null) AppendLineCallback("Вычисляем среднее взвешенное соседних точек");
                if (!relax)
                    gpu.Launch(_gridSize3, _blockSize3, "LaplaceSolver", devA[step & 1], devA[1 - (step & 1)],
                        devSizes,
                        devV[0], devV[1],
                        devW,
                        devB[0], devC[0]);
                else
                {
                    gpu.Launch(_gridSize3, _blockSize3, "Clear", devB[0]);
                    gpu.Launch(_gridSize3, _blockSize3, "Clear", devC[0]);

                    for (int p = 0; p < 2; p++)
                        gpu.Launch(_gridSize3, _blockSize3, "LaplaceSolverWithRelax", devA[0],
                            devSizes,
                            devV[0], devV[1],
                            devW,
                            devB[0], devC[0],
                            p);
                }

                // Суммируем амплитуды изменений, посчитанные в каждом процессе
                gpu.Launch(_gridSize1, _blockSize1, "Sum", devB[0], devB[1]);
                gpu.Launch(_gridSize1, _blockSize1, "Sum", devC[0], devC[1]);

                gpu.CopyFromDevice(devB[1], _b);
                gpu.CopyFromDevice(devC[1], _c);
                double deltaSum = _b[0];
                double squareSum = _c[0];

                //if (AppendLineCallback != null)
                //    AppendLineCallback(string.Format("Амплитуда изменений = {0}/{1}", deltaSum, squareSum));

                queue.Enqueue(deltaSum/squareSum);

                if (deltaSum > epsilon*squareSum) continue;

                if (appendLineCallback != null)
                    appendLineCallback(string.Format("Потребовалось {0} итераций", step + 1));

                // Если изменения меньше заданной величины, то возвращаем вычисленные значения
                if (appendLineCallback != null)
                    appendLineCallback("Копируем массив из видеопамяти в массив на компьютере");
                if (!relax)
                    gpu.CopyFromDevice(devA[1 - (step & 1)], _a);
                else
                    gpu.CopyFromDevice(devA[0], _a);

                break;
            }
            // free the memory allocated on the GPU
            if (appendLineCallback != null) appendLineCallback("Освобождаем видеоресурсы");
            gpu.FreeAll();
            return queue;
        }

        /// <summary>
        ///     Вычисление среднего взвешенного соседних по осям точек
        ///     для внутренних точек куба
        ///     и одновременно подсчитываем амплитуду изменений
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="prev"></param>
        /// <param name="next"></param>
        /// <param name="sizes"></param>
        /// <param name="extV"></param>
        /// <param name="intV"></param>
        /// <param name="w"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        [Cudafy]
        public static void LaplaceSolver(GThread thread,
            double[] prev, double[] next,
            int[] sizes,
            int[] extV, int[] intV,
            double[] w, double[] b, double[] c)
        {
            // Степень дифференциального оператора
            // Реализовано только для оператора Лапласа
            // Для больших степеней надо использовать соответствующие полиномы большей степени
            // Для дифференциального оператора степени 2 (оператора Лапласа) полином имеет степень 1
            // Для дифференциального оператора степени rank полином имеет степень rank-1
            const int rank = 2;

            double deltaSum = 0;
            double squareSum = 0;

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
                    int index = (rank >> 1) + (v%(sizes[i] - rank));
                    id += index*extV[i];
                    v = v/(sizes[i] - rank);
                }
                // Вычисляем среднее арифметическое соседних точек
                // для всех внутренних точек куба
                // и одновременно подсчитываем амплитуду изменений
                double x = prev[id];
                double y = x*w[sizes.Length];
                for (int i = 0; i < sizes.Length; i++)
                    y += (prev[id - extV[i]] + prev[id + extV[i]])*w[i];
                next[id] = y;
                double delta = (x - y);
                double square = (x + y);
                delta = delta*delta;
                square = square*square;
                deltaSum += delta;
                squareSum += square;
            }
            b[thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x] = deltaSum;
            c[thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x] = squareSum;
        }

        /// <summary>
        ///     Вычисление среднего взвешенного соседних по осям точек
        ///     для внутренних точек куба
        ///     и одновременно подсчитываем амплитуду изменений
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="array"></param>
        /// <param name="sizes"></param>
        /// <param name="extV"></param>
        /// <param name="intV"></param>
        /// <param name="w"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="p"></param>
        [Cudafy]
        public static void LaplaceSolverWithRelax(GThread thread,
            double[] array,
            int[] sizes,
            int[] extV, int[] intV,
            double[] w, double[] b, double[] c,
            int p)
        {
            // Степень дифференциального оператора
            // Реализовано только для оператора Лапласа
            // Для больших степеней надо использовать соответствующие полиномы большей степени
            // Для дифференциального оператора степени 2 (оператора Лапласа) полином имеет степень 1
            // Для дифференциального оператора степени rank полином имеет степень rank-1
            const int rank = 2;

            double deltaSum = b[thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x];
            double squareSum = c[thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x];

            // Перебор по индексам внутренних точек
            for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                tid < intV[sizes.Length];
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                // Преобразуем индекс внутренней точки в координаты
                // Преобразуем координаты в индекс точки
                // и подсчитваем чётность точки
                // Чётность точки равна сумме координат
                int id = 0;
                int parity = 0; // Чётность точки
                for (int i = 0, v = tid; i < sizes.Length; i++)
                {
                    int index = (rank >> 1) + (v%(sizes[i] - rank));
                    parity += index;
                    id += index*extV[i];
                    v = v/(sizes[i] - rank);
                }

                if (parity%2 != p) continue;

                // Вычисляем среднее арифметическое соседних точек
                // для всех внутренних точек куба
                // и одновременно подсчитываем амплитуду изменений
                double x = array[id];
                double y = x*w[sizes.Length];
                for (int i = 0; i < sizes.Length; i++)
                    y += (array[id - extV[i]] + array[id + extV[i]])*w[i];
                array[id] = y;
                double delta = (x - y);
                double square = (x + y);
                delta = delta * delta;
                square = square*square;
                deltaSum += delta;
                squareSum += square;
            }
            b[thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x] = deltaSum;
            c[thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x] = squareSum;
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

        /// <summary>
        ///     Обнуление массива в видеопамяти
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="array"></param>
        [Cudafy]
        public static void Clear(GThread thread, double[] array)
        {
            for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                tid < array.Length;
                tid += thread.blockDim.x*thread.gridDim.x)
                array[tid] = 0;
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