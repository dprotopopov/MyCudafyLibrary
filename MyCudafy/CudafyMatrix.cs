using System;
using System.Collections.Generic;
using System.Diagnostics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using MyLibrary.Collections;

namespace MyCudafy
{
    /// <summary>
    ///     Класс работы с матрицей
    ///     Данный класс реализует модель специализированного вычислительного устройства
    ///     с фиксированным набором элементарных операций и использует параллельные вычисления CUDA
    ///     для реализации этой модели
    /// </summary>
    public struct CudafyMatrix
    {
        /// <summary>
        ///     Семафор для блокирования одновременного доступа к данному статичному классу
        ///     из разных параллельных процессов
        ///     Надеюсь CUDAfy тоже заботится о блокировании одновременного доступа к видеокарточке ;)
        /// </summary>
        public static readonly object Semaphore = new Object();

        #region Регистры класса

        [Cudafy] private static int[,] _a;
        [Cudafy] private static int[,] _b;
        [Cudafy] private static int[] _c;
        [Cudafy] private static int[] _d;
        [Cudafy] private static readonly int[] E = new int[1];

        #endregion

        #region Установка текущих значений в регистрах (setter)

        public static void SetMatrix(int[,] value)
        {
            int rows = value.GetLength(0);
            int columns = value.GetLength(1);
            _a = value;
            _b = new int[rows, columns];
            _c = new int[rows];
            _d = new int[columns];
        }

        public static int[,] GetMatrix()
        {
            return _a;
        }

        public static void SetIndexes(int[] value)
        {
            _c = value;
        }

        public static void SetRow(int[] value)
        {
            _d = value;
        }

        #endregion

        /// <summary>
        ///     Вычисление целевой функции для симплекс-метода
        ///     Функция Мак-Лейна вычисляется после применения к данной матрице обратимого преобразования,
        ///     то есть умножение данной матрицы на обратимую матрицу, задаваемую списком индексов,
        ///     где строка i исходной матрицы должна быть добавлена к строкам i и indexes[i] результирующей матрицы, если
        ///     i!=indexes[i], и только к строке i результирующей матрицы, если i==indexes[i]
        ///     Таким образом, каждый базис в этом пространстве получается из данного базиса при помощи цепочки элементарных
        ///     преобразований. А на матричном языке проблема распознавания планарности сводится к нахождению такой матрицы в
        ///     классе эквивалентных матриц (т.е. матриц, которые получаются друг из друга при помощи элементарных преобразований
        ///     над строками), у которой в каждом столбце содержится не более двух единиц [6].
        ///     Указанный критерий позволяет разработать методику определения планарности графа, сводя проблему планарности к
        ///     отысканию минимума некоторого функционала на множестве базисов подпространства квазициклов. Определим следующий
        ///     функционал на матрице С, соответствующий базису подпространства квазициклов (и будем его впредь называть
        ///     функционалом Мак-Лейна)
        ///     Очевидно, что матрица С соответствует базису Мак-Лейна (т.е. базису, удовлетворяющему условию Мак-Лейна) тогда и
        ///     только тогда, когда F(С) = 0.
        /// </summary>
        public static void ExecuteMacLane()
        {
            Execute(new[] {"Push", "MultiplyBtoAbyC", "CountByColumn", "MacLane", "SumRow"},
                (int) Register.A + (int) Register.C, (int) Register.E);
        }

        /// <summary>
        ///     Вычисление матрицы, получаемой в результате применения обратимого преобразования,
        ///     то есть умножение данной матрицы на обратимую матрицу, задаваемую списком индексов,
        ///     где строка i исходной матрицы должна быть добавлена к строкам i и indexes[i] результирующей матрицы, если
        ///     i!=indexes[i], и только к строке i результирующей матрицы, если i==indexes[i]
        /// </summary>
        public static void ExecuteUpdate()
        {
            Execute(new[] {"Push", "MultiplyBtoAbyC"},
                (int) Register.A + (int) Register.C, (int) Register.A);
        }

        /// <summary>
        ///     Вычисление минимальной суммы элементов в строках.
        ///     Одновременно вычисляется сумма элементов в строках.
        /// </summary>
        public static void ExecuteCountMinInColumn()
        {
            var list = new StackListQueue<string> {"RepeatZero", "IndexOfNonZero", "Count", "MinInColumn"};
            Execute(list, (int) Register.A, (int) Register.C + (int) Register.E);
        }

        public static void ExecuteRepeatZeroIndexOfNonZero()
        {
            var list = new StackListQueue<string> {"RepeatZero", "IndexOfNonZero"};
            Execute(list, (int) Register.A, (int) Register.C);
        }

        public static void ExecuteRangeSelectFirstIndexOfNonNegative()
        {
            var list = new StackListQueue<string> {"Range", "Select", "FirstIndexOfNonNegative"};
            Execute(list, (int) Register.A, (int) Register.E);
        }

        public static void ExecuteRangeSelectFirstIndexOfNonZero()
        {
            var list = new StackListQueue<string> {"Range", "Select", "FirstIndexOfNonZero"};
            Execute(list, (int) Register.A, (int) Register.E);
        }

        public static void ExecuteRepeatZeroIndexOfZeroFirstIndexOfNonPositive()
        {
            var list = new StackListQueue<string> {"RepeatZero", "IndexOfZero", "FirstIndexOfNonPositive"};
            Execute(list, (int) Register.A, (int) Register.E);
        }

        public static void ExecuteRepeatZeroIndexOfZero()
        {
            var list = new StackListQueue<string> {"RepeatZero", "IndexOfZero"};
            Execute(list, (int) Register.A, (int) Register.C);
        }

        public static void ExecuteRepeatZeroCountOfZeroMinInColumn()
        {
            var list = new StackListQueue<string> {"RepeatZero", "CountOfZero", "MinInColumn"};
            Execute(list, (int) Register.A, (int) Register.E);
        }

        /// <summary>
        ///     Копирование регистра _a (матрица) в регистр _b (матрица)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void Push(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows*columns;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid/columns;
                int column = tid%columns;
                b[row, column] = a[row, column];
            }
        }

        /// <summary>
        ///     Копирование регистра _b (матрица) в регистр _a (матрица)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void Pop(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows*columns;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid/columns;
                int column = tid%columns;
                a[row, column] = b[row, column];
            }
        }

        /// <summary>
        ///     Прибавление к строкам регистра _a (матрица) строк регистра _b (матрица), задаваемых индексами строк
        ///     в регистре _c (столбец)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void MultiplyBtoAbyC(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < columns;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int column = tid;
                for (int row = 0; row < rows; row++)
                {
                    if (c[row] == row) continue;
                    a[c[row], column] ^= b[row, column];
                }
            }
        }

        /// <summary>
        ///     Очистка регистра _c (столбец)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void RepeatZero(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = c.GetLength(0);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
                c[tid] = 0;
        }

        /// <summary>
        ///     Заполнение регистра _c (столбец)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void Range(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = c.GetLength(0);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
                c[tid] = tid;
        }

        /// <summary>
        ///     Заполнение регистра _c (столбец) выборкой из строк регистра _a (матрица)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void Select(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = c.GetLength(0);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
                c[tid] = a[tid, c[tid]];
        }

        /// <summary>
        ///     Суммирование элементов строк регистра _a (матрица) в регистр _c (столбец)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void Count(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid;
                int column = c[row];
                for (c[row] = 0; 0 <= column && column < columns; column++)
                    c[row] += a[row, column];
            }
        }

        [Cudafy]
        public static void FirstIndexOfNonZero(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < 1;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = 0;
                for (e[0] = -1; row < rows && e[0] < 0; row++)
                    if (c[row] != 0)
                        e[0] = row;
            }
        }

        [Cudafy]
        public static void FirstIndexOfNonNegative(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < 1;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = 0;
                for (e[0] = -1; row < rows && e[0] < 0; row++)
                    if (c[row] >= 0)
                        e[0] = row;
            }
        }

        [Cudafy]
        public static void FirstIndexOfNonPositive(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < 1;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = 0;
                for (e[0] = -1; row < rows && e[0] < 0; row++)
                    if (c[row] <= 0)
                        e[0] = row;
            }
        }

        [Cudafy]
        public static void MinInColumn(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < 1;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                e[0] = c[0];
                for (int row = 1; row < rows; row++)
                    if (e[0] > c[row])
                        e[0] = c[row];
            }
        }


        [Cudafy]
        public static void SumRow(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int columns = d.Length;
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < 1;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                e[0] = d[0];
                for (int column = 1; column < columns; column++)
                    e[0] += d[column];
            }
        }

        [Cudafy]
        public static void CountByColumn(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < columns;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                d[tid] = a[0, tid];
                for (int row = 1; row < rows; row++)
                    d[tid] += a[row, tid];
            }
        }

        [Cudafy]
        public static void MacLane(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int columns = d.Length;
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < columns;
                tid += thread.blockDim.x*thread.gridDim.x)
                d[tid] = (d[tid] - 1)*(d[tid] - 2);
        }

        /// <summary>
        ///     Вызов и исполнение одной элементарной функции по имени функции
        /// </summary>
        /// <param name="function"></param>
        public static void Execute(IEnumerable<string> functions, int input, int output)
        {
            Debug.Assert(input != 0 && output != 0);

            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[,] devA = gpu.Allocate(_a);
            int[,] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(_d);
            int[] devE = gpu.Allocate(E);

            if ((input & (int) Register.A) != 0) gpu.CopyToDevice(_a, devA);
            if ((input & (int) Register.B) != 0) gpu.CopyToDevice(_b, devB);
            if ((input & (int) Register.C) != 0) gpu.CopyToDevice(_c, devC);
            if ((input & (int) Register.D) != 0) gpu.CopyToDevice(_d, devD);
            if ((input & (int) Register.E) != 0) gpu.CopyToDevice(E, devE);

            int rows = _a.GetLength(0);
            int columns = _a.GetLength(1);

            dim3 gridSize = Math.Min(15, (int) Math.Pow(rows*columns, 0.33333333333));
            dim3 blockSize = Math.Min(15, (int) Math.Pow(rows*columns, 0.33333333333));

            foreach (string function in functions)
                gpu.Launch(gridSize, blockSize, function, devA, devB, devC, devD, devE);

            if ((output & (int) Register.A) != 0) gpu.CopyFromDevice(devA, _a);
            if ((output & (int) Register.B) != 0) gpu.CopyFromDevice(devB, _b);
            if ((output & (int) Register.C) != 0) gpu.CopyFromDevice(devC, _c);
            if ((output & (int) Register.D) != 0) gpu.CopyFromDevice(devD, _d);
            if ((output & (int) Register.E) != 0) gpu.CopyFromDevice(devE, E);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Приведение матрицы к "каноническому" виду, методом Гаусса-Жордана,
        ///     то есть к матрице, получаемой в результате эквивалентных преобразований
        ///     над строками, и у которой выполнено следующее - если i - индекс первого ненулевого значения в строке, то во всех
        ///     остальных строках матрицы по индексу i содержится только ноль.
        ///     Очевидно, что если индекса первого нулевого значения нет (-1), то вся строка нулевая.
        ///     Приведение матрицы к каноническому виду используется при решении систем линейных уравнений и при поиске
        ///     фундаментальной системы решений системы линейных уравнений.
        ///     В данной реализации используется матрица на полем GF(2), то есть булева матрица.
        /// </summary>
        /// <param name="function"></param>
        public static void ExecuteGaussJordan()
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[,] devA = gpu.Allocate(_a);
            int[,] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(_d);
            int[] devE = gpu.Allocate(E);

            gpu.CopyToDevice(_a, devA);

            int rows = _a.GetLength(0);
            int columns = _a.GetLength(1);

            dim3 gridSize = Math.Min(15, (int) Math.Pow(rows*columns, 0.33333333333));
            dim3 blockSize = Math.Min(15, (int) Math.Pow(rows*columns, 0.33333333333));

            gpu.Launch(gridSize, blockSize, "RepeatZero", devA, devB, devC, devD, devE);
            for (int i = 0; i < Math.Min(rows, columns); i++)
            {
                gpu.Launch(gridSize, blockSize, "IndexOfNonZero", devA, devB, devC, devD, devE);
                gpu.CopyFromDevice(devC, _c);
                while (i < Math.Min(rows, columns) && _c[i] == -1) i++;
                if (i >= Math.Min(rows, columns)) break;
                int j = _c[i];
                gpu.Launch(gridSize, blockSize, "BooleanGaussJordan", devA, devB, i, j);
                int[,] t = devA;
                devA = devB;
                devB = t;
            }

            gpu.CopyFromDevice(devA, _a);
            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        [Cudafy]
        public static void GaussJordan(GThread thread, int[,] prev, int[,] next, int row, int col)
        {
            //Debug.Assert(prev.GetLength(0) == next.GetLength(0));
            //Debug.Assert(prev.GetLength(1) == next.GetLength(1));

            int rows = prev.GetLength(0);
            int columns = prev.GetLength(1);

            int d = prev[row, col];

            for (int tid = thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x;
                tid < rows * columns;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int i = tid / columns;
                int j = tid % columns;
                if (i == row && j == col)
                    next[i, j] = 1;
                else if (j == col)
                    next[i, j] = 0;
                else if (i == row)
                {
                    int a = prev[i, j];
                    int y = a / d;
                    next[i, j] = y;
                }
                else
                {
                    int a = prev[i, j];
                    int b = prev[i, col];
                    int c = prev[row, j];
                    int y = a - (b * c / d);
                    next[i, j] = y;
                }
            }
        }
        [Cudafy]
        public static void BooleanGaussJordan(GThread thread, int[,] prev, int[,] next, int row, int col)
        {
            //Debug.Assert(prev.GetLength(0) == next.GetLength(0));
            //Debug.Assert(prev.GetLength(1) == next.GetLength(1));

            int rows = prev.GetLength(0);
            int columns = prev.GetLength(1);

            int d = prev[row, col];

            for (int tid = thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x;
                tid < rows * columns;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                int i = tid / columns;
                int j = tid % columns;
                if (i == row && j == col)
                    next[i, j] = 1;
                else if (j == col)
                    next[i, j] = 0;
                else if (i == row)
                {
                    int a = prev[i, j];
                    int y = a * d;
                    next[i, j] = y;
                }
                else
                {
                    int a = prev[i, j];
                    int b = prev[i, col];
                    int c = prev[row, j];
                    int y = a ^ (b * c * d);
                    next[i, j] = y;
                }
            }
        }

        #region Подсчёт элемента в строке

        /// <summary>
        ///     Подсчёт ненулевого элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void CountOfNonZero(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid;
                int column = c[row];
                for (c[row] = 0; 0 <= column && column < columns; column++)
                    if (a[row, column] != 0)
                        c[row]++;
            }
        }

        /// <summary>
        ///     Подсчёт нулевого элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void CountOfZero(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid;
                int column = c[row];
                for (c[row] = 0; 0 <= column && column < columns; column++)
                    if (a[row, column] == 0)
                        c[row]++;
            }
        }

        /// <summary>
        ///     Подсчёт неотрицательного элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void CountOfNonNegative(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid;
                int column = c[row];
                for (c[row] = 0; 0 <= column && column < columns; column++)
                    if (a[row, column] >= 0)
                        c[row]++;
            }
        }

        /// <summary>
        ///     Подсчёт неположительного элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void CountOfNonPositive(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid;
                int column = c[row];
                for (c[row] = 0; 0 <= column && column < columns; column++)
                    if (a[row, column] <= 0)
                        c[row]++;
            }
        }

        #endregion

        #region Нахождение индекса первого элемента в строке

        /// <summary>
        ///     Нахождение индекса первого ненулевого элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        ///     Если элемент не найден, то возвращаемое значение -1
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void IndexOfNonZero(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid;
                int column = c[row];
                for (c[row] = -1; 0 <= column && column < columns; column++)
                    if (a[row, column] != 0)
                    {
                        c[row] = column;
                        break;
                    }
            }
        }

        /// <summary>
        ///     Нахождение индекса первого нулевого элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        ///     Если элемент не найден, то возвращаемое значение -1
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void IndexOfZero(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid;
                int column = c[row];
                for (c[row] = -1; 0 <= column && column < columns; column++)
                    if (a[row, column] == 0)
                    {
                        c[row] = column;
                        break;
                    }
            }
        }

        /// <summary>
        ///     Нахождение индекса первого неотрицательного элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        ///     Если элемент не найден, то возвращаемое значение -1
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void IndexOfNonNegative(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid;
                int column = c[row];
                for (c[row] = -1; 0 <= column && column < columns; column++)
                    if (a[row, column] >= 0)
                    {
                        c[row] = column;
                        break;
                    }
            }
        }

        /// <summary>
        ///     Нахождение индекса первого неположительного элемента в строке регистра _a (матрица)
        ///     и сохранение результата в регистр _c (столбец)
        ///     Если элемент не найден, то возвращаемое значение -1
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="c"></param>
        /// <param name="d"></param>
        [Cudafy]
        public static void IndexOfNonPositive(GThread thread, int[,] a, int[,] b, int[] c, int[] d, int[] e)
        {
            int rows = a.GetLength(0);
            int columns = a.GetLength(1);
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < rows;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int row = tid;
                int column = c[row];
                for (c[row] = -1; 0 <= column && column < columns; column++)
                    if (a[row, column] <= 0)
                    {
                        c[row] = column;
                        break;
                    }
            }
        }

        #endregion

        #region Получение текущих значений в регистрах (getter)

        public static int[] GetRow()
        {
            return _d;
        }

        public static int[] GetIndexes()
        {
            return _c;
        }

        public static int[] GetCounts()
        {
            return _c;
        }

        public static int GetMacLane()
        {
            return E[0];
        }

        public static int GetFirst()
        {
            return E[0];
        }

        public static int GetMinCount()
        {
            return E[0];
        }

        #endregion
    }
}