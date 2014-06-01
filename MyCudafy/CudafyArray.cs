using System;
using System.Diagnostics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace MyCudafy
{
    /// <summary>
    ///     Класс работы с массивом
    ///     Данный класс реализует модель специализированного вычислительного устройства
    ///     с фиксированным набором элементарных операций и использует параллельные вычисления CUDA
    ///     для реализации этой модели
    /// </summary>
    public struct CudafyArray
    {
        /// <summary>
        ///     Семафор для блокирования одновременного доступа к данному статичному классу
        ///     из разных параллельных процессов
        ///     Надеюсь CUDAfy тоже заботится о блокировании одновременного доступа к видеокарточке ;)
        /// </summary>
        public static readonly object Semaphore = new Object();

        private static int _gridSize;
        private static int _blockSize;
        private static int _length;
        private static int _ceiling;
        private static int _floor;
        private static int _middle;

        #region Регистры класса

        [Cudafy] private static int[] _a;
        [Cudafy] private static int[] _b;
        [Cudafy] private static int[] _c;
        [Cudafy] private static readonly int[] D = new int[1];
        private static int _ceilingOfCeiling;
        private static int _ceilingOfMiddle;

        #endregion

        public static int GetHash()
        {
            return D[0];
        }

        /// <summary>
        ///     Вызов и исполнение одной элементарной функции по имени функции
        /// </summary>
        /// <param name="function"></param>
        public static void Execute(string function)
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            gpu.Launch(_gridSize, _blockSize, function, devA, devB, devC, devD, 1);
            gpu.Launch(1, 1, function, devA, devB, devC, devD, 2);

            gpu.CopyFromDevice(devD, D);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Вызов и исполнение функции проверки что массив отсортирован
        /// </summary>
        public static void ExecuteSorted(int direction = 1)
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);


            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);
            int[] devC = gpu.Allocate(_c);
            int[] devD = gpu.Allocate(D);

            gpu.CopyToDevice(_a, devA);

            gpu.Launch(1, 1).Split(devA, devB, devC, _middle);
            gpu.Launch(_gridSize, _blockSize).Sorted(devA, devB, devC, devD, 0, direction);
            gpu.Launch(1, 1).Sorted(devA, devB, devC, devD, 1, direction);

            gpu.CopyFromDevice(devD, D);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Выполнение сортировки слияниями
        ///     Пример использования:
        ///     CudafySequencies.SetSequencies(arrayOfArray,arrayOfArray);
        ///     CudafySequencies.Execute("Compare");
        ///     var compare = CudafySequencies.GetMartix();
        ///     CudafyArray.SetArray(Enumerable.Range(0,n).ToArray());
        ///     CudafyArray.SetCompare(compare);
        ///     CudafyArray.MergeSort();
        ///     var indexesOfSorted = CudafyArray.GetArray();
        /// </summary>
        public static void MergeSort(int direction = 1)
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);

            gpu.CopyToDevice(_a, devA);

            for (int i = 0; i < _ceiling; i++)
            {
                int gridSize = Math.Min(15, (int) Math.Pow((_length >> i) + i, 0.333333333333));
                int blockSize = Math.Min(15, (int) Math.Pow((_length >> i) + i, 0.333333333333));
                gpu.Launch(gridSize, blockSize)
                    .MergeLinear(((i & 1) == 0) ? devA : devB, ((i & 1) == 0) ? devB : devA, i, 0,
                        _length,
                        direction);
            }
            gpu.CopyFromDevice(((_ceiling & 1) == 0) ? devA : devB, _a);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Выполнение чётно-нечётной сортировки
        ///     Пример использования:
        ///     CudafySequencies.SetSequencies(arrayOfArray,arrayOfArray);
        ///     CudafySequencies.Execute("Compare");
        ///     var compare = CudafySequencies.GetMartix();
        ///     CudafyArray.SetArray(Enumerable.Range(0,n).ToArray());
        ///     CudafyArray.SetCompare(compare);
        ///     CudafyArray.OddEvenSort();
        ///     var indexesOfSorted = CudafyArray.GetArray();
        /// </summary>
        public static void OddEvenSort(int direction = 1)
        {
            /*
            Для каждой итерации алгоритма операции сравнения-обмена для всех пар элементов независимы и
            выполняются одновременно. Рассмотрим случай, когда число процессоров равно числу элементов, т.е. p=n -
            число процессоров (сортируемых элементов). Предположим, что вычислительная система имеет топологию
            кольца. Пусть элементы ai (i = 1, .. , n), первоначально расположены на процессорах pi (i = 1, ... , n). В нечетной
            итерации каждый процессор с нечетным номером производит сравнение-обмен своего элемента с элементом,
            находящимся на процессоре-соседе справа. Аналогично в течение четной итерации каждый процессор с четным
            номером производит сравнение-обмен своего элемента с элементом правого соседа.
            На каждой итерации алгоритма нечетные и четные процессоры выполняют шаг сравнения-обмена с их
            правыми соседями за время Q(1). Общее количество таких итераций – n; поэтому время выполнения
            параллельной сортировки – Q(n).
            Когда число процессоров p меньше числа элементов n, то каждый из процессов получает свой блок
            данных n/p и сортирует его за время Q((n/p)·log(n/p)). Затем процессоры проходят p итераций (р/2 и чётных, и
            нечётных) и делают сравнивания-разбиения: смежные процессоры передают друг другу свои данные, а
            внутренне их сортируют (на каждой паре процессоров получаем одинаковые массивы). Затем удвоенный
            массив делится на 2 части; левый процессор обрабатывает далее только левую часть (с меньшими значениями
            данных), а правый – только правую (с большими значениями данных). Получаем отсортированный массив
            после p итераций.
            */
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);

            // Шаг первый - копируем исходный массив в память GPU 

            gpu.CopyToDevice(_a, devA);

            // запускаем задачи сортировки блоков
            // На выходе - отсортированные массивы размера до 1<<_middle

            for (int i = 1; i <= _middle; i++)
                for (int j = i; j-- > 0;)
                    gpu.Launch(_gridSize, _blockSize).Bitonic(devA, devB, j, i, _length, _middle, direction);

            if ((_length & ((1 << _middle) - 1)) != 0)
            {
                for (int i = 0; i <= _ceilingOfMiddle; i++)
                    gpu.Launch(_gridSize, _blockSize)
                        .MergeExponental(((i & 1) == 0) ? devA : devB, ((i & 1) == 0) ? devB : devA, i,
                            _length, _middle,
                            direction);
                if ((_ceilingOfMiddle & 1) == 0)
                    gpu.CopyOnDevice(devB, _length & -(1 << _middle), devA,
                        _length & -(1 << _middle), _length & ((1 << _middle) - 1));
            }

            // запускаем задачи сортировки данных в двух соседних блоках
            // чередуя соседние блоки

            for (int i = 0; i < ((_length >> _middle) + 1); i++)
                gpu.Launch(_gridSize, _blockSize)
                    .MergeLinear(((i & 1) == 0) ? devA : devB, ((i & 1) == 0) ? devB : devA,
                        _middle, i & 1, _length,
                        direction);

            gpu.CopyFromDevice((((_length >> _middle) & 1) == 0) ? devB : devA, _a);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Выполнение битонической сортировки
        ///     Пример использования:
        ///     CudafySequencies.SetSequencies(arrayOfArray,arrayOfArray);
        ///     CudafySequencies.Execute("Compare");
        ///     var compare = CudafySequencies.GetMartix();
        ///     CudafyArray.SetArray(Enumerable.Range(0,n).ToArray());
        ///     CudafyArray.SetCompare(compare);
        ///     CudafyArray.BitonicSort();
        ///     var indexesOfSorted = CudafyArray.GetArray();
        /// </summary>
        public static void BitonicSort(int direction = 1)
        {
            /*
            В основе этой сортировки лежит операция Bn(полуочиститель, half - cleaner) над массивом, параллельно
            упорядочивающая элементы пар xi и xi + n / 2.На рис. 1 полуочиститель может упорядочивать элементы пар как по
            возрастанию, так и по убыванию.Сортировка основана на понятии битонической последовательности и
            утверждении : если набор полуочистителей правильно сортирует произвольную последовательность нулей и
            единиц, то он корректно сортирует произвольную последовательность.
            Последовательность a0, a1, …, an - 1 называется битонической, если она или состоит из двух монотонных
            частей(т.е.либо сначала возрастает, а потом убывает, либо наоборот), или получена путем циклического
            сдвига из такой последовательности.Так, последовательность 5, 7, 6, 4, 2, 1, 3 битоническая, поскольку
            получена из 1, 3, 5, 7, 6, 4, 2 путем циклического сдвига влево на два элемента.
            Доказано, что если применить полуочиститель Bn к битонической последовательности a0, a1, …, an - 1,
            то получившаяся последовательность обладает следующими свойствами :
            • обе ее половины также будут битоническими.
            • любой элемент первой половины будет не больше любого элемента второй половины.
            • хотя бы одна из половин является монотонной.
            Применив к битонической последовательности a0, a1, …, an - 1 полуочиститель Bn, получим две
            последовательности длиной n / 2, каждая из которых будет битонической, а каждый элемент первой не превысит
            каждый элемент второй.Далее применим к каждой из получившихся половин полуочиститель Bn / 2.Получим
            уже четыре битонические последовательности длины n / 4.Применим к каждой из них полуочиститель Bn / 2 и
            продолжим этот процесс до тех пор, пока не придем к n / 2 последовательностей из двух элементов.Применив к
            каждой из них полуочиститель B2, отсортируем эти последовательности.Поскольку все последовательности
            уже упорядочены, то, объединив их, получим отсортированную последовательность.
            Итак, последовательное применение полуочистителей Bn, Bn / 2, …, B2 сортирует произвольную
            битоническую последовательность.Эту операцию называют битоническим слиянием и обозначают Mn.
            Например, к последовательности из 8 элементов a 0, a1, …, a7 применим полуочиститель B2, чтобы на
            соседних парах порядок сортировки был противоположен.На рис. 2 видно, что первые четыре элемента
            получившейся последовательности образуют битоническую последовательность.Аналогично последние
            четыре элемента также образуют битоническую последовательность.Поэтому каждую из этих половин можно
            отсортировать битоническим слиянием, однако проведем слияние таким образом, чтобы направление
            сортировки в половинах было противоположным.В результате обе половины образуют вместе битоническую
            Битоническая сортировка последовательности из n элементов разбивается пополам и каждая из
            половин сортируется в своем направлении.После этого полученная битоническая последовательность
            сортируется битоническим слиянием.
            */
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            int[] devA = gpu.Allocate(_a);
            int[] devB = gpu.Allocate(_b);

            gpu.CopyToDevice(_a, devA);

            // Число n представимо в виде суммы степеней двойки,
            // Поэтому, разбиваем исходные данные на подмассивы с длинами равными слагаемым этой суммы
            // и сортируем каждый подмассив битоническим алгоритмом 
            // В разультате получим равное числу слагаеммых отсортированных массивов длинами равным степеням двойки

            for (int i = 1; i <= _floor; i++)
                for (int j = i; j-- > 0;)
                    gpu.Launch(_gridSize, _blockSize).Bitonic(devA, devB, j, i, _length, _ceiling, direction);

            // Теперь надо произвести слияние уже отсортированных массивов

            for (int i = 0; i <= _ceilingOfCeiling; i++)
                gpu.Launch(_gridSize, _blockSize)
                    .MergeExponental(((i & 1) == 0) ? devA : devB, ((i & 1) == 0) ? devB : devA,
                        i, _length, _ceiling,
                        direction);

            gpu.CopyFromDevice((((_ceilingOfCeiling & 1) == 0) ? devA : devB), _a);

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        [Cudafy]
        public static void MergeExponental(GThread thread, int[] a, int[] b,
            int i, int length, int ceiling, int direction)
        {
            for (int tid = (2*(thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x) << i);
                tid < ceiling;
                tid += 2*thread.blockDim.x*thread.gridDim.x << i)
            {
                int index0 = length & -(1 << Math.Min(Math.Max(0, (tid) + (2 << i)), ceiling));
                int index1 = length & -(1 << Math.Min(Math.Max(0, (tid) + (1 << i)), ceiling));
                int index2 = length & -(1 << Math.Min(Math.Max(0, (tid) + (0 << i)), ceiling));
                int n0 = index1 - index0;
                int n1 = index2 - index1;
                int total = index2 - index0;
                while (n0 > 0 && n1 > 0)
                {
                    if (direction*(a[index0 + n0 - 1] - a[index1 + n1 - 1]) > 0)
                        b[index0 + --total] = a[index0 + --n0];
                    else
                        b[index0 + --total] = a[index1 + --n1];
                }
                while (n0 > 0) b[index0 + --total] = a[index0 + --n0];
                while (n1 > 0) b[index0 + --total] = a[index1 + --n1];
            }
        }

        [Cudafy]
        public static void Bitonic(GThread thread, int[] a, int[] b,
            int j, int i, int length, int k, int direction)
        {
            int step = 1 << j;
            for (int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < (length & (-1 << i));
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                if ((tid & step) == step) continue;
                int parity = (((tid) & ((1 << k) - 1)) >> i) & 1;
                while (parity > 1) parity = (parity >> 1) ^ (parity & 1);
                parity = 1 - (parity << 1);
                int value = parity*direction*(a[tid & ~step] - a[tid | step]);
                if (value <= 0) continue;
                int tmp = a[(tid & ~step)];
                a[tid & ~step] = a[tid | step];
                a[tid | step] = tmp;
            }
        }

        [Cudafy]
        public static void MergeLinear(GThread thread, int[] a, int[] b,
            int i, int parity, int length, int direction)
        {
            for (
                int tid = (2*(thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x) << i) - (parity << i);
                tid < length + (parity << i);
                tid += 2*thread.blockDim.x*thread.gridDim.x << i)
            {
                int index0 = Math.Min(Math.Max(0, tid + (0 << i)), length);
                int index1 = Math.Min(Math.Max(0, tid + (1 << i)), length);
                int index2 = Math.Min(Math.Max(0, tid + (2 << i)), length);
                int n0 = index1 - index0;
                int n1 = index2 - index1;
                int total = index2 - index0;
                while (n0 > 0 && n1 > 0)
                {
                    if (direction*(a[index0 + n0 - 1] - a[index1 + n1 - 1]) > 0)
                        b[index0 + --total] = a[index0 + --n0];
                    else
                        b[index0 + --total] = a[index1 + --n1];
                }
                while (n0 > 0) b[index0 + --total] = a[index0 + --n0];
                while (n1 > 0) b[index0 + --total] = a[index1 + --n1];
            }
        }

        [Cudafy]
        public static void Hash(GThread thread, int[] a, int[] b, int[] c, int[] d, int step)
        {
            switch (step)
            {
                case 1:
                    for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                        tid < c.Length;
                        tid += thread.blockDim.x*thread.gridDim.x)
                    {
                        int index0 = tid*a.Length/c.Length;
                        int index1 = (tid + 1)*a.Length/c.Length;
                        c[tid] = a[index0];
                        for (int i = index0 + 1; i < index1; i++)
                            c[tid] = (c[tid] << 1) ^ (c[tid] >> (8*sizeof (int) - 1)) ^ a[i];
                    }
                    break;
                case 2:
                    for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                        tid < 1;
                        tid += thread.blockDim.x*thread.gridDim.x)
                    {
                        d[0] = b[0];
                        for (int i = 1; i < c.Length; i++)
                            d[0] = (d[0] << 1) ^ (d[0] >> (8*sizeof (int) - 1)) ^ c[i];
                    }
                    break;
            }
        }

        [Cudafy]
        public static void Sorted(GThread thread, int[] a, int[] b, int[] c, int[] d, int step,
            int direction)
        {
            switch (step)
            {
                case 0:
                    for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                        tid < c.Length;
                        tid += thread.blockDim.x*thread.gridDim.x)
                    {
                        int index0 = tid*a.Length/c.Length;
                        int index1 = (tid + 1)*a.Length/c.Length;
                        c[tid] = 1;
                        for (int i = index0; i < index1 - 1; i++)
                            c[tid] = (direction*(a[i] - a[i + 1]) <= 0) ? 1 : 0;
                    }
                    break;
                case 1:
                    for (int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                        tid < 1;
                        tid += thread.blockDim.x*thread.gridDim.x)
                    {
                        d[0] = c[0];
                        for (int i = 1; i < b.Length && d[0] != 0; i++)
                            d[0] = c[i];
                    }
                    break;
            }
        }

        public static void SetArray(int[] array)
        {
            _length = array.Length;
            _ceiling = (int) Math.Ceiling(Math.Log(_length, 2));
            _middle = (_ceiling + _floor)/3;
            _floor = (int) Math.Floor(Math.Log(_length, 2));
            _ceilingOfCeiling = (int) Math.Ceiling(Math.Log(_ceiling, 2));
            _ceilingOfMiddle = (int) Math.Ceiling(Math.Log(_middle, 2));
            Debug.Assert(Math.Max(1 << (_ceiling - _middle), 1 << _middle)*
                         Math.Min(1 << (_ceiling - _middle), 1 << _middle) == (1 << _ceiling));
            Debug.Assert(Math.Max(1 << (_floor - _middle), 1 << _middle)*
                         Math.Min(1 << (_floor - _middle), 1 << _middle) == (1 << _floor));
            _a = array;
            _b = new int[array.Length];
            _c = new int[(_length >> _middle) + _middle + 1];
            _gridSize = Math.Min(15, (int) Math.Pow((_length >> _middle) + (1 << _middle), 0.333333333333));
            _blockSize = Math.Min(15, (int) Math.Pow((_length >> _middle) + (1 << _middle), 0.333333333333));
        }

        public static int[] GetArray()
        {
            return _a;
        }

        public static int GetSorted()
        {
            return D[0];
        }

        /// <summary>
        ///     Шаг вычисления максимума в массиве
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="prev"></param>
        /// <param name="next"></param>
        [Cudafy]
        public static void Max(GThread thread,
            int[] prev, int[] next)
        {
            for (int tid = (thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x);
                tid < next.Length;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                next[tid] = 0;
                for (int i = 0; i * next.Length + tid < prev.Length; i++)
                {
                    int index = i * next.Length + tid;
                    if (prev[index] > next[tid]) next[tid] = prev[index];
                }
            }
        }

        /// <summary>
        ///     Умножение элемента в массива на его индекс
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="prev"></param>
        /// <param name="next"></param>
        [Cudafy]
        public static void MulByIndex(GThread thread,
            int[] prev, int[] next)
        {
            for (int tid = (thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x);
                tid < next.Length;
                tid += thread.blockDim.x * thread.gridDim.x)
            {
                next[tid] = prev[tid] * tid;
            }
        }
    }
}