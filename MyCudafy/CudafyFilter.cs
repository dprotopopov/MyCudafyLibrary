using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace MyCudafy
{
    /// <summary>
    ///     Класс фильтра изображения
    /// </summary>
    public struct CudafyFilter
    {
        public static readonly object Semaphore = new Object();

        private static Bitmap _newbmp;
        private static Bitmap _oldbmp;
        private static int _n;
        private static int _nh;
        private static int _nhIndex;
        private static int _height;
        private static int _width;
        private static int _videoMemorySize = 1 << 16;
        private static int _itemSize;
        private static int _ceilingItemSize;
        private static int _ceilingMiddleSize;
        private static int _itemsCount;
        private static int _frameHeight;
        private static int _frameWidth;
        private static int _frameCount;
        private static int _frameItemsCount;

        [Cudafy] private static byte[] _color;

        #region Входные и выходные цветовые каналы

        private static byte[] _b0;
        private static byte[] _g0;
        private static byte[] _r0;
        private static byte[] _r1;
        private static byte[] _g1;
        private static byte[] _b1;

        #endregion

        /// <summary>
        ///     Инициализация алгоритма исходным изображением
        ///     и рассчёт параметров алгоритма, на основании разрещённого размера видеопамяти
        /// </summary>
        /// <param name="btm"></param>
        /// <param name="step"></param>
        /// <param name="memorySize"></param>
        public static void SetBitmap(Bitmap btm, int step, int memorySize)
        {
            _videoMemorySize = memorySize;
            if (_newbmp != null)
                _newbmp.Dispose();
            if (_oldbmp != null)
                _oldbmp.Dispose();
            _n = step%2 == 0 ? step += 1 : step;
            _nh = _n/2;
            _itemSize = (_n)*(_n);
            _oldbmp = new Bitmap(btm);
            _newbmp = new Bitmap(btm.Width, btm.Height);
            _height = btm.Height;
            _width = btm.Width;

            // Входные и выходные цветовые байт-каналы
            _r0 = new byte[_width*_height];
            _g0 = new byte[_width*_height];
            _b0 = new byte[_width*_height];
            _r1 = new byte[_width*_height];
            _g1 = new byte[_width*_height];
            _b1 = new byte[_width*_height];
            for (int y = 0; y < _height; y++)
                for (int x = 0; x < _width; x++)
                {
                    Color c = _oldbmp.GetPixel(x, y);
                    _r0[y*_width + x] = c.R;
                    _g0[y*_width + x] = c.G;
                    _b0[y*_width + x] = c.B;
                }

            Debug.Assert(_n*_n == (4*_nh*_nh + 4*_nh + 1));
            Debug.Assert(_width >= _n);
            Debug.Assert(_height >= _n);


            _ceilingItemSize = (int) Math.Ceiling(Math.Log(_itemSize, 2));
            _ceilingMiddleSize = 2*_ceilingItemSize/3;
            _nhIndex = 2*_nh*_nh + 2*_nh;
            _itemsCount = (_width - 2*_nh)*(_height - 2*_nh);

            // Рассчитывает размер фрейма, который может поместиться в заданном размере видеопамяти
            _frameHeight = (int) Math.Min(Math.Sqrt(_videoMemorySize/_itemSize), _height);
            _frameWidth = (int) Math.Min(Math.Sqrt(_videoMemorySize/_itemSize), _width);
            _frameItemsCount = (_frameWidth - 2*_nh)*(_frameHeight - 2*_nh);
            _frameCount = ((_width - 2*_nh + _frameWidth - 2*_nh - 1)/(_frameWidth - 2*_nh))*
                          ((_height - 2*_nh + _frameHeight - 2*_nh - 1)/(_frameHeight - 2*_nh));
            _color = new byte[_frameHeight*_frameWidth];

            Debug.WriteLine("Height\t:" + _height);
            Debug.WriteLine("Width\t:" + _width);
            Debug.WriteLine("n\t:" + _n);
            Debug.WriteLine("nh\t:" + _nh);
            Debug.WriteLine("nhIndex\t:" + _nhIndex);
            Debug.WriteLine("ceilingItemSize\t:" + _ceilingItemSize);
            Debug.WriteLine("ceilingMiddleSize\t:" + _ceilingMiddleSize);
            Debug.WriteLine("videoMemorySize\t:" + _videoMemorySize);
            Debug.WriteLine("_frameHeight\t:" + _frameHeight);
            Debug.WriteLine("_frameWidth\t:" + _frameWidth);
            Debug.WriteLine("_frameItemsCount\t:" + _frameItemsCount);
            Debug.WriteLine("_frameCount\t:" + _frameCount);
        }

        /// <summary>
        ///     Получение обработанного алгоритмом изображения
        /// </summary>
        /// <returns></returns>
        public static Bitmap GetBitmap()
        {
            for (int y = 0; y < _height; y++)
                for (int x = 0; x < _width; x++)
                {
                    _newbmp.SetPixel(x, y,
                        Color.FromArgb(_r1[y*_width + x], _g1[y*_width + x], _b1[y*_width + x]));
                }
            return _newbmp;
        }

        /// <summary>
        ///     Применение алгоритма медианного фильтра
        ///     Пример использования
        ///     lock (CudafyFilter.Semaphore)
        ///     {
        ///     CudafyFilter.SetBitmap( bitmap, 3, 1<<12);
        ///                                              CudafyFilter.MedianFilter();
        ///                                              bitmap= CudafyFilter.GetBitmap();
        ///     }
        /// </summary>
        public static void MedianFilter(int gridSize = 0, int blockSize = 0)
        {
            CudafyModule km = CudafyTranslator.Cudafy();

            GPGPU gpu = CudafyHost.GetDevice();
            gpu.LoadModule(km);

            byte[] devbytesA = gpu.Allocate<byte>(_videoMemorySize);
            byte[] devbytesB = gpu.Allocate<byte>(_videoMemorySize);

            byte[] devColor = gpu.Allocate(_color);

            int gridSize1 = (gridSize > 0)
                ? gridSize
                : Math.Min(15, (int) Math.Pow(_frameItemsCount, 0.333333333333));
            int blockSize1 = (blockSize > 0)
                ? blockSize
                : Math.Min(15, (int) Math.Pow(_frameItemsCount, 0.333333333333));

            int gridSize2 = (gridSize > 0)
                ? gridSize
                : Math.Min(15, (int) Math.Pow(_frameItemsCount*_itemSize, 0.333333333333));
            int blockSize2 = (blockSize > 0)
                ? blockSize
                : Math.Min(15, (int) Math.Pow(_frameItemsCount*_itemSize, 0.333333333333));

            int gridSize4 = (gridSize > 0)
                ? gridSize
                : Math.Min(15,
                    (int)
                        Math.Pow(
                            (_frameItemsCount*((1 << (_ceilingItemSize - _ceilingMiddleSize)) + _ceilingMiddleSize)),
                            0.333333333333));
            int blockSize4 = (blockSize > 0)
                ? blockSize
                : Math.Min(15,
                    (int)
                        Math.Pow(
                            (_frameItemsCount*((1 << (_ceilingItemSize - _ceilingMiddleSize)) + _ceilingMiddleSize)),
                            0.333333333333));

            // Цикл по цветам RGB - байтам
            // В видео памяти создаётся фрагмент изображения - фрейм с полями, 
            // который мы в цикле перемещаем по всему изображению
            // фрейм с полями копируем в видео память 
            // (два соседний фрейма без полей примыкают друг к другу и пересекаются полями)

            foreach (var pair in new Dictionary<byte[], byte[]> {{_r0, _r1}, {_g0, _g1}, {_b0, _b1}})
                for (int left = 0; left < (_width - _nh); left += _frameWidth - 2*_nh)
                    for (int top = 0; top < (_height - _nh); top += _frameHeight - 2*_nh)
                    {
                        int width = Math.Min(_frameWidth, _width - left);
                        int height = Math.Min(_frameHeight, _height - top);

                        int count = (width - 2*_nh)*(height - 2*_nh);

                        Debug.WriteLine("left:" + left + ",top:" + top + ",width:" + width + ",height:" + height +
                                        ",count:" + count);

                        // Копирование блока(фрейма) цветового слоя в видео память

                        for (int i = 0; i < width; i++)
                            for (int j = 0; j < height; j++)
                                _color[j*width + i] = pair.Key[(top + j)*_width + (left + i)];

                        gpu.CopyToDevice(_color, devColor);

                        // Формирование для каждой внутренней точки фрейма одномерного массива из _n*_n соседних точек
                        gpu.Launch(gridSize2, blockSize2).SelectColorBytes(devbytesA, devColor,
                            _itemSize, count,
                            width, height, _n, _nh);

                        // Выполнение чётно-нечётной сортировки параллельно для всех ранее созданных одномерных массивов

                        // Шаг 1 чётно-нечётной сортировки
                        // Выполнение сортировки слияниями
                        // На выходе отсортированные массивы размера до 1<<(_ceilingItemSize - _ceilingMiddleSize)
                        for (int i = 0; i < _ceilingItemSize - _ceilingMiddleSize; i++)
                        {
                            gpu.Launch(gridSize4, blockSize4)
                                .Merge(
                                    ((i & 1) == 0) ? devbytesA : devbytesB,
                                    ((i & 1) == 0) ? devbytesB : devbytesA,
                                    i, 0, _ceilingItemSize, _itemSize, count);
                        }

                        // Шаг 2 чётно-нечётной сортировки
                        // запускаем задачи сортировки данных в двух соседних блоках
                        // чередуя соседние блоки
                        for (int i = 0; i < (1 << _ceilingMiddleSize); i++)
                        {
                            gpu.Launch(gridSize4, blockSize4)
                                .Merge(
                                    ((i & 1) == ((_ceilingItemSize - _ceilingMiddleSize) & 1)) ? devbytesA : devbytesB,
                                    ((i & 1) == ((_ceilingItemSize - _ceilingMiddleSize) & 1)) ? devbytesB : devbytesA,
                                    _ceilingItemSize - _ceilingMiddleSize, i & 1,
                                    _ceilingItemSize, _itemSize, count);
                        }

                        // Выделение средних элементов в массивах и копирование их выходное изображение
                        gpu.Launch(gridSize1, blockSize1).SelectNhBytes(devColor,
                            (((1 << _ceilingMiddleSize) & 1) == ((_ceilingItemSize - _ceilingMiddleSize) & 1))
                                ? devbytesA
                                : devbytesB,
                            _nhIndex,
                            _itemSize, count,
                            width, height, _n, _nh);

                        gpu.CopyFromDevice(devColor, _color);

                        for (int i = _nh; i < (width - _nh); i++)
                            for (int j = _nh; j < (height - _nh); j++)
                                pair.Value[(top + j)*_width + (left + i)] = _color[j*width + i];
                    }

            // free the memory allocated on the GPU
            gpu.FreeAll();
        }

        /// <summary>
        ///     Выделение из набора одномерных массивов размера n*n точки по индексу index
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="color"></param>
        /// <param name="bytes"></param>
        /// <param name="index"></param>
        /// <param name="itemSize"></param>
        /// <param name="itemsCount"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="n"></param>
        /// <param name="nh"></param>
        [Cudafy]
        public static void SelectNhBytes(GThread thread,
            byte[] color, byte[] bytes, int index,
            int itemSize, int itemsCount,
            int width, int height, int n, int nh)
        {
            for (
                int tid = (thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x);
                tid < itemsCount;
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int y = (tid/(width - 2*nh)) + nh;
                int x = (tid%(width - 2*nh)) + nh;
                color[y*width + x] = bytes[tid*itemSize + index];
            }
        }

        /// <summary>
        ///     Создание для каждой внутренней точки фрейма одномерного массива из n*n соседних точек
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="bytes"></param>
        /// <param name="color"></param>
        /// <param name="itemSize"></param>
        /// <param name="itemsCount"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="n"></param>
        /// <param name="nh"></param>
        [Cudafy]
        public static void SelectColorBytes(GThread thread,
            byte[] bytes, byte[] color,
            int itemSize, int itemsCount,
            int width, int height, int n, int nh)
        {
            for (
                int tid = thread.blockDim.x*thread.blockIdx.x + thread.threadIdx.x;
                tid < (n*n*itemsCount);
                tid += thread.blockDim.x*thread.gridDim.x)
            {
                int blockId = tid/(n*n);
                int subPixelId = tid%(n*n);
                int dy = subPixelId/n;
                int dx = subPixelId%n;
                int y = (blockId)/(width - 2*nh) + dy;
                int x = (blockId)%(width - 2*nh) + dx;
                bytes[tid] = color[y*width + x];
            }
        }

        /// <summary>
        ///     Алгоритм слияния отсортированных подмассивов - выполняется параллельно для набора подмассивов
        ///     Используется на 1-ом и 2-ом шаге алгоритма чётно-нечётной сортировки
        /// </summary>
        /// <param name="thread"></param>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="i"></param>
        /// <param name="parity"></param>
        /// <param name="ceilingItemSize"></param>
        /// <param name="itemSize"></param>
        /// <param name="itemsCount"></param>
        [Cudafy]
        public static void Merge(GThread thread, byte[] a, byte[] b,
            int i, int parity,
            int ceilingItemSize, int itemSize, int itemsCount)
        {
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
                int n0 = index1 - index0;
                int n1 = index2 - index1;
                int total = index2 - index0;
                while (n0 > 0 && n1 > 0)
                {
                    if ((a[index0 + n0 - 1] > a[index1 + n1 - 1]))
                        b[index0 + --total] = a[index0 + --n0];
                    else
                        b[index0 + --total] = a[index1 + --n1];
                }
                while (n0 > 0) b[index0 + --total] = a[index0 + --n0];
                while (n1 > 0) b[index0 + --total] = a[index1 + --n1];
            }
        }
    }
}