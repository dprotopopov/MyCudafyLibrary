
// MyCudafy.CudafyFilter
extern "C" __global__ void SelectNhBytes( unsigned char* color, int colorLen0,  unsigned char* bytes, int bytesLen0, int index, int itemSize, int itemsCount, int width, int height, int n, int nh);
// MyCudafy.CudafyFilter
extern "C" __global__ void SelectColorBytes( unsigned char* bytes, int bytesLen0,  unsigned char* color, int colorLen0, int itemSize, int itemsCount, int width, int height, int n, int nh);
// MyCudafy.CudafyFilter
extern "C" __global__ void Merge( unsigned char* a, int aLen0,  unsigned char* b, int bLen0, int i, int parity, int ceilingItemSize, int itemSize, int itemsCount);

// MyCudafy.CudafyFilter
__constant__ unsigned char _color[144];
#define _colorLen0 144
// MyCudafy.CudafyFilter
extern "C" __global__ void SelectNhBytes( unsigned char* color, int colorLen0,  unsigned char* bytes, int bytesLen0, int index, int itemSize, int itemsCount, int width, int height, int n, int nh)
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < itemsCount; i += blockDim.x * gridDim.x)
	{
		int num = i / (width - 2 * nh) + nh;
		int num2 = i % (width - 2 * nh) + nh;
		color[(num * width + num2)] = bytes[(i * itemSize + index)];
	}
}
// MyCudafy.CudafyFilter
extern "C" __global__ void SelectColorBytes( unsigned char* bytes, int bytesLen0,  unsigned char* color, int colorLen0, int itemSize, int itemsCount, int width, int height, int n, int nh)
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n * n * itemsCount; i += blockDim.x * gridDim.x)
	{
		int num = i / (n * n);
		int num2 = i % (n * n);
		int num3 = num2 / n;
		int num4 = num2 % n;
		int num5 = num / (width - 2 * nh) + num3;
		int num6 = num % (width - 2 * nh) + num4;
		bytes[(i)] = color[(num5 * width + num6)];
	}
}
// MyCudafy.CudafyFilter
extern "C" __global__ void Merge( unsigned char* a, int aLen0,  unsigned char* b, int bLen0, int i, int parity, int ceilingItemSize, int itemSize, int itemsCount)
{
	for (int j = 0; j < itemsCount * ((1 << (ceilingItemSize - i - 1 & 31)) + parity); j++)
	{
		int num = j / ((1 << (ceilingItemSize - i - 1 & 31)) + parity);
		int num2 = j % ((1 << (ceilingItemSize - i - 1 & 31)) + parity);
		int num3 = num * itemSize + (num2 << (i + 1 & 31)) - (parity << (i & 31));
		int num4 = min(max(num * itemSize, num3), (num + 1) * itemSize);
		int num5 = min(max(num * itemSize, num3 + (1 << (i & 31))), (num + 1) * itemSize);
		int num6 = min(max(num * itemSize, num3 + (2 << (i & 31))), (num + 1) * itemSize);
		int k = num5 - num4;
		int l = num6 - num5;
		int num7 = num6 - num4;
		while (k > 0 && l > 0)
		{
			if (a[(num4 + k - 1)] > a[(num5 + l - 1)])
			{
				b[(num4 + --num7)] = a[(num4 + --k)];
			}
			else
			{
				b[(num4 + --num7)] = a[(num5 + --l)];
			}
		}
		while (k > 0)
		{
			b[(num4 + --num7)] = a[(num4 + --k)];
		}
		while (l > 0)
		{
			b[(num4 + --num7)] = a[(num5 + --l)];
		}
	}
}
