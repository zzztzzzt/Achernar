using ComputeSharp;

namespace AchernarCs;

public class ScalarFieldGpuEvaluator : IDisposable
{
    public const float FieldEpsilon = 1.0e-6f;
    public const float Subtract = 8.0f;

    private readonly GraphicsDevice _device;
    private readonly int _gridResolution;
    private readonly int _blobCount;
    private readonly float[] _fieldBuffer;

    private readonly ReadWriteBuffer<float> _dField;
    private readonly ReadOnlyBuffer<float> _dAxis;
    private readonly ReadOnlyBuffer<Float4> _dBlobs;

    private bool _disposed;

    public ScalarFieldGpuEvaluator(int gridResolution, int blobCount, float[] gridAxis)
    {
        _device = GraphicsDevice.GetDefault();

        _gridResolution = gridResolution;
        _blobCount = blobCount;

        int gridSize = gridResolution * gridResolution * gridResolution;
        _fieldBuffer = new float[gridSize];

        _dField = _device.AllocateReadWriteBuffer<float>(gridSize);
        _dAxis = _device.AllocateReadOnlyBuffer(gridAxis);
        _dBlobs = _device.AllocateReadOnlyBuffer<Float4>(blobCount);
    }

    public float[] Compute(Float4[] blobGpuData)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (blobGpuData.Length != _blobCount)
        {
            throw new ArgumentException(
                $"Expected {_blobCount} blobs, got {blobGpuData.Length}.",
                nameof(blobGpuData));
        }

        _dBlobs.CopyFrom(blobGpuData);

        var shader = new FieldKernel(
            _dField,
            _dAxis,
            _dBlobs,
            _blobCount,
            _gridResolution,
            FieldEpsilon,
            Subtract
        );

        _device.For(_gridResolution, _gridResolution, _gridResolution, shader);
        _dField.CopyTo(_fieldBuffer);

        return _fieldBuffer;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _dField.Dispose();
        _dAxis.Dispose();
        _dBlobs.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
