using ComputeSharp;

namespace AchernarCs;

public class OceanWaveGpuEvaluator : IDisposable
{
    private const int WorkgroupSize = 256;

    private readonly GraphicsDevice _device;
    private readonly int _pixelCount;
    private readonly int _componentCount;
    private readonly float[] _frameBuffer;

    private readonly ReadWriteBuffer<float> _deviceFrameBuffer;
    private readonly ReadOnlyBuffer<float> _deviceWaveNumberX;
    private readonly ReadOnlyBuffer<float> _deviceWaveNumberY;
    private readonly ReadOnlyBuffer<float> _deviceAngularFrequencies;
    private readonly ReadOnlyBuffer<float> _deviceWaveAmplitudes;
    private readonly ReadOnlyBuffer<float> _deviceInitialPhases;

    private bool _disposed;

    public OceanWaveGpuEvaluator(WaveComponentData components)
    {
        _device = GraphicsDevice.GetDefault();
        _pixelCount = WaveComponentsBuilder.PixelCount;
        _componentCount = WaveComponentsBuilder.ComponentCount;
        _frameBuffer = new float[_pixelCount];

        _deviceFrameBuffer = _device.AllocateReadWriteBuffer<float>(_pixelCount);
        _deviceWaveNumberX = _device.AllocateReadOnlyBuffer(components.Kx);
        _deviceWaveNumberY = _device.AllocateReadOnlyBuffer(components.Ky);
        _deviceAngularFrequencies = _device.AllocateReadOnlyBuffer(components.Omega);
        _deviceWaveAmplitudes = _device.AllocateReadOnlyBuffer(components.Amp);
        _deviceInitialPhases = _device.AllocateReadOnlyBuffer(components.Phase0);
    }

    public float[] Compute(float time)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var shader = new OceanWaveKernel(
            _deviceFrameBuffer,
            _deviceWaveNumberX,
            _deviceWaveNumberY,
            _deviceAngularFrequencies,
            _deviceWaveAmplitudes,
            _deviceInitialPhases,
            _pixelCount,
            _componentCount,
            WaveComponentsBuilder.Resolution,
            WaveComponentsBuilder.DomainSize,
            time);

        // ComputeSharp's For() takes the total number of threads to dispatch,
        // not the number of thread groups. The [ThreadGroupSize] attribute
        // controls how those threads are grouped on the GPU.
        _device.For(_pixelCount, 1, 1, shader);
        _deviceFrameBuffer.CopyTo(_frameBuffer);

        return _frameBuffer;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _deviceFrameBuffer.Dispose();
        _deviceWaveNumberX.Dispose();
        _deviceWaveNumberY.Dispose();
        _deviceAngularFrequencies.Dispose();
        _deviceWaveAmplitudes.Dispose();
        _deviceInitialPhases.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
