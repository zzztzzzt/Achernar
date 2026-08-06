namespace AchernarCs;

public class PhillipsOceanGpuSimulator : IDisposable
{
    public const int Resolution = WaveComponentsBuilder.Resolution;
    public const float FrameInterval = 1.0f / 60.0f;

    private readonly OceanWaveGpuEvaluator _evaluator;
    private readonly OceanFramePayloadBuilder _payloadBuilder;

    private bool _disposed;

    public PhillipsOceanGpuSimulator()
    {
        var components = WaveComponentsBuilder.Build();
        _evaluator = new OceanWaveGpuEvaluator(components);
        _payloadBuilder = new OceanFramePayloadBuilder(WaveComponentsBuilder.PixelCount);
    }

    public ArraySegment<byte> GenerateNextFramePayload(double currentTime)
    {
        var frameBuffer = _evaluator.Compute((float)currentTime);
        return _payloadBuilder.Build(frameBuffer);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _evaluator.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
