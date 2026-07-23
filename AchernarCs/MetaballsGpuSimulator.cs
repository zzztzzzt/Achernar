namespace AchernarCs;

public class MetaballsGpuSimulator : IDisposable
{
    public const int BallCount = 12;
    public const float FrameInterval = 1.0f / 120.0f;
    public const int GridResolution = 132;
    public const int GridSize = GridResolution * GridResolution * GridResolution;
    public const int CubeCount = (GridResolution - 1) * (GridResolution - 1) * (GridResolution - 1);
    public const int MaxTriangles = CubeCount * 5;

    private readonly Blob[] _blobs;
    private readonly float[] _gridAxis;

    private readonly BlobPhysicsSystem _physics;
    private readonly ScalarFieldGpuEvaluator _fieldEvaluator;
    private readonly MarchingCubesMesher _mesher;
    private readonly FramePayloadBuilder _payloadBuilder;

    private bool _disposed;

    public MetaballsGpuSimulator()
    {
        var rand = new Random();
        _blobs = new Blob[BallCount];
        for (int i = 0; i < BallCount; i++)
        {
            _blobs[i] = new Blob(
                (float)rand.NextDouble() * 0.6f + 0.2f,
                (float)rand.NextDouble() * 0.6f + 0.2f,
                (float)rand.NextDouble() * 0.6f + 0.2f,
                ((float)rand.NextDouble() - 0.5f) * 0.005f,
                ((float)rand.NextDouble() - 0.5f) * 0.005f,
                ((float)rand.NextDouble() - 0.5f) * 0.005f,
                0.45f + (float)rand.NextDouble() * 0.1f
            );
        }

        _gridAxis = new float[GridResolution];
        for (int i = 0; i < GridResolution; i++)
        {
            _gridAxis[i] = (float)i / (GridResolution - 1);
        }

        _physics = new BlobPhysicsSystem(BallCount);
        _fieldEvaluator = new ScalarFieldGpuEvaluator(GridResolution, BallCount, _gridAxis);
        _mesher = new MarchingCubesMesher(GridResolution, MaxTriangles, _gridAxis);
        _payloadBuilder = new FramePayloadBuilder(MaxTriangles);
    }

    public ArraySegment<byte> GenerateNextFramePayload(double currentTime)
    {
        var blobGpuData = _physics.Update(_blobs, currentTime);
        var fieldBuffer = _fieldEvaluator.Compute(blobGpuData);
        int vertexFloatCount = _mesher.Build(fieldBuffer, _blobs);
        return _payloadBuilder.Build(_mesher.VertexBuffer, _mesher.NormalBuffer, vertexFloatCount);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _fieldEvaluator.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}