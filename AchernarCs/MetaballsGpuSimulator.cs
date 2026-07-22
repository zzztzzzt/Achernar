using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ComputeSharp;

namespace AchernarCs;

public class Blob
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }
    public float Vx { get; set; }
    public float Vy { get; set; }
    public float Vz { get; set; }
    public float Size { get; set; }

    public Blob(float x, float y, float z, float vx, float vy, float vz, float size)
    {
        X = x; Y = y; Z = z;
        Vx = vx; Vy = vy; Vz = vz;
        Size = size;
    }
}

public class MetaballsGpuSimulator
{
    public const int BallCount = 12;
    public const float FrameInterval = 1.0f / 120.0f;
    public const float SpeedLimit = 0.008f;
    public const int GridResolution = 132;
    public const int GridSize = GridResolution * GridResolution * GridResolution;
    public const int CubeCount = (GridResolution - 1) * (GridResolution - 1) * (GridResolution - 1);
    public const int MaxTriangles = CubeCount * 5;
    public const float Isolevel = 80.0f;
    public const float Subtract = 8.0f;
    public const float FieldEpsilon = 1.0e-6f;

    private readonly Blob[] _blobs;
    private readonly float[] _gridAxis;
    private readonly float[] _fieldBuffer;
    private readonly float[] _vertexBuffer;
    private readonly float[] _normalBuffer;
    private readonly float[] _payloadBuffer;

    private readonly Float4[] _blobGpuData;
    private readonly ReadWriteBuffer<float> _dField;
    private readonly ReadOnlyBuffer<float> _dAxis;
    private readonly ReadOnlyBuffer<Float4> _dBlobs;

    private readonly float[][] _sliceVertices;
    private readonly float[][] _sliceNormals;
    private readonly int[] _sliceCounts;

    private static readonly (int, int)[] EdgeVertexIndices = new[]
    {
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    };

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

        _fieldBuffer = new float[GridSize];
        _vertexBuffer = new float[MaxTriangles * 9];
        _normalBuffer = new float[MaxTriangles * 9];
        _payloadBuffer = new float[1 + MaxTriangles * 18];

        _blobGpuData = new Float4[BallCount];
        _dField = GraphicsDevice.GetDefault().AllocateReadWriteBuffer<float>(GridSize);
        _dAxis = GraphicsDevice.GetDefault().AllocateReadOnlyBuffer(_gridAxis);
        _dBlobs = GraphicsDevice.GetDefault().AllocateReadOnlyBuffer<Float4>(BallCount);

        int sliceCount = GridResolution - 1;
        int maxSliceFloats = (GridResolution - 1) * (GridResolution - 1) * 5 * 9;
        _sliceVertices = new float[sliceCount][];
        _sliceNormals = new float[sliceCount][];
        _sliceCounts = new int[sliceCount];
        for (int i = 0; i < sliceCount; i++)
        {
            _sliceVertices[i] = new float[maxSliceFloats];
            _sliceNormals[i] = new float[maxSliceFloats];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GridIndex(int x, int y, int z) => x + y * GridResolution + z * GridResolution * GridResolution;

    public ArraySegment<byte> GenerateNextFramePayload(double currentTime)
    {
        UpdatePhysics(currentTime);
        ComputeFieldGpu();
        int vertexFloatCount = BuildMeshParallel();
        return BuildPayload(vertexFloatCount);
    }

    private void UpdatePhysics(double currentTime)
    {
        bool isGravityOn = (currentTime % 10) < 5;

        for (int i = 0; i < BallCount; i++)
        {
            var bi = _blobs[i];
            bi.X += bi.Vx;
            bi.Y += bi.Vy;
            bi.Z += bi.Vz;

            for (int j = 0; j < BallCount; j++)
            {
                if (i == j) continue;
                var bj = _blobs[j];
                float dx = bj.X - bi.X;
                float dy = bj.Y - bi.Y;
                float dz = bj.Z - bi.Z;
                float distSq = dx * dx + dy * dy + dz * dz + 0.01f;

                if (isGravityOn)
                {
                    float force = 0.000008f / distSq;
                    bi.Vx += dx * force;
                    bi.Vy += dy * force;
                    bi.Vz += dz * force;
                }
                else
                {
                    float pushForce = 0.00002f / distSq;
                    bi.Vx -= dx * pushForce;
                    bi.Vy -= dy * pushForce;
                    bi.Vz -= dz * pushForce;
                }
            }

            const float margin = 0.15f;
            if (bi.X < margin) bi.Vx += 0.001f;
            else if (bi.X > 1.0f - margin) bi.Vx -= 0.001f;

            if (bi.Y < margin) bi.Vy += 0.001f;
            else if (bi.Y > 1.0f - margin) bi.Vy -= 0.001f;

            if (bi.Z < margin) bi.Vz += 0.001f;
            else if (bi.Z > 1.0f - margin) bi.Vz -= 0.001f;

            bi.Vx = Math.Clamp(bi.Vx * 0.98f, -SpeedLimit, SpeedLimit);
            bi.Vy = Math.Clamp(bi.Vy * 0.98f, -SpeedLimit, SpeedLimit);
            bi.Vz = Math.Clamp(bi.Vz * 0.98f, -SpeedLimit, SpeedLimit);

            _blobGpuData[i] = new Float4(bi.X, bi.Y, bi.Z, bi.Size);
        }
    }

    private void ComputeFieldGpu()
    {
        _dBlobs.CopyFrom(_blobGpuData);

        var shader = new FieldKernel(
            _dField,
            _dAxis,
            _dBlobs,
            BallCount,
            GridResolution,
            FieldEpsilon,
            Subtract
        );

        GraphicsDevice.GetDefault().For(GridResolution, GridResolution, GridResolution, shader);
        _dField.CopyTo(_fieldBuffer);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private (float nx, float ny, float nz) SampleGradient(float x, float y, float z)
    {
        float gx = 0.0f, gy = 0.0f, gz = 0.0f;
        for (int i = 0; i < BallCount; i++)
        {
            var blob = _blobs[i];
            float dx = x - blob.X;
            float dy = y - blob.Y;
            float dz = z - blob.Z;
            float distSq = FieldEpsilon + dx * dx + dy * dy + dz * dz;
            float contrib = blob.Size / distSq - Subtract;
            if (contrib > 0.0f)
            {
                float scale = -2.0f * blob.Size / (distSq * distSq);
                gx += dx * scale;
                gy += dy * scale;
                gz += dz * scale;
            }
        }

        float len = MathF.Sqrt(gx * gx + gy * gy + gz * gz);
        if (len > 1.0e-6f)
        {
            float invLen = 1.0f / len;
            return (gx * invLen, gy * invLen, gz * invLen);
        }
        return (0.0f, 1.0f, 0.0f);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float x, float y, float z) InterpolateVertex(
        float ax, float ay, float az, float av,
        float bx, float by, float bz, float bv)
    {
        float delta = bv - av;
        float t = MathF.Abs(delta) < 1.0e-6f ? 0.5f : Math.Clamp((Isolevel - av) / delta, 0.0f, 1.0f);
        return (
            ax + (bx - ax) * t,
            ay + (by - ay) * t,
            az + (bz - az) * t
        );
    }

    private int BuildMeshParallel()
    {
        int sliceCount = GridResolution - 1;

        Parallel.For(0, sliceCount, z =>
        {
            float z0 = _gridAxis[z];
            float z1 = _gridAxis[z + 1];

            var localVerts = _sliceVertices[z];
            var localNorms = _sliceNormals[z];
            int offset = 0;

            Span<float> edgeX = stackalloc float[12];
            Span<float> edgeY = stackalloc float[12];
            Span<float> edgeZ = stackalloc float[12];
            Span<float> edgeNx = stackalloc float[12];
            Span<float> edgeNy = stackalloc float[12];
            Span<float> edgeNz = stackalloc float[12];
            Span<float> cubeValues = stackalloc float[8];
            Span<float> cubeX = stackalloc float[8];
            Span<float> cubeY = stackalloc float[8];
            Span<float> cubeZ = stackalloc float[8];

            for (int y = 0; y < GridResolution - 1; y++)
            {
                float y0 = _gridAxis[y];
                float y1 = _gridAxis[y + 1];

                for (int x = 0; x < GridResolution - 1; x++)
                {
                    float x0 = _gridAxis[x];
                    float x1 = _gridAxis[x + 1];

                    int q = GridIndex(x, y, z);
                    int q1 = GridIndex(x + 1, y, z);
                    int qy = GridIndex(x, y + 1, z);
                    int q1y = GridIndex(x + 1, y + 1, z);
                    int qz = GridIndex(x, y, z + 1);
                    int q1z = GridIndex(x + 1, y, z + 1);
                    int qyz = GridIndex(x, y + 1, z + 1);
                    int q1yz = GridIndex(x + 1, y + 1, z + 1);

                    cubeValues[0] = _fieldBuffer[q];
                    cubeValues[1] = _fieldBuffer[q1];
                    cubeValues[2] = _fieldBuffer[q1y];
                    cubeValues[3] = _fieldBuffer[qy];
                    cubeValues[4] = _fieldBuffer[qz];
                    cubeValues[5] = _fieldBuffer[q1z];
                    cubeValues[6] = _fieldBuffer[q1yz];
                    cubeValues[7] = _fieldBuffer[qyz];

                    int cubeIndex = 0;
                    for (int i = 0; i < 8; i++)
                    {
                        if (cubeValues[i] < Isolevel)
                        {
                            cubeIndex |= 1 << i;
                        }
                    }

                    int edgeMask = MarchingCubesTables.EdgeTable[cubeIndex];
                    if (edgeMask == 0) continue;

                    cubeX[0] = x0; cubeX[1] = x1; cubeX[2] = x1; cubeX[3] = x0;
                    cubeX[4] = x0; cubeX[5] = x1; cubeX[6] = x1; cubeX[7] = x0;

                    cubeY[0] = y0; cubeY[1] = y0; cubeY[2] = y1; cubeY[3] = y1;
                    cubeY[4] = y0; cubeY[5] = y0; cubeY[6] = y1; cubeY[7] = y1;

                    cubeZ[0] = z0; cubeZ[1] = z0; cubeZ[2] = z0; cubeZ[3] = z0;
                    cubeZ[4] = z1; cubeZ[5] = z1; cubeZ[6] = z1; cubeZ[7] = z1;

                    for (int edge = 0; edge < 12; edge++)
                    {
                        if ((edgeMask & (1 << edge)) == 0) continue;

                        var (ia, ib) = EdgeVertexIndices[edge];
                        var (vx, vy, vz) = InterpolateVertex(
                            cubeX[ia], cubeY[ia], cubeZ[ia], cubeValues[ia],
                            cubeX[ib], cubeY[ib], cubeZ[ib], cubeValues[ib]
                        );

                        edgeX[edge] = vx;
                        edgeY[edge] = vy;
                        edgeZ[edge] = vz;

                        var (nx, ny, nz) = SampleGradient(vx, vy, vz);
                        edgeNx[edge] = nx;
                        edgeNy[edge] = ny;
                        edgeNz[edge] = nz;
                    }

                    int triTableBase = cubeIndex * 16;
                    while (MarchingCubesTables.TriTable[triTableBase] != -1)
                    {
                        if (offset + 8 >= localVerts.Length) break;
                        int e1 = MarchingCubesTables.TriTable[triTableBase];
                        int e2 = MarchingCubesTables.TriTable[triTableBase + 1];
                        int e3 = MarchingCubesTables.TriTable[triTableBase + 2];

                        // Vertices (scaled to -1..1)
                        localVerts[offset]     = edgeX[e1] * 2.0f - 1.0f;
                        localVerts[offset + 1] = edgeY[e1] * 2.0f - 1.0f;
                        localVerts[offset + 2] = edgeZ[e1] * 2.0f - 1.0f;
                        localVerts[offset + 3] = edgeX[e2] * 2.0f - 1.0f;
                        localVerts[offset + 4] = edgeY[e2] * 2.0f - 1.0f;
                        localVerts[offset + 5] = edgeZ[e2] * 2.0f - 1.0f;
                        localVerts[offset + 6] = edgeX[e3] * 2.0f - 1.0f;
                        localVerts[offset + 7] = edgeY[e3] * 2.0f - 1.0f;
                        localVerts[offset + 8] = edgeZ[e3] * 2.0f - 1.0f;

                        // Normals
                        localNorms[offset]     = edgeNx[e1];
                        localNorms[offset + 1] = edgeNy[e1];
                        localNorms[offset + 2] = edgeNz[e1];
                        localNorms[offset + 3] = edgeNx[e2];
                        localNorms[offset + 4] = edgeNy[e2];
                        localNorms[offset + 5] = edgeNz[e2];
                        localNorms[offset + 6] = edgeNx[e3];
                        localNorms[offset + 7] = edgeNy[e3];
                        localNorms[offset + 8] = edgeNz[e3];

                        offset += 9;
                        triTableBase += 3;
                    }
                }
            }

            _sliceCounts[z] = offset;
        });

        int globalOffset = 0;
        for (int z = 0; z < sliceCount; z++)
        {
            int count = _sliceCounts[z];
            if (count == 0) continue;
            Array.Copy(_sliceVertices[z], 0, _vertexBuffer, globalOffset, count);
            Array.Copy(_sliceNormals[z], 0, _normalBuffer, globalOffset, count);
            globalOffset += count;
        }

        return globalOffset;
    }

    private ArraySegment<byte> BuildPayload(int vertexFloatCount)
    {
        _payloadBuffer[0] = (float)vertexFloatCount;
        Array.Copy(_vertexBuffer, 0, _payloadBuffer, 1, vertexFloatCount);
        Array.Copy(_normalBuffer, 0, _payloadBuffer, 1 + vertexFloatCount, vertexFloatCount);

        int totalFloats = 1 + vertexFloatCount * 2;
        int totalBytes = totalFloats * sizeof(float);

        var byteSpan = MemoryMarshal.AsBytes(_payloadBuffer.AsSpan(0, totalFloats));
        return new ArraySegment<byte>(byteSpan.ToArray());
    }
}
