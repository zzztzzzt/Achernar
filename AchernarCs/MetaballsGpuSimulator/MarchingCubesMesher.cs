using System.Runtime.CompilerServices;

namespace AchernarCs;

public class MarchingCubesMesher
{
    public const float Isolevel = 80.0f;

    private static readonly (int, int)[] EdgeVertexIndices = new[]
    {
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    };

    private readonly int _gridResolution;
    private readonly int _maxTriangles;
    private readonly float[] _gridAxis;
    private readonly float[] _vertexBuffer;
    private readonly float[] _normalBuffer;

    private readonly float[][] _sliceVertices;
    private readonly float[][] _sliceNormals;
    private readonly int[] _sliceCounts;

    private Blob[] _blobs = Array.Empty<Blob>();
    private const float FieldEpsilon = ScalarFieldGpuEvaluator.FieldEpsilon;
    private const float Subtract = ScalarFieldGpuEvaluator.Subtract;

    public MarchingCubesMesher(int gridResolution, int maxTriangles, float[] gridAxis)
    {
        _gridResolution = gridResolution;
        _maxTriangles = maxTriangles;
        _gridAxis = gridAxis;

        _vertexBuffer = new float[maxTriangles * 9];
        _normalBuffer = new float[maxTriangles * 9];

        int sliceCount = gridResolution - 1;
        int maxSliceFloats = (gridResolution - 1) * (gridResolution - 1) * 5 * 9;
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
    private int GridIndex(int x, int y, int z) => x + y * _gridResolution + z * _gridResolution * _gridResolution;

    public int Build(float[] fieldBuffer, Blob[] blobs)
    {
        _blobs = blobs;
        int sliceCount = _gridResolution - 1;

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

            for (int y = 0; y < _gridResolution - 1; y++)
            {
                float y0 = _gridAxis[y];
                float y1 = _gridAxis[y + 1];

                for (int x = 0; x < _gridResolution - 1; x++)
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

                    cubeValues[0] = fieldBuffer[q];
                    cubeValues[1] = fieldBuffer[q1];
                    cubeValues[2] = fieldBuffer[q1y];
                    cubeValues[3] = fieldBuffer[qy];
                    cubeValues[4] = fieldBuffer[qz];
                    cubeValues[5] = fieldBuffer[q1z];
                    cubeValues[6] = fieldBuffer[q1yz];
                    cubeValues[7] = fieldBuffer[qyz];

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

    public float[] VertexBuffer => _vertexBuffer;
    public float[] NormalBuffer => _normalBuffer;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private (float nx, float ny, float nz) SampleGradient(float x, float y, float z)
    {
        float gx = 0.0f, gy = 0.0f, gz = 0.0f;
        for (int i = 0; i < _blobs.Length; i++)
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
}
