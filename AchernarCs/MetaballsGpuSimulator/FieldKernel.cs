using ComputeSharp;

namespace AchernarCs;

[ThreadGroupSize(8, 8, 4)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct FieldKernel : IComputeShader
{
    public readonly ReadWriteBuffer<float> Field;
    public readonly ReadOnlyBuffer<float> Axis;
    public readonly ReadOnlyBuffer<float4> Blobs; // (x, y, z, size)
    public readonly int BallCount;
    public readonly int Resolution;
    public readonly float Epsilon;
    public readonly float Subtract;

    public FieldKernel(
        ReadWriteBuffer<float> field,
        ReadOnlyBuffer<float> axis,
        ReadOnlyBuffer<float4> blobs,
        int ballCount,
        int resolution,
        float epsilon,
        float subtract)
    {
        Field = field;
        Axis = axis;
        Blobs = blobs;
        BallCount = ballCount;
        Resolution = resolution;
        Epsilon = epsilon;
        Subtract = subtract;
    }

    public void Execute()
    {
        int x = ThreadIds.X;
        int y = ThreadIds.Y;
        int z = ThreadIds.Z;

        if (x >= Resolution || y >= Resolution || z >= Resolution) return;

        float px = Axis[x];
        float py = Axis[y];
        float pz = Axis[z];
        float value = 0.0f;

        for (int i = 0; i < BallCount; i++)
        {
            float4 blob = Blobs[i];
            float dx = px - blob.X;
            float dy = py - blob.Y;
            float dz = pz - blob.Z;
            float contrib = blob.W / (Epsilon + dx * dx + dy * dy + dz * dz) - Subtract;
            if (contrib > 0.0f)
            {
                value += contrib;
            }
        }

        int index = x + y * Resolution + z * Resolution * Resolution;
        Field[index] = value;
    }
}
