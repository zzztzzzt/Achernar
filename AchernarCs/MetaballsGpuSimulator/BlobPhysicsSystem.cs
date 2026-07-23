using ComputeSharp;

namespace AchernarCs;

public class BlobPhysicsSystem
{
    public const float SpeedLimit = 0.008f;

    private readonly Float4[] _gpuData;

    public BlobPhysicsSystem(int blobCount)
    {
        _gpuData = new Float4[blobCount];
    }

    public Float4[] Update(Blob[] blobs, double currentTime)
    {
        bool isGravityOn = (currentTime % 10) < 5;
        int count = blobs.Length;

        for (int i = 0; i < count; i++)
        {
            var bi = blobs[i];
            bi.X += bi.Vx;
            bi.Y += bi.Vy;
            bi.Z += bi.Vz;

            for (int j = 0; j < count; j++)
            {
                if (i == j) continue;
                var bj = blobs[j];
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

            _gpuData[i] = new Float4(bi.X, bi.Y, bi.Z, bi.Size);
        }

        return _gpuData;
    }
}
