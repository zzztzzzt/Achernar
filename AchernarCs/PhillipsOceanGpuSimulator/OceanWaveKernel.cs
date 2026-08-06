using ComputeSharp;

namespace AchernarCs;

[ThreadGroupSize(256, 1, 1)]
[GeneratedComputeShaderDescriptor]
public readonly partial struct OceanWaveKernel : IComputeShader
{
    public readonly ReadWriteBuffer<float> Frame;
    public readonly ReadOnlyBuffer<float> Kx;
    public readonly ReadOnlyBuffer<float> Ky;
    public readonly ReadOnlyBuffer<float> Omega;
    public readonly ReadOnlyBuffer<float> Amp;
    public readonly ReadOnlyBuffer<float> Phase0;
    public readonly int PixelCount;
    public readonly int ComponentCount;
    public readonly int Resolution;
    public readonly float DomainSize;
    public readonly float Time;

    public OceanWaveKernel(
        ReadWriteBuffer<float> frame,
        ReadOnlyBuffer<float> kx,
        ReadOnlyBuffer<float> ky,
        ReadOnlyBuffer<float> omega,
        ReadOnlyBuffer<float> amp,
        ReadOnlyBuffer<float> phase0,
        int pixelCount,
        int componentCount,
        int resolution,
        float domainSize,
        float time)
    {
        Frame = frame;
        Kx = kx;
        Ky = ky;
        Omega = omega;
        Amp = amp;
        Phase0 = phase0;
        PixelCount = pixelCount;
        ComponentCount = componentCount;
        Resolution = resolution;
        DomainSize = domainSize;
        Time = time;
    }

    public void Execute()
    {
        int pixelIndex = ThreadIds.X;
        if (pixelIndex >= PixelCount) return;

        // Compute world-space (X, Y) on the fly — mirrors the WGSL shader in PhillipsOceanAX.jl.
        // ix varies fastest (x-axis), iy varies slowest (y-axis), matching Julia column-major layout.
        int ix = pixelIndex % Resolution;
        int iy = pixelIndex / Resolution;
        float worldX = ((ix / (float)(Resolution - 1)) - 0.5f) * DomainSize;
        float worldY = ((iy / (float)(Resolution - 1)) - 0.5f) * DomainSize;

        float waveHeight = 0.0f;
        for (int j = 0; j < ComponentCount; j++)
        {
            float phase = Kx[j] * worldX + Ky[j] * worldY - Omega[j] * Time + Phase0[j];
            waveHeight += Amp[j] * Hlsl.Cos(phase);
        }

        Frame[pixelIndex] = waveHeight;
    }
}
