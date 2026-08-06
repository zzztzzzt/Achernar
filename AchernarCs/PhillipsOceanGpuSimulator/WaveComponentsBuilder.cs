namespace AchernarCs;

/// <summary>
/// CPU-side Phillips ocean initialization — mirrors PhillipsOcean.jl::build_components!.
/// Grid coordinates are now computed on-the-fly inside the GPU kernel, so this class
/// only produces the five small per-component arrays (each 128 floats).
/// </summary>
public static class WaveComponentsBuilder
{
    public const int Resolution = 512;
    public const int ComponentCount = 128;
    public const float DomainSize = 36.0f;
    public const float Gravity = 9.81f;
    public const float WindSpeed = 14.0f;
    public const float AmplitudeScale = 0.08f;

    public static readonly (float X, float Y) WindDirection = (0.92f, 0.38f);

    public static int PixelCount => Resolution * Resolution;

    public static (float X, float Y) Normalize2D(float x, float y)
    {
        float len = MathF.Sqrt(x * x + y * y);
        return len < 1e-6f ? (0f, 0f) : (x / len, y / len);
    }

    /// <summary>
    /// Mirrors PhillipsOcean.jl::phillips_spectrum exactly.
    /// </summary>
    public static float PhillipsSpectrum(float kx, float ky, float windX, float windY)
    {
        float k2 = kx * kx + ky * ky;
        if (k2 == 0f) return 0f;

        var (kDirX, kDirY) = Normalize2D(kx, ky);
        float alignment = MathF.Max(kDirX * windX + kDirY * windY, 0f);

        float L = (WindSpeed * WindSpeed) / Gravity;
        float smallWaveLength = L * 0.0015f;
        float smallWaveDamping = smallWaveLength * smallWaveLength;

        // Keep this identical to PhillipsOcean.jl::phillips_spectrum:
        // directional alignment is raised to the fourth power and short
        // wavelengths are damped to avoid excessive high-frequency energy.
        return MathF.Exp(-1f / (k2 * L * L))
            / (k2 * k2)
            * MathF.Pow(alignment, 4f)
            * MathF.Exp(-k2 * smallWaveDamping);
    }

    public static WaveComponentData Build()
    {
        var kx = new float[ComponentCount];
        var ky = new float[ComponentCount];
        var omega = new float[ComponentCount];
        var amp = new float[ComponentCount];
        var initialPhases = new float[ComponentCount];

        var rng = new AxisRng(42);
        var (windX, windY) = Normalize2D(WindDirection.X, WindDirection.Y);
        float baseAngle = MathF.Atan2(windY, windX);
        int pairCount = ComponentCount / 2;
        int idx = 0;

        for (int i = 0; i < pairCount; i++)
        {
            float band = pairCount <= 1 ? 0f : (float)i / (pairCount - 1);
            float wavelength = 1.2f + 9.0f * band * band;
            float k = MathF.Tau / wavelength;
            float angle = baseAngle + rng.NextStandardNormal() * 1.05f * (0.2f + 0.8f * band);

            foreach (var (directionMultiplier, amplitudeScale) in new (float DirectionMultiplier, float AmplitudeScale)[] { (1.0f, 1.0f), (-1.0f, 0.45f) })
            {
                float componentKx = directionMultiplier * MathF.Cos(angle) * k;
                float componentKy = directionMultiplier * MathF.Sin(angle) * k;
                float spectrumValue = PhillipsSpectrum(componentKx, componentKy, windX, windY);

                amp[idx] = AmplitudeScale * amplitudeScale * MathF.Sqrt(Math.Max(spectrumValue, 0f)) * (0.35f + 0.65f * (1f - band));
                initialPhases[idx] = rng.NextF32() * MathF.Tau;
                omega[idx] = MathF.Sqrt(Gravity * k);
                kx[idx] = componentKx;
                ky[idx] = componentKy;
                idx++;
            }
        }

        return new WaveComponentData(kx, ky, omega, amp, initialPhases);
    }
}

public readonly record struct WaveComponentData(
    float[] Kx,
    float[] Ky,
    float[] Omega,
    float[] Amp,
    float[] Phase0);
