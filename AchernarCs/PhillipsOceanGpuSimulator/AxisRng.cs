namespace AchernarCs;

/// <summary>
/// Minimal xorshift64 RNG — matches Julia/Rust AxisRng for identical wave seeds.
/// </summary>
internal sealed class AxisRng
{
    private ulong _state;

    public AxisRng(long seed)
    {
        _state = (ulong)Math.Max(seed, 1);
    }

    private uint NextU32()
    {
        ulong x = _state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        _state = x;
        return (uint)((x * 0x2545f4914f6cdd1dUL) >> 32);
    }

    public float NextF32() => (NextU32() >> 8) * (1f / (1 << 24));

    public float NextStandardNormal()
    {
        float u1 = Math.Max(NextF32(), float.Epsilon);
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * NextF32());
    }
}
