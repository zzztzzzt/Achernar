// Manually Validated by zzztzzzt-SakuraAxis 2026-07-29

using System.Runtime.InteropServices;

namespace AchernarCs;

/// <summary>
/// Warning : this class reuses an internal buffer to avoid per-frame allocations.
/// The ArraySegment returned by Build() points into that shared buffer, so the caller must finish consuming it ( e.g. await SendAsync to completion ) before
/// calling Build() again — otherwise the next frame's write will race with data that hasn't been sent yet.
/// If async / fire-and-forget sending is ever needed, switch to double-buffering or allocate a fresh array per call instead.
/// </summary>
public class FramePayloadBuilder
{
    private readonly float[] _payloadBuffer;
    private readonly byte[] _payloadByteBuffer;

    public FramePayloadBuilder(int maxTriangles)
    {
        _payloadBuffer = new float[1 + maxTriangles * 18];
        _payloadByteBuffer = new byte[_payloadBuffer.Length * sizeof(float)];
    }

    /// <summary>
    /// The returned ArraySegment points into a shared internal buffer ( not a fresh allocation ). The caller must be done with this data (e.g. SendAsync has
    /// completed) before calling Build() again, or the next frame's write will overwrite data that hasn't been sent yet.
    /// </summary>
    public ArraySegment<byte> Build(float[] vertexBuffer, float[] normalBuffer, int vertexFloatCount)
    {
        _payloadBuffer[0] = (float)vertexFloatCount;
        Array.Copy(vertexBuffer, 0, _payloadBuffer, 1, vertexFloatCount);
        Array.Copy(normalBuffer, 0, _payloadBuffer, 1 + vertexFloatCount, vertexFloatCount);

        int totalFloats = 1 + vertexFloatCount * 2;
        int totalBytes = totalFloats * sizeof(float);

        MemoryMarshal.AsBytes(_payloadBuffer.AsSpan(0, totalFloats))
            .CopyTo(_payloadByteBuffer);

        return new ArraySegment<byte>(_payloadByteBuffer, 0, totalBytes);
    }
}
