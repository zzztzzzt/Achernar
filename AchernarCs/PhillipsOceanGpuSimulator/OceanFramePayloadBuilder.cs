using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace AchernarCs;

/// <summary>
/// Envelope V1 frame packer — matches PhillipsOceanOxygen.jl and frame-parser.js.
/// </summary>
public class OceanFramePayloadBuilder
{
    private const byte EnvelopeVersion = 1;
    private const ushort ContentTypeFloat32Tensor = 1;
    private const int EnvelopeHeaderLen = 17;

    private readonly byte[] _payloadByteBuffer;

    public OceanFramePayloadBuilder(int pixelCount)
    {
        int payloadBytes = pixelCount * sizeof(float);
        _payloadByteBuffer = new byte[EnvelopeHeaderLen + payloadBytes];
    }

    /// <summary>
    /// The returned ArraySegment points into a shared internal buffer. Finish sending before calling Build() again.
    /// </summary>
    public ArraySegment<byte> Build(float[] frameBuffer)
    {
        int payloadBytes = frameBuffer.Length * sizeof(float);
        int totalBytes = EnvelopeHeaderLen + payloadBytes;

        _payloadByteBuffer[0] = EnvelopeVersion;
        BinaryPrimitives.WriteUInt16LittleEndian(_payloadByteBuffer.AsSpan(1, 2), ContentTypeFloat32Tensor);
        BinaryPrimitives.WriteUInt16LittleEndian(_payloadByteBuffer.AsSpan(3, 2), 0);
        BinaryPrimitives.WriteUInt64LittleEndian(
            _payloadByteBuffer.AsSpan(5, 8),
            (ulong)DateTime.UtcNow.Ticks * 100);
        BinaryPrimitives.WriteUInt32LittleEndian(_payloadByteBuffer.AsSpan(13, 4), (uint)payloadBytes);

        MemoryMarshal.AsBytes(frameBuffer.AsSpan()).CopyTo(_payloadByteBuffer.AsSpan(EnvelopeHeaderLen));

        return new ArraySegment<byte>(_payloadByteBuffer, 0, totalBytes);
    }
}
