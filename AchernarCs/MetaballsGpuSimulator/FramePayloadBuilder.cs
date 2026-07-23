using System.Runtime.InteropServices;

namespace AchernarCs;

public class FramePayloadBuilder
{
    private readonly float[] _payloadBuffer;

    public FramePayloadBuilder(int maxTriangles)
    {
        _payloadBuffer = new float[1 + maxTriangles * 18];
    }

    public ArraySegment<byte> Build(float[] vertexBuffer, float[] normalBuffer, int vertexFloatCount)
    {
        _payloadBuffer[0] = (float)vertexFloatCount;
        Array.Copy(vertexBuffer, 0, _payloadBuffer, 1, vertexFloatCount);
        Array.Copy(normalBuffer, 0, _payloadBuffer, 1 + vertexFloatCount, vertexFloatCount);

        int totalFloats = 1 + vertexFloatCount * 2;

        var byteSpan = MemoryMarshal.AsBytes(_payloadBuffer.AsSpan(0, totalFloats));
        return new ArraySegment<byte>(byteSpan.ToArray());
    }
}
