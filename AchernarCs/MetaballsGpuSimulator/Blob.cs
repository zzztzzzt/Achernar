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
