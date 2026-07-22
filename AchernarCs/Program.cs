using System.Net.WebSockets;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AchernarCs;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.UseWebSockets();

app.Map("/metaballs", async context =>
{
    if (context.WebSockets.IsWebSocketRequest)
    {
        using var webSocket = await context.WebSockets.AcceptWebSocketAsync();
        var simulator = new MetaballsGpuSimulator();
        var cancellationToken = context.RequestAborted;

        try
        {
            var targetFrameTicks = TimeSpan.FromSeconds(MetaballsGpuSimulator.FrameInterval).Ticks;
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            while (webSocket.State == WebSocketState.Open && !cancellationToken.IsCancellationRequested)
            {
                var frameStartTicks = stopwatch.ElapsedTicks;
                var elapsedTotal = stopwatch.Elapsed.TotalSeconds;

                var payloadBytes = simulator.GenerateNextFramePayload(elapsedTotal);
                await webSocket.SendAsync(payloadBytes, WebSocketMessageType.Binary, true, cancellationToken);

                var frameElapsedTicks = stopwatch.ElapsedTicks - frameStartTicks;
                var remainingTicks = targetFrameTicks - frameElapsedTicks;

                if (remainingTicks > 0)
                {
                    var sleepMs = (int)(remainingTicks * 1000 / System.Diagnostics.Stopwatch.Frequency);
                    if (sleepMs > 0)
                    {
                        await Task.Delay(sleepMs, cancellationToken);
                    }
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (WebSocketException) { }
    }
    else
    {
        context.Response.StatusCode = StatusCodes.Status400BadRequest;
    }
});

app.Run("http://localhost:8080");
