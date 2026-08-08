using System.Net.WebSockets;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AchernarCs;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.UseWebSockets();

app.Map("/phillips-ocean", async context =>
{
    if (context.WebSockets.IsWebSocketRequest)
    {
        using var webSocket = await context.WebSockets.AcceptWebSocketAsync();
        using var simulator = new PhillipsOceanGpuSimulator();
        var cancellationToken = context.RequestAborted;

        try
        {
            var targetFrameTicks = TimeSpan.FromSeconds(PhillipsOceanGpuSimulator.FrameInterval).Ticks;
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            while (webSocket.State == WebSocketState.Open && !cancellationToken.IsCancellationRequested)
            {
                var frameStartTicks = stopwatch.ElapsedTicks;
                var elapsedSeconds = stopwatch.Elapsed.TotalSeconds;

                var payloadBytes = simulator.GenerateNextFramePayload(elapsedSeconds);
                await webSocket.SendAsync(payloadBytes, WebSocketMessageType.Binary, true, cancellationToken);

                var frameElapsedTicks = stopwatch.ElapsedTicks - frameStartTicks;
                var remainingTicks = targetFrameTicks - frameElapsedTicks;

                if (remainingTicks > 0)
                {
                    var delayMilliseconds = (int)(remainingTicks * 1000 / System.Diagnostics.Stopwatch.Frequency);
                    if (delayMilliseconds > 0)
                    {
                        await Task.Delay(delayMilliseconds, cancellationToken);
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

app.Map("/metaballs", async context =>
{
    if (context.WebSockets.IsWebSocketRequest)
    {
        using var webSocket = await context.WebSockets.AcceptWebSocketAsync();
        using var simulator = new MetaballsGpuSimulator();
        var cancellationToken = context.RequestAborted;

        try
        {
            var targetFrameTicks = TimeSpan.FromSeconds(MetaballsGpuSimulator.FrameInterval).Ticks;
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            while (webSocket.State == WebSocketState.Open && !cancellationToken.IsCancellationRequested)
            {
                var frameStartTicks = stopwatch.ElapsedTicks;
                var elapsedSeconds = stopwatch.Elapsed.TotalSeconds;

                var payloadBytes = simulator.GenerateNextFramePayload(elapsedSeconds);
                await webSocket.SendAsync(payloadBytes, WebSocketMessageType.Binary, true, cancellationToken);

                var frameElapsedTicks = stopwatch.ElapsedTicks - frameStartTicks;
                var remainingTicks = targetFrameTicks - frameElapsedTicks;

                if (remainingTicks > 0)
                {
                    var delayMilliseconds = (int)(remainingTicks * 1000 / System.Diagnostics.Stopwatch.Frequency);
                    if (delayMilliseconds > 0)
                    {
                        await Task.Delay(delayMilliseconds, cancellationToken);
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
