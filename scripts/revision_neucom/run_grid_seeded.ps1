# Seeded re-run of the five-metric grid (NEUCOM revision). Runs, one after another:
#   configs/revision_grid_seeded.yaml          RN-50, ViT-B/16  (100 images)
#   configs/revision_grid_seeded_attnlrp.yaml  AttnLRP on ViT-B/16, own process
#   configs/revision_grid_seeded_50.yaml       the six other backbones (50 images)
# Each part runs at below-normal priority and is resumed up to three times if the
# process dies (e.g. a GPU fault) or leaves failed evaluations, which the runner
# retries on resume. Keeps Windows awake until done.
# Progress: grid_seeded.log; runner output: grid_seeded_<part>.out/.err.
# Writes DONE_GRID at the end. A STOP file in this folder halts the queue.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $here "..\..")).Path
$log  = Join-Path $here "grid_seeded.log"
$py   = Join-Path $env:USERPROFILE "miniconda3\python.exe"
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $log }

Remove-Item (Join-Path $here "DONE_GRID") -ErrorAction SilentlyContinue
Start-Process $py -ArgumentList "keep_awake.py", $PID -WorkingDirectory $here -WindowStyle Hidden
Log "queue started (pid $PID)"

$parts = @(
    @{ cfg = "configs/revision_grid_seeded.yaml";         run = "revision_grid_seeded" },
    @{ cfg = "configs/revision_grid_seeded_attnlrp.yaml"; run = "revision_grid_seeded_attnlrp" },
    @{ cfg = "configs/revision_grid_seeded_50.yaml";      run = "revision_grid_seeded_50" }
)
foreach ($part in $parts) {
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        if (Test-Path (Join-Path $here "STOP")) { Log "STOP file found, halting"; exit 1 }
        $out = Join-Path $here "grid_seeded_$($part.run).out"
        $err = Join-Path $here "grid_seeded_$($part.run).err"
        Log "start $($part.cfg) (attempt $attempt)"
        $p = Start-Process $py -ArgumentList "-u", "-m", "xai_bench.runner", "--config", $part.cfg `
             -WorkingDirectory $root -NoNewWindow -PassThru `
             -RedirectStandardOutput $out -RedirectStandardError $err
        $null = $p.Handle   # cache the handle, or ExitCode is empty after exit (PS 5.1)
        try { $p.PriorityClass = "BelowNormal" } catch {}
        $p.WaitForExit()
        $code = $p.ExitCode
        $failed = 0
        $fj = Join-Path $root "results\$($part.run)\failures.json"
        if (Test-Path $fj) { $failed = @((Get-Content $fj -Raw | ConvertFrom-Json).evaluations).Count }
        Log "end $($part.cfg): exit $code, failed evaluations $failed"
        if ($code -eq 0 -and $failed -eq 0) { break }
        Start-Sleep 60
    }
}
Log "queue finished"
"done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Encoding utf8 (Join-Path $here "DONE_GRID")
