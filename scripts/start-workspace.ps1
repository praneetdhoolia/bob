param([switch]$Open)
$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$workspacePython = Join-Path $workspaceRoot '.runtime\omni-python\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $workspacePython)) { throw 'The pinned numerical runtime is missing. See USAGE.md.' }
$workspaceAvailable = $false
try { $null = Invoke-RestMethod 'http://127.0.0.1:3217/api/health' -TimeoutSec 3; $workspaceAvailable = $true } catch { }
if (-not $workspaceAvailable) {
    $workspaceScript = Join-Path $workspaceRoot 'scripts\serve.py'
    $workspaceMetadata = Join-Path $workspaceRoot '.runtime\agent-server.json'
    $workspaceAlreadyStarting = $false
    if (Test-Path -LiteralPath $workspaceMetadata) {
        $workspaceInfo = Get-Content -LiteralPath $workspaceMetadata -Raw | ConvertFrom-Json
        $workspaceProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($workspaceInfo.pid)" -ErrorAction SilentlyContinue
        $workspaceAlreadyStarting = $workspaceProcess -and $workspaceProcess.CommandLine.Contains($workspaceScript)
    }
    if (-not $workspaceAlreadyStarting) {
        $workspaceListener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 3217 -State Listen -ErrorAction SilentlyContinue
        if ($workspaceListener) { throw 'Port 3217 is occupied by an unrecognized service. See USAGE.md; no process was stopped.' }
        Start-Process -FilePath $workspacePython -ArgumentList ('-u "' + $workspaceScript + '"') -WorkingDirectory $workspaceRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $workspaceRoot '.runtime\agent.out.log') -RedirectStandardError (Join-Path $workspaceRoot '.runtime\agent.err.log')
    }
}
Write-Output 'Bob: http://127.0.0.1:3217/ (one numerical agent; loading status appears in the interface)'
if ($Open) {
    # The listener appears only after Python imports Torch; open once it answers.
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        try { $null = Invoke-RestMethod 'http://127.0.0.1:3217/api/health' -TimeoutSec 2; break } catch { Start-Sleep -Milliseconds 500 }
    }
    Start-Process 'http://127.0.0.1:3217/'
}
