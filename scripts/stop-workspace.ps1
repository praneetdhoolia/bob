[CmdletBinding(SupportsShouldProcess)]
param()
$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$workspaceMetadata = Join-Path $workspaceRoot '.runtime\agent-server.json'
if (-not (Test-Path -LiteralPath $workspaceMetadata)) { Write-Output 'No managed agent process is recorded.'; return }
$workspaceInfo = Get-Content -LiteralPath $workspaceMetadata -Raw | ConvertFrom-Json
$workspaceProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($workspaceInfo.pid)" -ErrorAction SilentlyContinue
if (-not $workspaceProcess) { Write-Output 'Agent already stopped.'; return }
$workspaceExpected = Join-Path $workspaceRoot 'scripts\serve.py'
if (-not $workspaceProcess.CommandLine.Contains($workspaceExpected)) { throw 'Recorded PID belongs to another process; nothing stopped.' }
$workspaceListener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 3217 -State Listen -ErrorAction SilentlyContinue
if (-not $workspaceListener -or $workspaceListener.OwningProcess -ne $workspaceInfo.pid) { throw 'Listener does not match the owned agent process; nothing stopped.' }
if ($PSCmdlet.ShouldProcess("Bob agent PID $($workspaceInfo.pid) on 3217", 'Save numerical state and shut down')) {
    $workspaceSession = Invoke-RestMethod 'http://127.0.0.1:3217/api/session' -TimeoutSec 5
    $null = Invoke-RestMethod 'http://127.0.0.1:3217/api/shutdown' -Method Post -Headers @{'X-Bob-Token'=$workspaceSession.token} -ContentType application/json -Body '{}' -TimeoutSec 20
    Wait-Process -Id $workspaceInfo.pid -Timeout 15 -ErrorAction SilentlyContinue
    if (Get-Process -Id $workspaceInfo.pid -ErrorAction SilentlyContinue) { throw 'Agent has not exited yet. Inspect its log; it was not forcibly killed.' }
    Write-Output 'Numerical state saved; Bob stopped. Start again with npm start.'
}
