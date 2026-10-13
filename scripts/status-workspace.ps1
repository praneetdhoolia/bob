$ErrorActionPreference = 'Stop'
Write-Output 'Bob: http://127.0.0.1:3217/'
try {
    $workspaceStatus = Invoke-RestMethod 'http://127.0.0.1:3217/api/status' -TimeoutSec 5
    [pscustomobject]@{State=$workspaceStatus.state;Ready=$workspaceStatus.ready;Active=$workspaceStatus.active;Source=$workspaceStatus.source;Seconds=[math]::Round($workspaceStatus.physicalSeconds,1);Ticks=$workspaceStatus.ticks;Observations=$workspaceStatus.observations;ObservationMs=[math]::Round($workspaceStatus.latestObservationMs,1);MissedDeadlines=$workspaceStatus.deadlineMisses;Fit=$workspaceStatus.weightsUpdate} | Format-List
    if ($workspaceStatus.error) { Write-Output $workspaceStatus.error }
} catch { Write-Output 'Agent unavailable. Run npm start; see .runtime/agent.err.log if startup fails.' }
$workspaceListeners = Get-NetTCPConnection -LocalAddress 127.0.0.1 -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in @(3210,3213,3214,3215,3217,3218,11435) }
$workspaceListeners | Select-Object LocalPort,OwningProcess | Format-Table -AutoSize
