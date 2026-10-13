param([Parameter(Mandatory=$true)][string]$Manifest)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$bobSpeechItems = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$bobFixtureSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    foreach ($bobSpeechItem in $bobSpeechItems) {
        $bobFixtureSynth.SelectVoice($bobSpeechItem.voice)
        $bobFixtureSynth.SetOutputToWaveFile($bobSpeechItem.path)
        $bobFixtureSynth.Speak($bobSpeechItem.text)
        $bobFixtureSynth.SetOutputToNull()
    }
} finally {
    $bobFixtureSynth.Dispose()
}
