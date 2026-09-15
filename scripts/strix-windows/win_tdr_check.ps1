$ev = Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddHours(-3)} -MaxEvents 200 -ErrorAction SilentlyContinue
$ev | Where-Object { $_.ProviderName -match 'Display|amd|dxg|WHEA' } | Select-Object -First 8 TimeCreated,Id,ProviderName | Format-Table -AutoSize | Out-String
"--- TDR registry ---"
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -ErrorAction SilentlyContinue | Select-Object TdrLevel,TdrDelay,TdrDdiDelay | Format-List | Out-String
