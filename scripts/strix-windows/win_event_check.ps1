$ev = Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddHours(-5)} -MaxEvents 500 -ErrorAction SilentlyContinue
$gpu = $ev | Where-Object { $_.Message -match 'display|driver|GPU|reset|timeout|TDR|amdwddmg|dxgkrnl' -or $_.ProviderName -match 'Display|amd' }
if ($gpu) { $gpu | Select-Object -First 8 TimeCreated,Id,ProviderName | Format-Table -AutoSize | Out-String }
else { "No GPU/driver events in the last 5 hours" }
"--- WHEA errors ---"
$whea = $ev | Where-Object { $_.ProviderName -match 'WHEA' }
if ($whea) { $whea | Select-Object -First 4 TimeCreated,Id,Message | Format-List | Out-String }
else { "No WHEA errors" }
