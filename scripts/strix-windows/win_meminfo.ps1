$os = Get-CimInstance Win32_OperatingSystem
"TotalVisMemGB: " + [math]::Round($os.TotalVisibleMemorySize/1MB,1)
"FreeVisMemGB: " + [math]::Round($os.FreePhysicalMemory/1MB,1)
"TotalVirtGB: " + [math]::Round($os.TotalVirtualMemorySize/1MB,1)
"FreeVirtGB: " + [math]::Round($os.FreeVirtualMemory/1MB,1)
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 5 Name,@{n='WS_GB';e={[math]::Round($_.WorkingSet64/1GB,1)}} | Format-Table -AutoSize | Out-String
