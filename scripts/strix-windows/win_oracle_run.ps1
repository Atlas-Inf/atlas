Set-Location C:\Users\azeez\code\atlas-inf-pr9
& .\target\release\examples\dense_gemm_bf16_oracle.exe 2>&1 | Out-File C:\Users\azeez\q38-win-oracle.log -Encoding utf8
"ORACLE-EXIT: $LASTEXITCODE"
Get-Content C:\Users\azeez\q38-win-oracle.log -Tail 45
