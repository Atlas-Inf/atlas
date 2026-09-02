Set-Location C:\Users\azeez\code\atlas-inf-pr9
$tests = @("gdn_split4_microtest", "inferspark_attn_paged_bf16_microtest", "inferspark_attn_microtest", "w8a16_microtest", "w4a16_parity_microtest")
foreach ($t in $tests) {
    $log = "C:\Users\azeez\q38-win-$t.log"
    & ".\target\release\examples\$t.exe" *> $log
    "$t EXIT: $LASTEXITCODE"
    Get-Content $log -Tail 4
    "---"
}
