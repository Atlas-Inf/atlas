$env:PATH = "C:\Users\azeez\.cargo\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;" + $env:PATH
Set-Location C:\Users\azeez\code\atlas-inf-pr9
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.2"
$env:ATLAS_HIPCC = "$env:HIP_PATH\bin\hipcc.exe"
$env:ATLAS_TARGET_HW = "strix-hip"
$env:ATLAS_TARGET_MODEL = "*"
$env:ATLAS_TARGET_QUANT = "nvfp4"
$env:CUDARC_CUDA_VERSION = "12080"
$env:ATLAS_NO_RDMA = "1"
$env:ATLAS_HIP_COMPAT_INCLUDE = "C:\Users\azeez\code\atlas-inf-pr9\crates\atlas-kernels\hip\compat"
cargo build --release -p spark-model --no-default-features --features cuda,gpu-examples --example dense_gemv_bf16_oracle --example gemv_fp4_vs_fp8_microtest 2>&1 | Select-Object -Last 3
if ($LASTEXITCODE -ne 0) { "BUILD FAILED"; exit 1 }
Copy-Item C:\Users\azeez\code\atlas-inf-pr9\target\x86_64-pc-windows-msvc\release\*.dll C:\Users\azeez\code\atlas-inf-pr9\target\release\examples\ -Force -ErrorAction SilentlyContinue
"=== dense_gemv_bf16_oracle (correctness) ==="
& .\target\release\examples\dense_gemv_bf16_oracle.exe 2>&1 | Select-Object -Last 4
"=== gemv_fp4_vs_fp8_microtest (bandwidth) ==="
& .\target\release\examples\gemv_fp4_vs_fp8_microtest.exe 2>&1 | Select-Object -Last 5
