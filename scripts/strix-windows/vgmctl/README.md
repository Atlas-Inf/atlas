<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# vgmctl — AMD Variable Graphics Memory control (ADLX)

Non-interactive VGM tool for Strix Halo Windows. `vgmctl set 32` is required
before serving Flash-Next: VGM 96 GB leaves only ~48 GiB usable to HIP, VGM
0.5 GB ~63 GB; VGM 32 GB exposes 96 GB (see win_serve_flashnext_nvfp4.ps1).

Build (needs MSVC + the AMD ADLX SDK — NOT vendored here):
1. Clone the ADLX SDK beside vgmctl.cpp so `SDK/` resolves:
   `git clone https://github.com/GPUOpen-LibrariesAndSDKs/ADLX` and copy its
   `SDK/` directory next to `vgmctl.cpp` (or unpack the SDK zip the same way).
2. From a `vcvars64` shell, compile the three sources:

```
cl /nologo /EHsc /O2 /std:c++17 /I. /Ivgmctl vgmctl\vgmctl.cpp ^
   SDK\ADLXHelper\Windows\Cpp\ADLXHelper.cpp SDK\Platform\Windows\WinAPIs.cpp ^
   /Fe:vgmctl.exe
```

Usage: `vgmctl list` prints supported/default/current options;
`vgmctl set <carvedGB>` applies the matching option — ADLX restarts the
system on success. Requires `amdadlx64.dll` (ships with the Adrenalin driver).
