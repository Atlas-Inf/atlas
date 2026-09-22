// SPDX-License-Identifier: AGPL-3.0-only
// vgmctl — non-interactive AMD Variable Graphics Memory control via ADLX.
//   vgmctl list            print supported/default/current + every available option
//   vgmctl set <carvedGB>  SetOption on the available option whose MemoryCarved == carvedGB
//                          (ADLX triggers a system RESTART on success)
#include "SDK/ADLXHelper/Windows/Cpp/ADLXHelper.h"
#include "SDK/Include/ISystem3.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
using namespace adlx;
static ADLXHelper g_ADLXHelp;

static void printOpt(const char* tag, const IADLXVariableGraphicsMemoryOptionPtr& o) {
    const char* name = nullptr; ADLX_VARIABLE_GRAPHICS_MEMORY_MODE mode{}; adlx_double carved = 0, rem = 0;
    o->Name(&name); o->Mode(&mode); o->MemoryCarved(&carved); o->MemoryRemaining(&rem);
    printf("%s name=%s mode=%d carved_GB=%.2f remaining_GB=%.2f\n", tag, name ? name : "?", (int)mode, carved, rem);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: vgmctl list | set <carvedGB>\n"); return 2; }
    ADLX_RESULT res = g_ADLXHelp.Initialize();
    if (ADLX_FAILED(res)) { fprintf(stderr, "ADLX init failed: %d\n", (int)res); return 1; }
    IADLXSystem3Ptr system3;
    res = g_ADLXHelp.GetSystemServices()->QueryInterface(IADLXSystem3::IID(), (void**)&system3);
    if (ADLX_FAILED(res)) { fprintf(stderr, "IADLXSystem3 unavailable: %d\n", (int)res); return 1; }
    IADLXVariableGraphicsMemoryPtr vgm;
    res = system3->GetVariableGraphicsMemory(&vgm);
    if (ADLX_FAILED(res) || vgm == nullptr) { fprintf(stderr, "VGM interface unavailable: %d\n", (int)res); return 1; }
    adlx_bool supported = false; vgm->IsSupported(&supported);
    printf("supported=%d\n", (int)supported);
    IADLXVariableGraphicsMemoryOptionPtr def, cur;
    if (ADLX_SUCCEEDED(vgm->GetDefaultOption(&def)) && def) printOpt("default", def);
    if (ADLX_SUCCEEDED(vgm->GetOption(&cur)) && cur) printOpt("current", cur);
    IADLXVariableGraphicsMemoryOptionListPtr list;
    res = vgm->GetAvailableOptions(&list);
    if (ADLX_FAILED(res) || list == nullptr) { fprintf(stderr, "GetAvailableOptions failed: %d\n", (int)res); return 1; }
    adlx_uint n = list->Size();
    printf("available=%u\n", n);
    IADLXVariableGraphicsMemoryOptionPtr match;
    double want = (argc >= 3) ? atof(argv[2]) : -1;
    for (adlx_uint i = 0; i < n; ++i) {
        IADLXVariableGraphicsMemoryOptionPtr o;
        if (ADLX_FAILED(list->At(i, &o)) || !o) continue;
        char tag[32]; snprintf(tag, sizeof tag, "option[%u]", i); printOpt(tag, o);
        adlx_double carved = 0; o->MemoryCarved(&carved);
        if (want >= 0 && std::fabs(carved - want) < 0.5) match = o;
    }
    if (strcmp(argv[1], "list") == 0) { g_ADLXHelp.Terminate(); return 0; }
    if (strcmp(argv[1], "set") != 0 || want < 0) { fprintf(stderr, "usage: vgmctl set <carvedGB>\n"); return 2; }
    if (!match) { fprintf(stderr, "no available option with carved_GB=%.0f\n", want); return 3; }
    printf("SETTING carved_GB=%.0f — ADLX will restart the system\n", want); fflush(stdout);
    res = vgm->SetOption(match);
    printf("SetOption result=%d\n", (int)res); fflush(stdout);
    g_ADLXHelp.Terminate();
    return ADLX_SUCCEEDED(res) ? 0 : 1;
}
