# Header overrides for Eden's nvdrv

These headers are searched before Eden's own (`-I runtime/gpu/override` first) when Zonaite compiles Eden's
`core/hle/service/nvdrv` sources. They replace Eden's kernel and service-framework headers with small classes
that declare only the members nvdrv uses, backed by Zonaite's runtime (runtime/gpu/kernel_shim.cpp). That way
the nvdrv device code compiles unmodified without pulling in Eden's kernel.

Only nvdrv and runtime/gpu sources are compiled with this directory. video_core never sees it; the classes it
replaces are not used there (memory_manager.cpp includes k_process.h but doesn't use KProcess).
