// fsp-srv: the game's RomFS (a virtual RomFS image over the extracted dump, see tools/romfs.py) and save data
// (a host folder per save in %APPDATA%\Zonaite\save). Commands per switchbrew "Filesystem services".
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "zonaite/ipc.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace fs = std::filesystem;

namespace zn {

std::string g_dump_dir;  // set by main from the command line
std::string g_build_dir;

namespace {

constexpr uint32_t kResultPathNotFound = 0x202;       // fs (2), desc 1
constexpr uint32_t kResultPathAlreadyExists = 0x402;  // fs, desc 2
constexpr uint32_t kResultTargetNotFound = 0x7D402;   // fs, desc 1002 (save data not found)

// ---------------------------------------------------------------- virtual RomFS image

class RomFsImage {
public:
    bool load(const std::string& build_dir, const std::string& romfs_dir) {
        romfs_dir_ = romfs_dir;
        std::ifstream meta(build_dir + "/romfs_meta.bin", std::ios::binary);
        if (!meta) return false;
        meta_.assign(std::istreambuf_iterator<char>(meta), {});
        std::ifstream list(build_dir + "/romfs_files.txt");
        std::string line;
        std::getline(list, line);  // "# tables_offset N total_size M"
        std::sscanf(line.c_str(), "# tables_offset %llu total_size %llu", (unsigned long long*)&tables_offset_,
                    (unsigned long long*)&total_);
        while (std::getline(list, line)) {
            File f;
            size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
            f.offset = std::stoull(line.substr(0, t1));
            f.size = std::stoull(line.substr(t1 + 1, t2 - t1 - 1));
            f.path = line.substr(t2 + 1);
            if (f.size) files_.push_back(std::move(f));
        }
        std::sort(files_.begin(), files_.end(), [](const File& a, const File& b) { return a.offset < b.offset; });
        log("RomFS: %zu files, %.2f GB virtual image", files_.size(), total_ / 1e9);
        return true;
    }

    uint64_t size() const { return total_; }

    void read(uint8_t* dst, uint64_t off, uint64_t size) {
        while (size) {
            uint64_t n;
            if (off < 0x200) {  // header
                n = std::min<uint64_t>(size, 0x200 - off);
                std::memcpy(dst, meta_.data() + off, n);
            } else if (off >= tables_offset_) {  // hash and meta tables
                uint64_t rel = off - tables_offset_;
                n = rel < meta_.size() - 0x200 ? std::min<uint64_t>(size, meta_.size() - 0x200 - rel) : size;
                if (rel < meta_.size() - 0x200) std::memcpy(dst, meta_.data() + 0x200 + rel, n);
                else std::memset(dst, 0, n);
            } else {  // file data, with zero padding between files
                auto it = std::upper_bound(files_.begin(), files_.end(), off,
                                           [](uint64_t o, const File& f) { return o < f.offset; });
                if (it != files_.begin() && off < std::prev(it)->offset + std::prev(it)->size) {
                    const File& f = *std::prev(it);
                    n = std::min<uint64_t>(size, f.offset + f.size - off);
                    read_file(f, dst, off - f.offset, n);
                } else {
                    uint64_t next = it != files_.end() ? it->offset : tables_offset_;
                    n = std::min<uint64_t>(size, next - off);
                    std::memset(dst, 0, n);
                }
            }
            dst += n;
            off += n;
            size -= n;
        }
    }

private:
    struct File {
        uint64_t offset, size;
        std::string path;
    };

    // positional reads on shared handles: safe from any number of guest threads
    void read_file(const File& f, uint8_t* dst, uint64_t at, uint64_t n) {
        HANDLE h = handle(f);
        while (n) {
            DWORD chunk = (DWORD)std::min<uint64_t>(n, 1u << 30), got = 0;
            OVERLAPPED ov{};
            ov.Offset = (DWORD)at;
            ov.OffsetHigh = (DWORD)(at >> 32);
            if (!ReadFile(h, dst, chunk, &got, &ov) || got == 0) {
                log("RomFS: read failed: %s at %llu", f.path.c_str(), (unsigned long long)at);
                std::memset(dst, 0, n);
                return;
            }
            dst += got;
            at += got;
            n -= got;
        }
    }

    HANDLE handle(const File& f) {
        std::lock_guard<std::mutex> l(lock_);
        auto it = handles_.find(&f);
        if (it != handles_.end()) return it->second;
        if (handles_.size() > 512) {  // keep the open-file count bounded
            for (auto& [k, h] : handles_) CloseHandle(h);
            handles_.clear();
        }
        std::wstring path = fs::path(romfs_dir_ + "/" + f.path).wstring();
        HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) fatal(nullptr, "RomFS: cannot open %s", f.path.c_str());
        handles_[&f] = h;
        return h;
    }

    std::string romfs_dir_;
    std::vector<uint8_t> meta_;
    std::vector<File> files_;
    uint64_t tables_offset_ = 0, total_ = 0;
    std::mutex lock_;
    std::unordered_map<const File*, HANDLE> handles_;
};

RomFsImage& romfs() {
    static RomFsImage img;
    static bool loaded = [] {
        std::string dir;
        for (const char* n : {"RomFs", "romfs", "RomFS"})
            if (fs::is_directory(g_dump_dir + "/" + n)) dir = g_dump_dir + "/" + n;
        if (dir.empty() || !img.load(g_build_dir, dir))
            fatal(nullptr, "RomFS: need the dump folder (argument 2) and build/game/romfs_meta.bin");
        return true;
    }();
    (void)loaded;
    return img;
}

class RomStorage : public Service {
public:
    RomStorage() : Service("fs:IStorage(RomFS)") {
        on(0, "Read", [](IpcCtx& ctx) {
            uint64_t off = ctx.pop<uint64_t>(), size = ctx.pop<uint64_t>();
            Buffer b = ctx.out_buffer();
            size = std::min(size, b.size);
            romfs().read(ctx.m + b.addr, off, size);
        });
        on(4, "GetSize", [](IpcCtx& ctx) { ctx.push<uint64_t>(romfs().size()); });
    }
};

// ---------------------------------------------------------------- host-folder file system (save data)

std::string read_path(IpcCtx& ctx, size_t index = 0) {
    Buffer b = ctx.in_buffer(index);
    std::string s;
    for (uint64_t i = 0; i < b.size && ctx.m[b.addr + i]; i++) s += (char)ctx.m[b.addr + i];
    return s;
}

class HostFile : public Service {
public:
    explicit HostFile(fs::path p) : Service("fs:IFile"), path_(std::move(p)) {
        on(0, "Read", [this](IpcCtx& ctx) {
            ctx.pop<uint64_t>();  // option + padding
            uint64_t off = ctx.pop<uint64_t>(), size = ctx.pop<uint64_t>();
            Buffer b = ctx.out_buffer();
            std::ifstream f(path_, std::ios::binary);
            f.seekg((std::streamoff)off);
            f.read((char*)ctx.m + b.addr, (std::streamsize)std::min(size, b.size));
            ctx.push<uint64_t>((uint64_t)f.gcount());
        });
        on(1, "Write", [this](IpcCtx& ctx) {
            ctx.pop<uint64_t>();
            uint64_t off = ctx.pop<uint64_t>(), size = ctx.pop<uint64_t>();
            Buffer b = ctx.in_buffer();
            std::fstream f(path_, std::ios::binary | std::ios::in | std::ios::out);
            f.seekp((std::streamoff)off);
            f.write((const char*)ctx.m + b.addr, (std::streamsize)std::min(size, b.size));
        });
        on(2, "Flush", [](IpcCtx&) {});
        on(3, "SetSize", [this](IpcCtx& ctx) { fs::resize_file(path_, ctx.pop<uint64_t>()); });
        on(4, "GetSize", [this](IpcCtx& ctx) { ctx.push<uint64_t>(fs::file_size(path_)); });
    }

private:
    fs::path path_;
};

class HostDirectory : public Service {
public:
    explicit HostDirectory(fs::path p) : Service("fs:IDirectory") {
        for (auto& e : fs::directory_iterator(p)) entries_.push_back(e);
        on(0, "Read", [this](IpcCtx& ctx) {
            Buffer b = ctx.out_buffer();
            uint64_t n = 0;
            while (pos_ < entries_.size() && (n + 1) * 0x310 <= b.size) {
                uint8_t* e = ctx.m + b.addr + n * 0x310;
                std::memset(e, 0, 0x310);
                std::string name = entries_[pos_].path().filename().string();
                std::memcpy(e, name.data(), std::min<size_t>(name.size(), 0x300));
                bool dir = entries_[pos_].is_directory();
                e[0x304] = dir ? 0 : 1;
                zg::st64(e, 0x308, dir ? 0 : (uint64_t)entries_[pos_].file_size());
                pos_++;
                n++;
            }
            ctx.push<uint64_t>(n);
        });
        on(1, "GetEntryCount", [this](IpcCtx& ctx) { ctx.push<uint64_t>(entries_.size()); });
    }

private:
    std::vector<fs::directory_entry> entries_;
    size_t pos_ = 0;
};

class HostFileSystem : public Service {
public:
    explicit HostFileSystem(fs::path root) : Service("fs:IFileSystem"), root_(std::move(root)) {
        fs::create_directories(root_);
        on(0, "CreateFile", [this](IpcCtx& ctx) {
            ctx.pop<uint32_t>();  // option
            uint64_t size = ctx.pop<uint64_t>();
            fs::path p = resolve(read_path(ctx));
            if (fs::exists(p)) { ctx.result = kResultPathAlreadyExists; return; }
            std::ofstream(p, std::ios::binary).close();
            fs::resize_file(p, size);
        });
        on(1, "DeleteFile", [this](IpcCtx& ctx) {
            if (!fs::remove(resolve(read_path(ctx)))) ctx.result = kResultPathNotFound;
        });
        on(2, "CreateDirectory", [this](IpcCtx& ctx) {
            fs::path p = resolve(read_path(ctx));
            if (fs::exists(p)) { ctx.result = kResultPathAlreadyExists; return; }
            fs::create_directory(p);
        });
        on(3, "DeleteDirectory", [this](IpcCtx& ctx) { fs::remove(resolve(read_path(ctx))); });
        on(4, "DeleteDirectoryRecursively", [this](IpcCtx& ctx) { fs::remove_all(resolve(read_path(ctx))); });
        on(5, "RenameFile", [this](IpcCtx& ctx) { fs::rename(resolve(read_path(ctx, 0)), resolve(read_path(ctx, 1))); });
        on(6, "RenameDirectory",
           [this](IpcCtx& ctx) { fs::rename(resolve(read_path(ctx, 0)), resolve(read_path(ctx, 1))); });
        on(7, "GetEntryType", [this](IpcCtx& ctx) {
            fs::path p = resolve(read_path(ctx));
            if (!fs::exists(p)) { ctx.result = kResultPathNotFound; return; }
            ctx.push<uint32_t>(fs::is_directory(p) ? 0 : 1);
        });
        on(8, "OpenFile", [this](IpcCtx& ctx) {
            fs::path p = resolve(read_path(ctx));
            if (!fs::is_regular_file(p)) { ctx.result = kResultPathNotFound; return; }
            ctx.push_object(std::make_shared<HostFile>(p));
        });
        on(9, "OpenDirectory", [this](IpcCtx& ctx) {
            fs::path p = resolve(read_path(ctx));
            if (!fs::is_directory(p)) { ctx.result = kResultPathNotFound; return; }
            ctx.push_object(std::make_shared<HostDirectory>(p));
        });
        on(10, "Commit", [](IpcCtx&) {});
        on(11, "GetFreeSpaceSize", [](IpcCtx& ctx) { ctx.push<uint64_t>(4ull << 30); });
        on(12, "GetTotalSpaceSize", [](IpcCtx& ctx) { ctx.push<uint64_t>(8ull << 30); });
        on(13, "CleanDirectoryRecursively", [this](IpcCtx& ctx) {
            fs::path p = resolve(read_path(ctx));
            for (auto& e : fs::directory_iterator(p)) fs::remove_all(e.path());
        });
    }

private:
    fs::path resolve(const std::string& guest) const {
        std::string rel = guest;
        while (!rel.empty() && rel[0] == '/') rel.erase(0, 1);
        return root_ / fs::u8path(rel);
    }
    fs::path root_;
};

// (tests point ZN_SAVE_DIR at a scratch copy, never the player's saves)
fs::path save_root() { return fs::u8path(save_dir()); }

constexpr uint64_t kProgramId = 0x0100F2C0115B6000ull;
constexpr uint8_t kSaveTypeAccount = 1, kSaveTypeDevice = 3, kSaveTypeTemporary = 4, kSaveTypeCache = 5;

// SaveDataAttribute: program_id u64, user id u128, system save id u64, type u8, rank u8, index u16, ...
// Program id 0 means the calling program. Sets *type to the save data type.
fs::path save_path(IpcCtx& ctx, uint8_t* out_type = nullptr, uint64_t* out_program = nullptr) {
    ctx.pop<uint64_t>();  // space id (u8) + padding
    uint64_t program = ctx.pop<uint64_t>();
    if (program == 0) program = kProgramId;
    if (out_program) *out_program = program;
    uint64_t uid_lo = ctx.pop<uint64_t>(), uid_hi = ctx.pop<uint64_t>();
    uint64_t system_id = ctx.pop<uint64_t>();
    uint8_t type = ctx.pop<uint8_t>();
    if (out_type) *out_type = type;
    char name[96];
    std::snprintf(name, sizeof name, "%016llx/%u_%016llx%016llx_%016llx", (unsigned long long)program, type,
                  (unsigned long long)uid_hi, (unsigned long long)uid_lo, (unsigned long long)system_id);
    return save_root() / name;
}

// Cache storage (index 0). Its sizes, as the game created it, are kept beside it in "<dir>.size": the game deletes
// and recreates cache storage whose size it doesn't expect.
fs::path cache_storage_path() {
    char name[96];
    std::snprintf(name, sizeof name, "%016llx/%u_%016llx%016llx_%016llx", (unsigned long long)kProgramId,
                  kSaveTypeCache, 0ull, 0ull, 0ull);
    return save_root() / name;
}

}  // namespace

void fs_create_cache_storage(int64_t size, int64_t journal_size) {
    fs::create_directories(cache_storage_path());
    std::ofstream(cache_storage_path().string() + ".size") << size << " " << journal_size << "\n";
    log("fs: created cache storage (%lld bytes)", (long long)size);
}

namespace {

// OpenSaveDataFileSystem and its read-only form. The game's own account, device and cache saves exist from the
// first launch on (am EnsureSaveData creates them on a console), so they are created on first open; other
// programs' saves are only opened if they exist.
void open_save_data(IpcCtx& ctx, bool writable) {
    uint8_t type = 0;
    uint64_t program = 0;
    fs::path p = save_path(ctx, &type, &program);
    if (writable && program == kProgramId && !fs::exists(p) &&
        (type == kSaveTypeAccount || type == kSaveTypeDevice || type == kSaveTypeCache)) {
        fs::create_directories(p);
        log("fs: created save data %s", p.string().c_str());
    }
    // Temporary storage exists from launch and starts empty every run
    static std::once_flag temporary_reset;
    if (type == kSaveTypeTemporary)
        std::call_once(temporary_reset, [&] {
            std::error_code ec;
            fs::remove_all(p, ec);
            fs::create_directories(p);
        });
    if (!fs::exists(p)) {
        log("fs: no save data %s (type %u)", p.string().c_str(), type);
        ctx.result = kResultTargetNotFound;
        return;
    }
    ctx.push_object(std::make_shared<HostFileSystem>(p));
}

// SaveDataInfo, as ISaveDataInfoReader returns it
struct SaveDataInfo {
    uint64_t save_id;
    uint8_t space_id, type;
    uint8_t pad0[6];
    uint64_t uid_lo, uid_hi;
    uint64_t system_save_id, program_id, size;
    uint16_t index;
    uint8_t rank;
    uint8_t pad1[0x25];
};
static_assert(sizeof(SaveDataInfo) == 0x60);

// Lists this program's save data directories (named "<type>_<uid hi><uid lo>_<system save id>" by save_path)
class SaveDataInfoReader : public Service {
public:
    explicit SaveDataInfoReader(uint8_t space) : Service("fs:ISaveDataInfoReader") {
        std::error_code ec;
        uint64_t id = 1;
        for (const auto& e : fs::directory_iterator(save_root() / fs::u8path(program_dir()), ec)) {
            unsigned type = 0;
            unsigned long long hi = 0, lo = 0, sys = 0;
            if (!e.is_directory() ||
                std::sscanf(e.path().filename().string().c_str(), "%u_%16llx%16llx_%16llx", &type, &hi, &lo, &sys) != 4)
                continue;
            SaveDataInfo info{};
            info.save_id = id++;
            info.space_id = space;
            info.type = (uint8_t)type;
            info.uid_lo = lo;
            info.uid_hi = hi;
            info.system_save_id = sys;
            info.program_id = kProgramId;
            infos_.push_back(info);
        }
        on(0, "ReadSaveDataInfo", [this](IpcCtx& ctx) {
            const Buffer b = ctx.out_buffer();
            uint64_t n = 0;
            while (next_ < infos_.size() && (n + 1) * sizeof(SaveDataInfo) <= b.size) {
                std::memcpy(ctx.m + b.addr + n * sizeof(SaveDataInfo), &infos_[next_++], sizeof(SaveDataInfo));
                n++;
            }
            ctx.push<uint64_t>(n);
        });
    }

private:
    static std::string program_dir() {
        char name[17];
        std::snprintf(name, sizeof name, "%016llx", (unsigned long long)kProgramId);
        return name;
    }
    std::vector<SaveDataInfo> infos_;
    size_t next_ = 0;
};

// ---------------------------------------------------------------- the proxy

class FileSystemProxy : public Service {
public:
    FileSystemProxy() : Service("fsp-srv") {
        on(1, "SetCurrentProcess", [](IpcCtx&) {});
        on(22, "CreateSaveDataFileSystem", [](IpcCtx& ctx) {
            fs::path p = save_path(ctx);
            fs::create_directories(p);
            log("fs: created save data %s", p.string().c_str());
        });
        on(51, "OpenSaveDataFileSystem", [](IpcCtx& ctx) { open_save_data(ctx, true); });
        // Read-only: other games' saves too (TotK looks for Breath of the Wild's for its bonus items)
        on(53, "OpenReadOnlySaveDataFileSystem", [](IpcCtx& ctx) { open_save_data(ctx, false); });
        on(33, "DeleteCacheStorage", [](IpcCtx&) {
            std::error_code ec;
            fs::remove_all(cache_storage_path(), ec);
            fs::remove(cache_storage_path().string() + ".size", ec);
            log("fs: deleted cache storage");
        });
        on(34, "GetCacheStorageSize", [](IpcCtx& ctx) {
            long long size = 0, journal = 0;
            if (std::ifstream f{cache_storage_path().string() + ".size"}) f >> size >> journal;
            ctx.push<int64_t>(size);
            ctx.push<int64_t>(journal);
        });
        on(60, "OpenSaveDataInfoReader",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<SaveDataInfoReader>(1)); });
        on(61, "OpenSaveDataInfoReaderBySaveDataSpaceId",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<SaveDataInfoReader>(ctx.pop<uint8_t>())); });
        on(62, "OpenSaveDataInfoReaderWithFilter",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<SaveDataInfoReader>(ctx.pop<uint8_t>())); });
        on(200, "OpenDataStorageByCurrentProcess", [](IpcCtx& ctx) {
            romfs();
            ctx.push_object(std::make_shared<RomStorage>());
        });
        on(203, "OpenPatchDataStorageByCurrentProcess", [](IpcCtx& ctx) {
            romfs();
            ctx.push_object(std::make_shared<RomStorage>());
        });
        on(1003, "DisableAutoSaveDataCreation", [](IpcCtx&) {});
        on(1004, "SetGlobalAccessLogMode", [](IpcCtx&) {});
        on(1005, "GetGlobalAccessLogMode", [](IpcCtx& ctx) { ctx.push<uint32_t>(0); });
        on(1006, "OutputAccessLogToSdCard", [](IpcCtx&) {});
        on(1011, "GetProgramIndexForAccessLog", [](IpcCtx& ctx) {
            ctx.push<uint32_t>(0);
            ctx.push<uint32_t>(1);
        });
        on(1016, "FlushAccessLogOnSdCard", [](IpcCtx&) {});
    }
};

ServiceRegistrar reg("fsp-srv", [] { return std::make_shared<FileSystemProxy>(); });

}  // namespace
std::string save_dir() {
    if (const char* dir = std::getenv("ZN_SAVE_DIR")) return dir;
    const char* appdata = std::getenv("APPDATA");
    const std::string base = std::string(appdata ? appdata : ".") + "\Zonaite\\";
    return base + (game_version() == "1.2.1" ? "save" : "save-" + game_version());
}

}  // namespace zn
