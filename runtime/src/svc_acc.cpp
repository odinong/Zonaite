// acc: user accounts. There is one local user, always signed in and open; the game keys its save data by this uid.
#include <cstring>

#include "zonaite/ipc.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

struct Uid {
    uint64_t lo, hi;
};
constexpr Uid kUser{0x5A4F4E4149544501ull, 0x0000000000000001ull};  // "ZONAITE" + 1, any nonzero value works
constexpr uint32_t kResultUserNotFound = 124 | 100 << 9;           // acc, desc 100

// ProfileBase (0x38): uid, last edit timestamp, nickname (UTF-8, NUL padded)
struct ProfileBase {
    Uid uid;
    uint64_t timestamp;
    char nickname[0x20];
};
static_assert(sizeof(ProfileBase) == 0x38);

ProfileBase profile_base() {
    ProfileBase p{};
    p.uid = kUser;
    p.timestamp = 1700000000;
    std::strncpy(p.nickname, "Player", sizeof p.nickname);
    return p;
}

void write_out(IpcCtx& ctx, size_t i, const void* data, size_t size) {
    const Buffer b = ctx.out_buffer(i);
    std::memcpy(ctx.m + b.addr, data, size < b.size ? size : b.size);
}

class Profile : public Service {
public:
    Profile() : Service("acc:IProfile") {
        on(0, "Get", [](IpcCtx& ctx) {
            uint8_t user_data[0x80]{};  // icon id, background colour, ...: zeros are the defaults
            write_out(ctx, 0, user_data, sizeof user_data);
            ctx.push(profile_base());
        });
        on(1, "GetBase", [](IpcCtx& ctx) { ctx.push(profile_base()); });
        on(10, "GetImageSize", [](IpcCtx& ctx) { ctx.push<uint32_t>(0); });
        on(11, "LoadImage", [](IpcCtx& ctx) { ctx.push<uint32_t>(0); });
    }
};

class ManagerForApplication : public Service {
public:
    ManagerForApplication() : Service("acc:IManagerForApplication") {
        on(0, "CheckAvailability", [](IpcCtx&) {});
        on(1, "GetAccountId", [](IpcCtx& ctx) { ctx.push<uint64_t>(kUser.lo); });
        on(130, "GetNintendoAccountUserResourceCacheForApplication", [](IpcCtx& ctx) { ctx.push<uint64_t>(0); });
        on(160, "StoreOpenContext", [](IpcCtx&) {});
    }
};

bool is_user(IpcCtx& ctx) {
    const Uid u = ctx.pop<Uid>();
    return u.lo == kUser.lo && u.hi == kUser.hi;
}

class AccountService : public Service {
public:
    explicit AccountService(const char* name) : Service(name) {
        on(0, "GetUserCount", [](IpcCtx& ctx) { ctx.push<int32_t>(1); });
        on(1, "GetUserExistence", [](IpcCtx& ctx) { ctx.push<uint8_t>(is_user(ctx)); });
        on(2, "ListAllUsers", [](IpcCtx& ctx) { write_out(ctx, 0, &kUser, sizeof kUser); });
        on(3, "ListOpenUsers", [](IpcCtx& ctx) { write_out(ctx, 0, &kUser, sizeof kUser); });
        on(4, "GetLastOpenedUser", [](IpcCtx& ctx) { ctx.push(kUser); });
        on(5, "GetProfile", [](IpcCtx& ctx) {
            if (!is_user(ctx)) {
                ctx.result = kResultUserNotFound;
                return;
            }
            ctx.push_object(std::make_shared<Profile>());
        });
        on(50, "IsUserRegistrationRequestPermitted", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(51, "TrySelectUserWithoutInteraction", [](IpcCtx& ctx) { ctx.push(kUser); });
        on(100, "InitializeApplicationInfoRestricted", [](IpcCtx&) {});
        on(101, "GetBaasAccountManagerForApplication",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<ManagerForApplication>()); });
        on(110, "StoreSaveDataThumbnail", [](IpcCtx&) {});
        on(111, "ClearSaveDataThumbnail", [](IpcCtx&) {});
        on(140, "InitializeApplicationInfo", [](IpcCtx&) {});
        on(160, "InitializeApplicationInfoV2", [](IpcCtx&) {});
        on(150, "IsUserAccountSwitchLocked", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
    }
};

ServiceRegistrar reg_u0("acc:u0", [] { return std::make_shared<AccountService>("acc:u0"); });
ServiceRegistrar reg_u1("acc:u1", [] { return std::make_shared<AccountService>("acc:u1"); });

}  // namespace

void account_current_user(uint64_t uid[2]) {
    uid[0] = kUser.lo;
    uid[1] = kUser.hi;
}

}  // namespace zn
