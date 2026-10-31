// time: clocks and calendars. The steady clock counts seconds since Zonaite started; the system clocks are the
// host's UTC time; the one time zone is UTC. The SDK reads the clocks from a shared memory page as well as through
// the commands, so both are kept consistent. Layouts per switchbrew "PSC services" / "Glue services".
#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>
#include <memory>

#include "zonaite/ipc.h"
#include "zonaite/kernel.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

struct ClockSourceId {
    uint8_t bytes[16];
};
struct SteadyClockTimePoint {
    int64_t time_point;  // seconds
    ClockSourceId source;
};
struct SystemClockContext {
    int64_t offset;  // posix time = offset + steady time point
    SteadyClockTimePoint timestamp;
};
struct CalendarTime {
    int16_t year;
    int8_t month, day, hour, minute, second, pad;
};
struct CalendarAdditionalInfo {
    int32_t day_of_week, day_of_year;
    char name[8];
    int32_t is_dst, utc_offset;
};
struct ContinuousAdjustmentTimePoint {
    int64_t rtc_offset, diff_scale, shift_amount, lower, upper;
    ClockSourceId source;
};
struct ClockSnapshot {
    SystemClockContext user_context, network_context;
    int64_t user_time, network_time;
    CalendarTime user_calendar, network_calendar;
    CalendarAdditionalInfo user_info, network_info;
    SteadyClockTimePoint steady;
    char location[0x24];
    uint8_t automatic_correction;
    uint8_t type;
    uint16_t pad;
};
static_assert(sizeof(SteadyClockTimePoint) == 0x18 && sizeof(SystemClockContext) == 0x20);
static_assert(sizeof(CalendarTime) == 8 && sizeof(CalendarAdditionalInfo) == 0x18);
static_assert(sizeof(ContinuousAdjustmentTimePoint) == 0x38 && sizeof(ClockSnapshot) == 0xD0);

constexpr ClockSourceId kSource{{'Z', 'O', 'N', 'A', 'I', 'T', 'E', 0, 'S', 'T', 'E', 'A', 'D', 'Y', 0, 1}};
constexpr char kLocation[] = "UTC";

int64_t steady_seconds() { return (int64_t)(system_tick() / 19'200'000); }

int64_t host_posix() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

SteadyClockTimePoint steady_now() { return {steady_seconds(), kSource}; }

// Fixed at startup, so time advances with the steady clock afterwards
SystemClockContext system_context() {
    static const SystemClockContext c{host_posix() - steady_seconds(), {0, kSource}};
    return c;
}

int64_t system_time() { return system_context().offset + steady_seconds(); }

void to_calendar(int64_t posix, CalendarTime& t, CalendarAdditionalInfo& info) {
    const std::time_t tt = (std::time_t)posix;
    std::tm tm{};
    gmtime_s(&tm, &tt);
    t = {(int16_t)(tm.tm_year + 1900), (int8_t)(tm.tm_mon + 1), (int8_t)tm.tm_mday, (int8_t)tm.tm_hour,
         (int8_t)tm.tm_min, (int8_t)tm.tm_sec, 0};
    info = {};
    info.day_of_week = tm.tm_wday;
    info.day_of_year = tm.tm_yday;
    std::memcpy(info.name, kLocation, sizeof kLocation);
}

int64_t from_calendar(const CalendarTime& t) {
    std::tm tm{};
    tm.tm_year = t.year - 1900;
    tm.tm_mon = t.month - 1;
    tm.tm_mday = t.day;
    tm.tm_hour = t.hour;
    tm.tm_min = t.minute;
    tm.tm_sec = t.second;
    return (int64_t)_mkgmtime(&tm);
}

// Lock-free atomic slot in the shared page: { u32 counter; T values[2] }, the reader takes values[counter & 1]
template <typename T>
void write_slot(uint8_t* base, size_t offset, const T& v) {
    auto* counter = reinterpret_cast<std::atomic<uint32_t>*>(base + offset);
    const uint32_t next = counter->load(std::memory_order_relaxed) + 1;
    const size_t values = offset + ((4 + alignof(T) - 1) & ~(alignof(T) - 1));
    std::memcpy(base + values + (next & 1) * sizeof(T), &v, sizeof(T));
    counter->store(next, std::memory_order_release);
}

std::shared_ptr<KSharedMemory> time_shared_memory() {
    static const auto shm = [] {
        auto s = create_shared_memory(0x1000);
        // steady clock base: SDK steady time = this + seconds since boot, which is our steady clock
        write_slot(s->host, 0x00, SteadyClockTimePoint{0, kSource});
        write_slot(s->host, 0x38, system_context());  // local system clock
        write_slot(s->host, 0x80, system_context());  // network system clock
        write_slot(s->host, 0xC8, (uint8_t)0);        // automatic correction off
        write_slot(s->host, 0xD0, ContinuousAdjustmentTimePoint{0, 0, 0, 0, 0, kSource});
        return s;
    }();
    return shm;
}

class SystemClock : public Service {
public:
    SystemClock() : Service("time:ISystemClock") {
        on(0, "GetCurrentTime", [](IpcCtx& ctx) { ctx.push<int64_t>(system_time()); });
        on(1, "SetCurrentTime", [](IpcCtx&) {});
        on(2, "GetSystemClockContext", [](IpcCtx& ctx) { ctx.push(system_context()); });
        on(3, "SetSystemClockContext", [](IpcCtx&) {});
        on(4, "GetOperationEventReadableHandle",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); });
    }
};

class SteadyClock : public Service {
public:
    SteadyClock() : Service("time:ISteadyClock") {
        on(0, "GetCurrentTimePoint", [](IpcCtx& ctx) { ctx.push(steady_now()); });
        on(2, "GetTestOffset", [](IpcCtx& ctx) { ctx.push<int64_t>(0); });
        on(3, "SetTestOffset", [](IpcCtx&) {});
        on(100, "GetRtcValue", [](IpcCtx& ctx) { ctx.push<int64_t>(host_posix()); });
        on(101, "IsRtcResetDetected", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(102, "GetSetupResultValue", [](IpcCtx& ctx) { ctx.push<uint32_t>(0); });
        on(200, "GetInternalOffset", [](IpcCtx& ctx) { ctx.push<int64_t>(0); });
    }
};

void write_out(IpcCtx& ctx, size_t i, const void* data, size_t size) {
    const Buffer b = ctx.out_buffer(i);
    std::memcpy(ctx.m + b.addr, data, size < b.size ? size : b.size);
}

class TimeZoneService : public Service {
public:
    TimeZoneService() : Service("time:ITimeZoneService") {
        on(0, "GetDeviceLocationName", [](IpcCtx& ctx) {
            char name[0x24]{};
            std::memcpy(name, kLocation, sizeof kLocation);
            ctx.push(name);
        });
        on(1, "SetDeviceLocationName", [](IpcCtx&) {});
        on(2, "GetTotalLocationNameCount", [](IpcCtx& ctx) { ctx.push<uint32_t>(1); });
        on(3, "LoadLocationNameList", [](IpcCtx& ctx) {
            char name[0x24]{};
            std::memcpy(name, kLocation, sizeof kLocation);
            write_out(ctx, 0, name, sizeof name);
            ctx.push<uint32_t>(1);
        });
        // The rule is opaque to the game: an all-zero rule, which the conversions below ignore (always UTC)
        on(4, "LoadTimeZoneRule", [](IpcCtx& ctx) {
            const Buffer b = ctx.out_buffer();
            std::memset(ctx.m + b.addr, 0, b.size);
        });
        on(5, "GetTimeZoneRuleVersion", [](IpcCtx& ctx) {
            char version[0x10] = "2024a";
            ctx.push(version);
        });
        auto calendar = [](IpcCtx& ctx) {
            CalendarTime t;
            CalendarAdditionalInfo info;
            to_calendar(ctx.pop<int64_t>(), t, info);
            ctx.push(t);
            ctx.push(info);
        };
        on(100, "ToCalendarTime", calendar);
        on(101, "ToCalendarTimeWithMyRule", calendar);
        auto posix = [](IpcCtx& ctx) {
            const int64_t t = from_calendar(ctx.pop<CalendarTime>());
            write_out(ctx, 0, &t, sizeof t);
            ctx.push<uint32_t>(1);
        };
        on(201, "ToPosixTime", posix);
        on(202, "ToPosixTimeWithMyRule", posix);
    }
};

class StaticService : public Service {
public:
    explicit StaticService(const char* name) : Service(name) {
        auto system_clock = [](IpcCtx& ctx) { ctx.push_object(std::make_shared<SystemClock>()); };
        on(0, "GetStandardUserSystemClock", system_clock);
        on(1, "GetStandardNetworkSystemClock", system_clock);
        on(2, "GetStandardSteadyClock", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<SteadyClock>()); });
        on(3, "GetTimeZoneService", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<TimeZoneService>()); });
        on(4, "GetStandardLocalSystemClock", system_clock);
        on(5, "GetEphemeralNetworkSystemClock", system_clock);
        on(20, "GetSharedMemoryNativeHandle",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(time_shared_memory())); });
        on(100, "IsStandardUserSystemClockAutomaticCorrectionEnabled", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(300, "CalculateMonotonicSystemClockBaseTimePoint", [](IpcCtx& ctx) {
            const auto c = ctx.pop<SystemClockContext>();
            ctx.push<int64_t>(c.offset + c.timestamp.time_point - steady_seconds() + steady_seconds());
        });
        on(400, "GetClockSnapshot", [](IpcCtx& ctx) {
            ClockSnapshot s{};
            s.user_context = s.network_context = system_context();
            s.user_time = s.network_time = system_time();
            to_calendar(s.user_time, s.user_calendar, s.user_info);
            to_calendar(s.network_time, s.network_calendar, s.network_info);
            s.steady = steady_now();
            std::memcpy(s.location, kLocation, sizeof kLocation);
            s.type = ctx.pop<uint8_t>();
            write_out(ctx, 0, &s, sizeof s);
        });
        on(500, "CalculateStandardUserSystemClockDifferenceByUser", [](IpcCtx& ctx) { ctx.push<int64_t>(0); });
        on(501, "CalculateSpanBetween", [](IpcCtx& ctx) {
            // two ClockSnapshots in input buffers: the span between their steady clocks, in nanoseconds
            ClockSnapshot a{}, b{};
            const Buffer ba = ctx.in_buffer(0), bb = ctx.in_buffer(1);
            std::memcpy(&a, ctx.m + ba.addr, std::min<size_t>(sizeof a, ba.size));
            std::memcpy(&b, ctx.m + bb.addr, std::min<size_t>(sizeof b, bb.size));
            ctx.push<int64_t>((b.steady.time_point - a.steady.time_point) * 1'000'000'000);
        });
    }
};

ServiceRegistrar reg_u("time:u", [] { return std::make_shared<StaticService>("time:u"); });
ServiceRegistrar reg_a("time:a", [] { return std::make_shared<StaticService>("time:a"); });
ServiceRegistrar reg_s("time:s", [] { return std::make_shared<StaticService>("time:s"); });

}  // namespace

}  // namespace zn
