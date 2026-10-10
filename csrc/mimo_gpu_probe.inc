/* MiMo device microbench probe (mimo/m4-attribution), TEST builds only: included by q3_lookahead4_exl3.c under
 * Q3LD_INJECT (the host compiles that define into the plugin's test binaries alone; served and profile objects never
 * see this file). Times MLX's own command buffers with Metal's GPUStartTime / GPUEndTime without MLX internals:
 * -commit of the class MLX's buffers dispatch to (found on a buffer of the system device's own queue) is replaced by
 * one that, while armed, records the buffer's host commit time and adds a completed handler (a hand-built stack
 * block: no -fblocks dependency) storing the GPU times. Rows are seconds on the mach-absolute host clock, as Metal
 * reports them. The CPU self-test hooks a fake class hierarchy (no Metal object). */
#include <dlfcn.h>
#include <objc/message.h>
#include <pthread.h>
#include <objc/runtime.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <time.h>

#define MGP_MAX_HOOKS 4
typedef void (*mgp_commit_fn)(id, SEL);
static Class mgp_classes[MGP_MAX_HOOKS];
static mgp_commit_fn mgp_origs[MGP_MAX_HOOKS];
static int mgp_nhooks = 0;
static double *mgp_rows = 0;
static uint32_t mgp_cap = 0;
static atomic_uint mgp_committed;
static atomic_uint mgp_completed;
static atomic_uint mgp_dropped;
static atomic_int mgp_armed;
/* mimo/gpu-warm: record only the arming thread's commits (the warm engine commits its own queue's chunks from its
 * thread; `mimo_gpu_probe_arm_thread`). `mimo_gpu_probe_arm` records every thread's, as before. */
static atomic_int mgp_thread_only;
static pthread_t mgp_thread;

/* The block ABI (Block_private.h): a stack block, no copy/dispose helpers; Metal copies it (_Block_copy memmoves
 * descriptor->size bytes, the captured row pointer with it). */
struct mgp_block_descriptor {
    unsigned long reserved;
    unsigned long size;
};
struct mgp_block {
    void *isa;
    int flags;
    int reserved;
    void (*invoke)(struct mgp_block *, id);
    const struct mgp_block_descriptor *descriptor;
    double *row;
};
extern void *_NSConcreteStackBlock[32];
static const struct mgp_block_descriptor mgp_descriptor = {0, sizeof(struct mgp_block)};

static double mgp_now_s(void) { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

static void mgp_done(struct mgp_block *b, id cb) {
    b->row[1] = ((double (*)(id, SEL))objc_msgSend)(cb, sel_registerName("GPUStartTime"));
    b->row[2] = ((double (*)(id, SEL))objc_msgSend)(cb, sel_registerName("GPUEndTime"));
    atomic_fetch_add(&mgp_completed, 1u);
}

static mgp_commit_fn mgp_original(id self) {
    for (Class k = object_getClass(self); k != Nil; k = class_getSuperclass(k))
        for (int i = 0; i < mgp_nhooks; i++)
            if (mgp_classes[i] == k) return mgp_origs[i];
    return 0;
}

static void mgp_commit(id self, SEL sel) {
    mgp_commit_fn orig = mgp_original(self);
    if (orig == 0) abort(); /* dispatch reached the override through a hooked class */
    if (atomic_load(&mgp_armed) && (!atomic_load(&mgp_thread_only) || pthread_equal(pthread_self(), mgp_thread))) {
        uint32_t i = atomic_fetch_add(&mgp_committed, 1u);
        if (i < mgp_cap) {
            double *r = mgp_rows + 3 * (size_t)i;
            r[0] = mgp_now_s();
            r[1] = 0;
            r[2] = 0;
            struct mgp_block blk = {_NSConcreteStackBlock, 0, 0, mgp_done, &mgp_descriptor, r};
            ((void (*)(id, SEL, void *))objc_msgSend)(self, sel_registerName("addCompletedHandler:"), &blk);
        } else {
            atomic_fetch_add(&mgp_dropped, 1u);
        }
    }
    orig(self, sel);
}

/* Hooks the class owning the -commit that dispatch on `cls` resolves to (once per class). 0, or -1 when no class of
 * the chain implements -commit, -2 when the hook table is full. */
int32_t mimo_gpu_probe_hook_class(void *cls) {
    SEL commit = sel_registerName("commit");
    for (Class k = (Class)cls; k != Nil; k = class_getSuperclass(k)) {
        unsigned n = 0;
        Method *ms = class_copyMethodList(k, &n);
        Method own = 0;
        for (unsigned j = 0; j < n; j++)
            if (method_getName(ms[j]) == commit) own = ms[j];
        free(ms);
        if (own == 0) continue;
        for (int i = 0; i < mgp_nhooks; i++)
            if (mgp_classes[i] == k) return 0;
        if (mgp_nhooks == MGP_MAX_HOOKS) return -2;
        mgp_classes[mgp_nhooks] = k;
        mgp_origs[mgp_nhooks] = (mgp_commit_fn)method_getImplementation(own);
        mgp_nhooks++;
        method_setImplementation(own, (IMP)mgp_commit);
        return 0;
    }
    return -1;
}

/* The real hook: a command buffer of the system default device's own queue names the class MLX's buffers share
 * (MLX: queue->commandBufferWithUnretainedReferences() on the same device). Touches Metal: lock-gated callers only.
 * 0, -3 no Metal device, -4 no buffer, or mimo_gpu_probe_hook_class's code. */
int32_t mimo_gpu_probe_install(void) {
    void *sym = dlsym(RTLD_DEFAULT, "MTLCreateSystemDefaultDevice");
    if (sym == 0) return -3;
    id dev = ((id (*)(void))sym)();
    if (dev == nil) return -3;
    id queue = ((id (*)(id, SEL))objc_msgSend)(dev, sel_registerName("newCommandQueue"));
    if (queue == nil) return -4;
    id cb = ((id (*)(id, SEL))objc_msgSend)(queue, sel_registerName("commandBufferWithUnretainedReferences"));
    if (cb == nil) return -4;
    return mimo_gpu_probe_hook_class((void *)object_getClass(cb));
}

/* rows: double[3 * cap] (host commit, GPU start, GPU end; s). Resets the counts and arms. */
void mimo_gpu_probe_arm(double *rows, uint32_t cap) {
    mgp_rows = rows;
    mgp_cap = cap;
    atomic_store(&mgp_committed, 0u);
    atomic_store(&mgp_completed, 0u);
    atomic_store(&mgp_dropped, 0u);
    atomic_store(&mgp_thread_only, 0);
    atomic_store(&mgp_armed, 1);
}

/* As mimo_gpu_probe_arm, recording only the calling thread's commits (mimo/gpu-warm's bench: the warm thread's chunks
 * are not the wave's). */
void mimo_gpu_probe_arm_thread(double *rows, uint32_t cap) {
    atomic_store(&mgp_armed, 0);
    mgp_thread = pthread_self();
    mgp_rows = rows;
    mgp_cap = cap;
    atomic_store(&mgp_committed, 0u);
    atomic_store(&mgp_completed, 0u);
    atomic_store(&mgp_dropped, 0u);
    atomic_store(&mgp_thread_only, 1);
    atomic_store(&mgp_armed, 1);
}

void mimo_gpu_probe_disarm(void) { atomic_store(&mgp_armed, 0); }

/* committed (while armed, recorded or dropped), completed handlers, dropped (past cap), hooked classes */
void mimo_gpu_probe_counts(uint32_t out[4]) {
    out[0] = atomic_load(&mgp_committed);
    out[1] = atomic_load(&mgp_completed);
    out[2] = atomic_load(&mgp_dropped);
    out[3] = (uint32_t)mgp_nhooks;
}

/* ---- CPU self-test: a fake buffer hierarchy (FakeBase implements -commit, -addCompletedHandler: runs the block at
 * once, GPUStartTime 1.5 s, GPUEndTime 2.25 s; FakeSub inherits them). No Metal object. 0 or the failed check. */
static int mgp_fake_commits = 0;
static void mgp_fake_commit(id self, SEL sel) {
    (void)self;
    (void)sel;
    mgp_fake_commits++;
}
static void mgp_fake_add(id self, SEL sel, void *block) {
    (void)sel;
    struct mgp_block *b = (struct mgp_block *)block;
    b->invoke(b, self);
}
static double mgp_fake_start(id self, SEL sel) {
    (void)self;
    (void)sel;
    return 1.5;
}
static double mgp_fake_end(id self, SEL sel) {
    (void)self;
    (void)sel;
    return 2.25;
}

int32_t mimo_gpu_probe_selftest(void) {
    Class base = objc_getClass("MimoGpuProbeFakeBase");
    if (base == Nil) {
        base = objc_allocateClassPair(objc_getClass("NSObject"), "MimoGpuProbeFakeBase", 0);
        if (base == Nil) return 1;
        class_addMethod(base, sel_registerName("commit"), (IMP)mgp_fake_commit, "v@:");
        class_addMethod(base, sel_registerName("addCompletedHandler:"), (IMP)mgp_fake_add, "v@:@?");
        class_addMethod(base, sel_registerName("GPUStartTime"), (IMP)mgp_fake_start, "d@:");
        class_addMethod(base, sel_registerName("GPUEndTime"), (IMP)mgp_fake_end, "d@:");
        objc_registerClassPair(base);
        Class sub = objc_allocateClassPair(base, "MimoGpuProbeFakeSub", 0);
        if (sub == Nil) return 2;
        objc_registerClassPair(sub);
    }
    Class sub = objc_getClass("MimoGpuProbeFakeSub");
    if (mimo_gpu_probe_hook_class((void *)sub) != 0) return 3;
    if (mimo_gpu_probe_hook_class((void *)sub) != 0 || mgp_nhooks < 1) return 4; /* idempotent */
    id obj = class_createInstance(sub, 0);
    if (obj == nil) return 5;
    double rows[3 * 2];
    int before = mgp_fake_commits;
    ((void (*)(id, SEL))objc_msgSend)(obj, sel_registerName("commit")); /* disarmed: not recorded */
    mimo_gpu_probe_arm(rows, 2);
    double t0 = mgp_now_s();
    for (int i = 0; i < 3; i++) ((void (*)(id, SEL))objc_msgSend)(obj, sel_registerName("commit"));
    mimo_gpu_probe_disarm();
    ((void (*)(id, SEL))objc_msgSend)(obj, sel_registerName("commit"));
    uint32_t c[4];
    mimo_gpu_probe_counts(c);
    int rc = 0;
    if (mgp_fake_commits - before != 5) rc = 6;           /* every call reached the original once */
    else if (c[0] != 3 || c[1] != 2 || c[2] != 1) rc = 7; /* 3 armed commits: 2 rows, 1 dropped past cap */
    else if (rows[1] != 1.5 || rows[2] != 2.25 || rows[4] != 1.5 || rows[5] != 2.25) rc = 8;
    else if (rows[0] < t0 || rows[3] < rows[0]) rc = 9;
    object_dispose(obj);
    return rc;
}
