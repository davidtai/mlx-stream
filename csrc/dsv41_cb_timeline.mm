// PROFILE builds only (see dsv41_cb_timeline.h). MLX exposes no hook on its command buffers' commit (CommandEncoder
// commits buffer_ from queue_->commandBufferWithUnretainedReferences() through metal-cpp's -commit), so -commit is
// overridden on every class of a live buffer's chain (object_getClass on the instance, then its superclasses) that
// implements it itself, each class's own implementation kept as its original. A later buffer whose class chain holds
// no hooked class is hooked the same way (`dsv41tl_rehook`). One override serves every hooked class: the outermost
// call records the buffer (nested -commit calls a subclass's implementation makes into its superclass's run the next
// original along the chain), adds a completed handler for GPUStartTime / GPUEndTime, then calls the original.
// Diagnostics count every override entry, the tagged ones and the buffers of another queue. Non-ARC Objective-C++.
#import <Metal/Metal.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <stdlib.h>
#include <atomic>
#include <string.h>
#include <time.h>

#include "dsv41_cb_timeline.h"

typedef void (*CommitImp)(id, SEL);

#define TL_MAX_CLASSES 16
static Class g_classes[TL_MAX_CLASSES];
static CommitImp g_origs[TL_MAX_CLASSES];
static std::atomic<int> g_nclasses{0};
static id g_queue = nil;
static Dsv41tlRow *g_rows = NULL;
static uint32_t g_cap = 0;
static std::atomic<uint32_t> g_committed{0};
static std::atomic<uint32_t> g_completed{0};
static std::atomic<uint32_t> g_dropped{0};
static std::atomic<uint64_t> g_tag{0};
static std::atomic<uint64_t> g_calls{0};
static std::atomic<uint64_t> g_tagged{0};
static std::atomic<uint64_t> g_other_queue{0};
/* The thread's current outer call: the object whose -commit is running and how deep along its chain. A -commit of
 * another object made from inside it starts at depth 0 (its own outermost call, recorded). */
static thread_local int t_depth = 0;
static thread_local id t_self = nil;
/* install / rehook run on one thread (the inference thread): hook_chain is not reentrant. Plain statics: the first
 * hook_chain (the install) sets them before any other call can reach hook_chain. */
static pthread_t g_owner;
static int g_owned = 0;

uint64_t dsv41tl_now(void) {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

static int hooked_index(Class c) {
  int n = g_nclasses.load(std::memory_order_acquire);
  for (int i = 0; i < n; i++) if (g_classes[i] == c) return i;
  return -1;
}

/* The original for this call: the depth-th hooked class along self's chain (depth 0 = the class dispatch found). */
static CommitImp original_for(id self, int depth) {
  for (Class c = object_getClass(self); c != Nil; c = class_getSuperclass(c)) {
    int i = hooked_index(c);
    if (i < 0) continue;
    if (depth == 0) return g_origs[i];
    depth--;
  }
  return NULL;
}

static void kl_close(uint32_t row);

static void tl_commit(id self, SEL sel) {
  int depth = self == t_self ? t_depth : 0;
  CommitImp orig = original_for(self, depth);
  if (orig == NULL) abort();   /* dispatch reached the override through a hooked class: the original exists */
  if (depth == 0) {
    g_calls.fetch_add(1, std::memory_order_relaxed);
    uint64_t tag = g_tag.load(std::memory_order_relaxed);
    if (tag != 0) {
      g_tagged.fetch_add(1, std::memory_order_relaxed);
      int same = [(id<MTLCommandBuffer>)self commandQueue] == g_queue;
      if (!same) g_other_queue.fetch_add(1, std::memory_order_relaxed);
      uint32_t i = g_committed.fetch_add(1);
      kl_close(i < g_cap ? i : 0xFFFFFFFEu);
      if (i < g_cap) {
        Dsv41tlRow *r = &g_rows[i];
        r->tag = tag;
        r->queue = (uint64_t)same;
        r->host_commit = dsv41tl_now();
        [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> cb) {
          r->gpu_start = (uint64_t)(cb.GPUStartTime * 1e9);
          r->gpu_end = (uint64_t)(cb.GPUEndTime * 1e9);
          r->status = (uint64_t)cb.status;
          r->host_done = dsv41tl_now();
          g_completed.fetch_add(1, std::memory_order_release);
        }];
      } else {
        g_dropped.fetch_add(1);
      }
    } else {
      kl_close(0xFFFFFFFFu);
    }
  }
  id saved_self = t_self;
  int saved_depth = t_depth;
  t_self = self;
  t_depth = depth + 1;
  orig(self, sel);
  t_self = saved_self;
  t_depth = saved_depth;
}

/* Every class of obj's chain that implements -commit itself, not yet hooked: hooked. Returns the classes added, or
 * -2 when the chain implements no -commit, -3 when the table is full. */
#define TL_MAX_SEEN 32
static Class g_seen[TL_MAX_SEEN];
static int g_nseen = 0;

static int hook_chain(id obj) {
  if (!g_owned) {
    g_owner = pthread_self();
    g_owned = 1;
  } else if (!pthread_equal(g_owner, pthread_self())) {
    return -4;   /* not the thread that installed: hook_chain is not reentrant */
  }
  Class dyn = object_getClass(obj);
  for (int i = 0; i < g_nseen; i++) if (g_seen[i] == dyn) return 0;   /* this dynamic class's chain is done */
  int added = 0;
  int found = 0;
  for (Class c = object_getClass(obj); c != Nil; c = class_getSuperclass(c)) {
    unsigned n = 0;
    Method *ms = class_copyMethodList(c, &n);
    Method own = NULL;
    for (unsigned k = 0; k < n; k++) if (method_getName(ms[k]) == @selector(commit)) own = ms[k];
    free(ms);
    if (own == NULL) continue;
    found = 1;
    if (hooked_index(c) >= 0) continue;
    int i = g_nclasses.load();
    if (i >= TL_MAX_CLASSES) return -3;
    g_classes[i] = c;
    g_origs[i] = (CommitImp)method_getImplementation(own);
    g_nclasses.store(i + 1, std::memory_order_release);
    method_setImplementation(own, (IMP)tl_commit);
    added++;
  }
  if (!found) return -2;
  if (g_nseen < TL_MAX_SEEN) g_seen[g_nseen++] = dyn;
  return added;
}

int dsv41tl_install(void *cmdbuf, Dsv41tlRow *rows, uint32_t cap) {
  if (cmdbuf == NULL || rows == NULL || cap == 0) return -1;
  id buf = (id)cmdbuf;
  int r = hook_chain(buf);
  if (r < 0) return r;
  g_tag.store(0);
  g_queue = [(id<MTLCommandBuffer>)buf commandQueue];
  g_rows = rows;
  g_cap = cap;
  g_committed.store(0);
  g_completed.store(0);
  g_dropped.store(0);
  g_calls.store(0);
  g_tagged.store(0);
  g_other_queue.store(0);
  return 0;
}

int dsv41tl_rehook(void *cmdbuf) {
  if (cmdbuf == NULL) return -1;
  return hook_chain((id)cmdbuf);
}

void dsv41tl_tag(uint64_t tag) {
  g_tag.store(tag, std::memory_order_relaxed);
}

void dsv41tl_counts(uint32_t out[3]) {
  out[0] = g_committed.load();
  out[1] = g_completed.load(std::memory_order_acquire);
  out[2] = g_dropped.load();
}

void dsv41tl_hook_stats(uint64_t out[4]) {
  out[0] = g_calls.load();
  out[1] = g_tagged.load();
  out[2] = g_other_queue.load();
  out[3] = (uint64_t)g_nclasses.load();
}

size_t dsv41tl_hooked_names(char *out, size_t cap) {
  size_t n = 0;
  int k = g_nclasses.load();
  for (int i = 0; i < k && n + 2 < cap; i++) {
    const char *name = class_getName(g_classes[i]);
    size_t len = strlen(name);
    if (n + len + 2 >= cap) break;
    if (i > 0) out[n++] = ',';
    memcpy(out + n, name, len);
    n += len;
  }
  if (cap > 0) out[n < cap ? n : cap - 1] = 0;
  return n;
}

// ── the kernel log (PROFILE builds; dsv41kl_hook once, dsv41kl_arm per phase) ──
// Every compute dispatch the arming thread encodes (MLX's inference thread): its kernel's function name, its grid and
// threadgroup, and the row of the command buffer that carried it (the outermost -commit above closes the thread's
// pending dispatches with its row). The names come from the pipelines' creation on the device's class
// (-newComputePipelineStateWithFunction:error: and -newComputePipelineStateWithDescriptor:options:reflection:error:),
// the kernel from -setComputePipelineState: and the dispatches from -dispatchThreadgroups:threadsPerThreadgroup: and
// -dispatchThreads:threadsPerThreadgroup: on the compute encoder's class; each override calls the original it replaced.
// A pipeline made before dsv41kl_hook is named "?". Storage is static: past the cap a dispatch is counted, not logged.
typedef id (*KlNewFnImp)(id, SEL, id, NSError **);
typedef id (*KlNewDescImp)(id, SEL, id, MTLPipelineOption, MTLAutoreleasedComputePipelineReflection *, NSError **);
typedef void (*KlSetImp)(id, SEL, id);
typedef void (*KlDispatchImp)(id, SEL, MTLSize, MTLSize);

#define KL_CAP (1u << 21)
#define KL_NAMES (1u << 18)
#define KL_SLOTS (1u << 13)
static Dsv41klDispatch g_kl[KL_CAP];
static char g_kl_names[KL_NAMES];   /* "?" at 0 (set by the hook) */
static uint32_t g_kl_names_len = 2;
static uintptr_t g_kl_slot_key[KL_SLOTS];
static uint32_t g_kl_slot_name[KL_SLOTS];
static uint32_t g_kl_pipes = 0;
static pthread_mutex_t g_kl_mu = PTHREAD_MUTEX_INITIALIZER;
static KlNewFnImp g_kl_new_fn = NULL;
static KlNewDescImp g_kl_new_desc = NULL;
static KlSetImp g_kl_set = NULL;
static KlDispatchImp g_kl_tg = NULL;
static KlDispatchImp g_kl_th = NULL;
static std::atomic<int> g_kl_on{0};
static pthread_t g_kl_owner;
static std::atomic<uint32_t> g_kl_n{0};
static std::atomic<uint32_t> g_kl_dropped{0};
static uint32_t g_kl_pending = 0;   /* the owner's first dispatch not yet closed by a commit */
static thread_local uint32_t t_kl_name = 0;

static uint32_t kl_slot(uintptr_t key) {   /* caller holds g_kl_mu */
  uint32_t h = (uint32_t)((key >> 4) * 2654435761u) & (KL_SLOTS - 1);
  while (g_kl_slot_key[h] != 0 && g_kl_slot_key[h] != key) h = (h + 1) & (KL_SLOTS - 1);
  return h;
}

static void kl_name_pipeline(id pipe, NSString *name) {
  if (pipe == nil) return;
  const char *utf = name ? [name UTF8String] : NULL;
  pthread_mutex_lock(&g_kl_mu);
  uint32_t h = kl_slot((uintptr_t)pipe);
  if (g_kl_slot_key[h] == 0 && g_kl_pipes < KL_SLOTS / 2) {
    uint32_t off = 0;
    size_t len = utf ? strlen(utf) : 0;
    if (len > 0 && g_kl_names_len + len + 1 <= KL_NAMES) {
      off = g_kl_names_len;
      memcpy(g_kl_names + off, utf, len + 1);
      g_kl_names_len += (uint32_t)len + 1;
    }
    g_kl_slot_key[h] = (uintptr_t)pipe;
    g_kl_slot_name[h] = off;
    g_kl_pipes++;
  } else if (g_kl_slot_key[h] == (uintptr_t)pipe && utf) {
    /* a pipeline object reused at the same address after a release: its new name */
    size_t len = strlen(utf);
    if (g_kl_names_len + len + 1 <= KL_NAMES) {
      g_kl_slot_name[h] = g_kl_names_len;
      memcpy(g_kl_names + g_kl_names_len, utf, len + 1);
      g_kl_names_len += (uint32_t)len + 1;
    }
  }
  pthread_mutex_unlock(&g_kl_mu);
}

static id kl_new_fn(id self, SEL sel, id fn, NSError **err) {
  id p = g_kl_new_fn(self, sel, fn, err);
  kl_name_pipeline(p, fn ? [(id<MTLFunction>)fn name] : nil);
  return p;
}

static id kl_new_desc(id self, SEL sel, id desc, MTLPipelineOption opt, MTLAutoreleasedComputePipelineReflection *refl, NSError **err) {
  id p = g_kl_new_desc(self, sel, desc, opt, refl, err);
  id<MTLFunction> fn = desc ? [(MTLComputePipelineDescriptor *)desc computeFunction] : nil;
  kl_name_pipeline(p, fn ? [fn name] : nil);
  return p;
}

/* The name lookup runs only while the log is armed (outside it the override costs one load): a pipeline set just before
 * the arm and dispatched after it is named by the previous set's name. */
static void kl_set(id self, SEL sel, id pipe) {
  if (g_kl_on.load(std::memory_order_relaxed)) {
    pthread_mutex_lock(&g_kl_mu);
    uint32_t h = kl_slot((uintptr_t)pipe);
    t_kl_name = g_kl_slot_key[h] == (uintptr_t)pipe ? g_kl_slot_name[h] : 0;
    pthread_mutex_unlock(&g_kl_mu);
  }
  g_kl_set(self, sel, pipe);
}

static void kl_record(MTLSize grid, MTLSize group) {
  if (!g_kl_on.load(std::memory_order_relaxed) || !pthread_equal(g_kl_owner, pthread_self())) return;
  uint32_t i = g_kl_n.fetch_add(1, std::memory_order_relaxed);
  if (i >= KL_CAP) {
    g_kl_dropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  Dsv41klDispatch *d = &g_kl[i];
  d->row = 0xFFFFFFFDu;   /* pending: no commit closed it (yet) */
  d->name = t_kl_name;
  d->grid[0] = (uint32_t)grid.width; d->grid[1] = (uint32_t)grid.height; d->grid[2] = (uint32_t)grid.depth;
  d->group[0] = (uint32_t)group.width; d->group[1] = (uint32_t)group.height; d->group[2] = (uint32_t)group.depth;
}

static void kl_tg(id self, SEL sel, MTLSize grid, MTLSize group) {
  kl_record(grid, group);
  g_kl_tg(self, sel, grid, group);
}

static void kl_th(id self, SEL sel, MTLSize grid, MTLSize group) {
  /* dispatchThreads: the grid in threads; logged as threadgroups (rounded up) so both kinds read alike */
  MTLSize tg = MTLSizeMake(group.width ? (grid.width + group.width - 1) / group.width : 0,
                           group.height ? (grid.height + group.height - 1) / group.height : 0,
                           group.depth ? (grid.depth + group.depth - 1) / group.depth : 0);
  kl_record(tg, group);
  g_kl_th(self, sel, grid, group);
}

/* The outermost -commit on the owner thread: its pending dispatches carried by `row` (0xFFFFFFFF untagged, 0xFFFFFFFE
 * past the rows' cap). */
static void kl_close(uint32_t row) {
  if (!g_kl_on.load(std::memory_order_relaxed) || !pthread_equal(g_kl_owner, pthread_self())) return;
  uint32_t n = g_kl_n.load(std::memory_order_relaxed);
  if (n > KL_CAP) n = KL_CAP;
  for (uint32_t k = g_kl_pending; k < n; k++) g_kl[k].row = row;
  g_kl_pending = n;
}

/* The original is stored before the override is installed, so a concurrent call never sees a null original. */
static int kl_swap(Class cls, SEL sel, IMP imp, void **orig) {
  Method m = class_getInstanceMethod(cls, sel);
  if (m == NULL) return -1;
  *orig = (void *)method_getImplementation(m);
  method_setImplementation(m, imp);
  return 0;
}

int dsv41kl_hook(void) {
  static int done = 1;
  if (done <= 0) return done;
  g_kl_names[0] = '?';
  g_kl_names[1] = 0;
  int rc = 0;
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) return done = -1;
    Class dc = object_getClass(dev);
    if (kl_swap(dc, @selector(newComputePipelineStateWithFunction:error:), (IMP)kl_new_fn, (void **)&g_kl_new_fn) != 0) rc = -2;
    if (rc == 0 && kl_swap(dc, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), (IMP)kl_new_desc, (void **)&g_kl_new_desc) != 0) rc = -3;
    /* the compute encoder's class, from a throwaway encoder of the kind MLX makes (concurrent dispatch; never committed) */
    id<MTLCommandQueue> q = rc == 0 ? [dev newCommandQueue] : nil;
    id<MTLCommandBuffer> cb = q ? [q commandBuffer] : nil;
    id<MTLComputeCommandEncoder> enc = cb ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : nil;
    if (rc == 0 && enc == nil) rc = -4;
    if (rc == 0) {
      Class ec = object_getClass(enc);
      [enc endEncoding];
      if (kl_swap(ec, @selector(setComputePipelineState:), (IMP)kl_set, (void **)&g_kl_set) != 0) rc = -5;
      if (rc == 0 && kl_swap(ec, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)kl_tg, (void **)&g_kl_tg) != 0) rc = -6;
      if (rc == 0 && kl_swap(ec, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)kl_th, (void **)&g_kl_th) != 0) rc = -7;
    }
    [q release];
    [dev release];
  }
  return done = rc;
}

void dsv41kl_arm(int on) {
  g_kl_on.store(0, std::memory_order_relaxed);
  if (!on) return;
  g_kl_owner = pthread_self();
  g_kl_n.store(0);
  g_kl_dropped.store(0);
  g_kl_pending = 0;
  t_kl_name = 0;
  g_kl_on.store(1, std::memory_order_release);
}

const Dsv41klDispatch *dsv41kl_rows(uint32_t out[4]) {
  uint32_t n = g_kl_n.load();
  out[0] = n < KL_CAP ? n : KL_CAP;
  out[1] = g_kl_dropped.load();
  out[2] = g_kl_names_len;
  out[3] = g_kl_pipes;
  return g_kl;
}

const char *dsv41kl_names(void) {
  return g_kl_names;
}

// ── the host self-test: a fake buffer hierarchy (no Metal object) ──
// FakeBase implements -commit (counts), -commandQueue, -addCompletedHandler: (runs the block now), GPUStartTime 1 s,
// GPUEndTime 2 s, status 4; FakeSub overrides -commit (counts, then [super commit]); FakeSib inherits FakeBase's;
// FakeOther is an unrelated root with its own -commit. Returns 0, or the number of the first failed check.
static int st_base = 0, st_sub = 0, st_other = 0;
static id st_queue = nil;
static void fb_commit(id, SEL) { st_base++; }
static id fb_queue(id, SEL) { return st_queue; }
static void fb_handler(id self, SEL, void (^block)(id)) { block(self); }
static double fb_start(id, SEL) { return 1.0; }
static double fb_end(id, SEL) { return 2.0; }
static NSUInteger fb_status(id, SEL) { return 4; }
static void fs_commit(id self, SEL sel) {
  st_sub++;
  struct objc_super sup = { self, class_getSuperclass(object_getClass(self)) };
  ((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel);
}
static void fo_commit(id, SEL) { st_other++; }
static int st_nest = 0;
static id st_inner = nil;
static void fn_commit(id self, SEL sel) {   /* commits another buffer from inside its own commit, then its super's */
  st_nest++;
  [st_inner commit];
  struct objc_super sup = { self, class_getSuperclass(object_getClass(self)) };
  ((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(&sup, sel);
}

static int selftest_body(Dsv41tlRow *rows, uint32_t cap);

static Class st_class(const char *name, Class super) {
  Class c = objc_getClass(name);
  if (c) return c;
  c = objc_allocateClassPair(super, name, 0);
  return c;
}

int dsv41tl_selftest(Dsv41tlRow *rows, uint32_t cap) {
  static int done = -1;   /* once per process: the hooks it installs stay */
  if (done >= 0) return done;
  Class base = objc_getClass("Dsv41tlFakeBase");
  if (base == Nil) {
    base = st_class("Dsv41tlFakeBase", objc_getClass("NSObject"));
    class_addMethod(base, @selector(commit), (IMP)fb_commit, "v@:");
    class_addMethod(base, @selector(commandQueue), (IMP)fb_queue, "@@:");
    class_addMethod(base, @selector(addCompletedHandler:), (IMP)fb_handler, "v@:@?");
    class_addMethod(base, @selector(GPUStartTime), (IMP)fb_start, "d@:");
    class_addMethod(base, @selector(GPUEndTime), (IMP)fb_end, "d@:");
    class_addMethod(base, @selector(status), (IMP)fb_status, "Q@:");
    objc_registerClassPair(base);
    Class sub = st_class("Dsv41tlFakeSub", base);
    class_addMethod(sub, @selector(commit), (IMP)fs_commit, "v@:");
    objc_registerClassPair(sub);
    Class sib = st_class("Dsv41tlFakeSib", base);
    objc_registerClassPair(sib);
    Class other = st_class("Dsv41tlFakeOther", objc_getClass("NSObject"));
    class_addMethod(other, @selector(commit), (IMP)fo_commit, "v@:");
    class_addMethod(other, @selector(commandQueue), (IMP)fb_queue, "@@:");
    class_addMethod(other, @selector(addCompletedHandler:), (IMP)fb_handler, "v@:@?");
    class_addMethod(other, @selector(GPUStartTime), (IMP)fb_start, "d@:");
    class_addMethod(other, @selector(GPUEndTime), (IMP)fb_end, "d@:");
    class_addMethod(other, @selector(status), (IMP)fb_status, "Q@:");
    objc_registerClassPair(other);
    Class nest = st_class("Dsv41tlFakeNest", base);
    class_addMethod(nest, @selector(commit), (IMP)fn_commit, "v@:");
    objc_registerClassPair(nest);
  }
  done = selftest_body(rows, cap);
  return done;
}

static int selftest_body(Dsv41tlRow *rows, uint32_t cap) {
  st_queue = (id)objc_getClass("Dsv41tlFakeBase");
  id sub = [[objc_getClass("Dsv41tlFakeSub") alloc] init];
  id sib = [[objc_getClass("Dsv41tlFakeSib") alloc] init];
  id other = [[objc_getClass("Dsv41tlFakeOther") alloc] init];
  st_base = st_sub = st_other = 0;
  if (dsv41tl_install((void *)sub, rows, cap) != 0) return 1;
  uint64_t st[4];
  // untagged: the originals run, nothing recorded, the entry counted
  [sub commit];
  dsv41tl_hook_stats(st);
  if (st_sub != 1 || st_base != 1 || st[0] != 1 || g_committed.load() != 0) return 2;
  // tagged: one row for one commit (the subclass's [super commit] runs FakeBase's original, not a second record)
  dsv41tl_tag(7);
  [sub commit];
  if (st_sub != 2 || st_base != 2 || g_committed.load() != 1) return 3;
  if (rows[0].tag != 7 || rows[0].queue != 1 || rows[0].gpu_start != 1000000000ull || rows[0].gpu_end != 2000000000ull || rows[0].status != 4) return 4;
  // a sibling class inheriting the hooked base's -commit is recorded through it
  [sib commit];
  if (st_base != 3 || g_committed.load() != 2) return 5;
  // an unrelated class is not, until a buffer of it is hooked
  [other commit];
  if (st_other != 1 || g_committed.load() != 2) return 6;
  if (dsv41tl_rehook((void *)other) != 1) return 7;
  [other commit];
  if (st_other != 2 || g_committed.load() != 3) return 8;
  dsv41tl_hook_stats(st);
  if (st[0] != 4 || st[1] != 3 || st[2] != 0 || st[3] < 3) return 9;
  // a commit of another buffer made from inside a commit is its own outermost call: recorded, its original run
  id nest = [[objc_getClass("Dsv41tlFakeNest") alloc] init];
  st_inner = sib;
  if (dsv41tl_rehook((void *)nest) != 1) return 10;
  int base0 = st_base;
  [nest commit];
  if (st_nest != 1 || st_base != base0 + 2 || g_committed.load() != 5) return 11;
  [nest release];
  dsv41tl_tag(0);
  [sub release];
  [sib release];
  [other release];
  return 0;
}
