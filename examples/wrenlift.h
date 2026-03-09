/*
 * wrenlift.h — C header for the WrenLift embedding API.
 *
 * This is ABI-compatible with wren.h from wren-lang/wren, so existing
 * C/C++ embeddings can switch to WrenLift by relinking.
 */
#ifndef WRENLIFT_H
#define WRENLIFT_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef struct WrenVM WrenVM;
typedef struct WrenHandle WrenHandle;

/* Enums */
typedef enum {
    WREN_RESULT_SUCCESS       = 0,
    WREN_RESULT_COMPILE_ERROR = 1,
    WREN_RESULT_RUNTIME_ERROR = 2,
} WrenInterpretResult;

typedef enum {
    WREN_ERROR_COMPILE    = 0,
    WREN_ERROR_RUNTIME    = 1,
    WREN_ERROR_STACK_TRACE = 2,
} WrenErrorType;

typedef enum {
    WREN_TYPE_BOOL    = 0,
    WREN_TYPE_NUM     = 1,
    WREN_TYPE_FOREIGN = 2,
    WREN_TYPE_LIST    = 3,
    WREN_TYPE_MAP     = 4,
    WREN_TYPE_NULL    = 5,
    WREN_TYPE_STRING  = 6,
    WREN_TYPE_UNKNOWN = 7,
} WrenType;

/* Callback types */
typedef void (*WrenWriteFn)(WrenVM* vm, const char* text);
typedef void (*WrenErrorFn)(WrenVM* vm, WrenErrorType type,
                            const char* module, int line,
                            const char* message);
typedef void (*WrenForeignMethodFn)(WrenVM* vm);
typedef void (*WrenFinalizerFn)(void* data);
typedef const char* (*WrenResolveModuleFn)(WrenVM* vm,
                                          const char* importer,
                                          const char* name);

typedef struct {
    const char* source;
    void (*onComplete)(WrenVM*, const char*, void*);
    void* userData;
} WrenLoadModuleResult;

typedef WrenLoadModuleResult (*WrenLoadModuleFn)(WrenVM* vm, const char* name);

typedef WrenForeignMethodFn (*WrenBindForeignMethodFn)(
    WrenVM* vm, const char* module, const char* className,
    bool isStatic, const char* signature);

typedef struct {
    WrenForeignMethodFn allocate;
    WrenFinalizerFn     finalize;
} WrenForeignClassMethods;

typedef WrenForeignClassMethods (*WrenBindForeignClassFn)(
    WrenVM* vm, const char* module, const char* className);

/* Configuration */
typedef struct {
    void*                    reallocateFn;
    WrenResolveModuleFn      resolveModuleFn;
    WrenLoadModuleFn         loadModuleFn;
    WrenBindForeignMethodFn  bindForeignMethodFn;
    WrenBindForeignClassFn   bindForeignClassFn;
    WrenWriteFn              writeFn;
    WrenErrorFn              errorFn;
    size_t                   initialHeapSize;
    size_t                   minHeapSize;
    int                      heapGrowthPercent;
    void*                    userData;
} WrenConfiguration;

/* ---- Core API ---- */
int               wrenGetVersionNumber(void);
void              wrenInitConfiguration(WrenConfiguration* config);
WrenVM*           wrenNewVM(const WrenConfiguration* config);
void              wrenFreeVM(WrenVM* vm);
void              wrenCollectGarbage(WrenVM* vm);

WrenInterpretResult wrenInterpret(WrenVM* vm, const char* module,
                                  const char* source);

WrenHandle*       wrenMakeCallHandle(WrenVM* vm, const char* signature);
WrenInterpretResult wrenCall(WrenVM* vm, WrenHandle* method);
void              wrenReleaseHandle(WrenVM* vm, WrenHandle* handle);

/* ---- Slot API ---- */
int               wrenGetSlotCount(WrenVM* vm);
void              wrenEnsureSlots(WrenVM* vm, int numSlots);
WrenType          wrenGetSlotType(WrenVM* vm, int slot);

bool              wrenGetSlotBool(WrenVM* vm, int slot);
double            wrenGetSlotDouble(WrenVM* vm, int slot);
const char*       wrenGetSlotString(WrenVM* vm, int slot);
const char*       wrenGetSlotBytes(WrenVM* vm, int slot, int* length);
void*             wrenGetSlotForeign(WrenVM* vm, int slot);
WrenHandle*       wrenGetSlotHandle(WrenVM* vm, int slot);

void              wrenSetSlotBool(WrenVM* vm, int slot, bool value);
void              wrenSetSlotDouble(WrenVM* vm, int slot, double value);
void              wrenSetSlotNull(WrenVM* vm, int slot);
void              wrenSetSlotString(WrenVM* vm, int slot, const char* text);
void              wrenSetSlotBytes(WrenVM* vm, int slot,
                                   const char* bytes, int length);
void              wrenSetSlotHandle(WrenVM* vm, int slot, WrenHandle* handle);
void              wrenSetSlotNewList(WrenVM* vm, int slot);
void              wrenSetSlotNewMap(WrenVM* vm, int slot);
void*             wrenSetSlotNewForeign(WrenVM* vm, int slot,
                                        int classSlot, size_t size);

/* ---- List API ---- */
int               wrenGetListCount(WrenVM* vm, int slot);
void              wrenGetListElement(WrenVM* vm, int listSlot,
                                     int index, int elementSlot);
void              wrenSetListElement(WrenVM* vm, int listSlot,
                                     int index, int elementSlot);
void              wrenInsertInList(WrenVM* vm, int listSlot,
                                   int index, int elementSlot);

/* ---- Map API ---- */
int               wrenGetMapCount(WrenVM* vm, int slot);
bool              wrenGetMapContainsKey(WrenVM* vm, int mapSlot, int keySlot);
void              wrenGetMapValue(WrenVM* vm, int mapSlot,
                                  int keySlot, int valueSlot);
void              wrenSetMapValue(WrenVM* vm, int mapSlot,
                                  int keySlot, int valueSlot);
void              wrenRemoveMapValue(WrenVM* vm, int mapSlot,
                                     int keySlot, int removedValueSlot);

/* ---- Module / Variable API ---- */
bool              wrenHasModule(WrenVM* vm, const char* module);
bool              wrenHasVariable(WrenVM* vm, const char* module,
                                  const char* name);
void              wrenGetVariable(WrenVM* vm, const char* module,
                                  const char* name, int slot);

/* ---- Misc ---- */
void              wrenAbortFiber(WrenVM* vm, int slot);
void*             wrenGetUserData(WrenVM* vm);
void              wrenSetUserData(WrenVM* vm, void* userData);

#ifdef __cplusplus
}
#endif

#endif /* WRENLIFT_H */
